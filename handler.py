"""
handler.py  --  Runpod Serverless worker for CrossTooth (CVPR 2025)
"3D Dental Model Segmentation with Geometrical Boundary Preserving"

Request  (job["input"]):
  stl_base64        base64 of the scan (binary/ASCII STL; may be gzip-compressed)   -- or --
  stl_url           URL to download the scan from (STL/PLY/OBJ)
  jaw               "upper" | "lower"  (only changes tooth names / FDI numbers in the reply)
  flip_z, flip_y    true/false to force the orientation, omit for auto-detection
  target_faces      decimation target (default 15999 -> the model sees 16000 points)
  seed              random seed for the point shuffle (default 1); change it to get another "draw"
  n_runs            test-time ensemble: run the network on n_runs shuffles and majority-vote (default 5)
  output_frame      "original" (default, aligned with the input STL) | "model" (rotated frame)
  ascii_ply         write an ASCII PLY instead of binary (default false)
  compress_output   gzip the PLY before base64 (default false)
  return_decimated  also return the segmented ~16k-face mesh, model frame (default false)

Reply:
  ply_base64 (or ply_url when BUCKET_ENDPOINT_URL is configured), ply_gzipped,
  n_faces, n_vertices, labels {palette, names, fdi, face_counts}, orientation, timings_ms, warnings
"""
import base64
import gzip
import io
import os
import sys
import time
import traceback
import tempfile
import argparse
from urllib.parse import urlparse

import numpy as np

REPO_DIR = os.environ.get("CROSSTOOTH_DIR", "/app/CrossTooth_CVPR2025")
WEIGHTS = os.environ.get("CROSSTOOTH_WEIGHTS", os.path.join(REPO_DIR, "models", "PTv1", "point_best_model.pth"))
NUM_POINTS = 16000
SEED = 1
MAX_INPUT_BYTES = int(os.environ.get("MAX_INPUT_BYTES", 200 * 1024 * 1024))

sys.path.insert(0, REPO_DIR)                     # models/, dataset/, utils.py of the repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                     # noqa: E402
import vedo                                      # noqa: E402
import runpod                                    # noqa: E402
import requests                                  # noqa: E402

from models.PTv1.point_transformer_seg import PointTransformerSeg38   # noqa: E402
from dataset.data import ToothData                                    # noqa: E402
from utils import label2color_lower, label2color_upper, FDI2color     # noqa: E402
import crosstooth_pipeline as pipe                                    # noqa: E402

# --------------------------------------------------------------------------- #
# Model (loaded once per worker)
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# label 0 = gum, 1..16 = teeth; the checkpoint was exported with the "lower" palette for both jaws
PALETTE = np.array([[125, 125, 125]] + [list(label2color_lower[i][2]) for i in range(1, 17)], dtype=np.uint8)
NAME_TO_FDI = {v[1]: k for k, v in FDI2color.items()}


def _label_legend(jaw):
    table = label2color_upper if jaw == "upper" else label2color_lower
    names = {0: "gum"}
    fdi = {}
    for i in range(1, 17):
        n = table[i][1]                                   # e.g. "UL1", "LR6"
        names[i] = n if jaw else n[1:]                    # "L1" if jaw unknown
        if jaw:
            fdi[i] = NAME_TO_FDI.get(n)
    return names, fdi


def _load_model():
    model = PointTransformerSeg38(in_channels=6, num_classes=17 + 2, pretrain=False,
                                  add_cbl=False, enable_pic_feat=False)
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


print(f"[init] device={DEVICE} {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}", flush=True)
t_init = time.time()
MODEL = _load_model()
DATASET = ToothData(argparse.Namespace(num_points=NUM_POINTS, sample_points=NUM_POINTS))
print(f"[init] model loaded in {time.time() - t_init:.1f}s", flush=True)


@torch.no_grad()
def predict_labels(ply_path, seed=SEED, n_runs=1):
    """
    Same maths as predict.py:  mesh -> (face centre, face normal) x 16000 -> PointTransformerSeg38.

    The original code shuffles the points with an unseeded np.random.permutation, and the
    network's farthest-point sampling depends on that order, so predict.py gives slightly
    different results on every run.  Here the shuffle is seeded (reproducible) and, with
    n_runs > 1, the network is run on n_runs different shuffles and every face takes the
    majority label (test-time ensemble).

    Returns labels (n,), face centres (n,3), face vertex indices (n,3) for the predicted faces
    (padding rows dropped), the mesh points, and the per-face agreement (fraction of runs that
    voted for the winning label).  Coordinates are those of the PLY (model frame).
    """
    n_runs = max(1, int(n_runs))
    votes = {}                                   # face (tuple of 3 vertex ids) -> [count per label]
    for r in range(n_runs):
        np.random.seed(int(seed) + r)
        torch.manual_seed(int(seed) + r)
        pointcloud, point_coords, face_info = DATASET.get_by_name(ply_path)
        pointcloud = pointcloud.unsqueeze(0).to(DEVICE).permute(0, 2, 1).contiguous()   # (1, 6, N)
        seg_logits, _edge = MODEL(pointcloud)                                             # (1, 19, N)
        pred = torch.softmax(seg_logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        pred[pred >= 17] = 0                                                              # extra classes -> gum
        face_info = np.asarray(face_info).astype(np.int64)
        valid = ~np.all(face_info == 0, axis=1)                                           # padding rows
        for f, lab in zip(map(tuple, face_info[valid]), pred[valid]):
            v = votes.setdefault(f, np.zeros(17, np.int32))
            v[lab] += 1

    faces = np.array(list(votes.keys()), dtype=np.int64)
    counts = np.stack(list(votes.values()))                                                # (n, 17)
    labels = counts.argmax(axis=1).astype(np.int64)                                       # ties -> lowest label
    agreement = counts.max(axis=1) / counts.sum(axis=1)
    point_coords = np.asarray(point_coords, dtype=np.float64)
    centers = point_coords[faces].mean(axis=1)
    return labels, centers, faces, point_coords, agreement


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
MESH_EXT = (".stl", ".ply", ".obj", ".vtk", ".vtp", ".off")


def _fetch_input(inp, tmpdir):
    """Returns the path of the mesh file written in tmpdir."""
    if inp.get("stl_base64"):
        raw = inp["stl_base64"]
        if isinstance(raw, str) and raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        data = base64.b64decode("".join(str(raw).split()))
        ext = "." + str(inp.get("input_format", "stl")).lower().lstrip(".")
    elif inp.get("stl_url"):
        url = str(inp["stl_url"])
        r = requests.get(url, timeout=int(inp.get("download_timeout", 120)), stream=True)
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(1024 * 1024):
            buf.write(chunk)
            if buf.tell() > MAX_INPUT_BYTES:
                raise ValueError(f"input larger than {MAX_INPUT_BYTES} bytes")
        data = buf.getvalue()
        ext = os.path.splitext(urlparse(url).path)[1].lower()
        if inp.get("input_format"):
            ext = "." + str(inp["input_format"]).lower().lstrip(".")
        if ext not in MESH_EXT:
            ext = ".stl"
    else:
        raise ValueError("provide 'stl_base64' or 'stl_url'")

    if data[:2] == b"\x1f\x8b":                                  # gzip magic
        data = gzip.decompress(data)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"input larger than {MAX_INPUT_BYTES} bytes")
    if len(data) < 84:
        raise ValueError("input file is empty or too small to be a mesh")
    path = os.path.join(tmpdir, "input" + ext)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _bool(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "si", "sí", "y")
    return bool(v)


def _tri_bool(v):
    """None -> auto, otherwise True/False."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto", "null", "none")):
        return None
    return _bool(v, None)


def _jaw(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("upper", "superior", "maxilar", "maxilla", "up", "u"):
        return "upper"
    if s in ("lower", "inferior", "mandibula", "mandíbula", "mandible", "low", "l"):
        return "lower"
    raise ValueError("jaw must be 'upper' or 'lower'")


def _package_file(path, name, compress, tmpdir):
    """Return the file inline (base64) or upload it to the configured bucket."""
    if os.environ.get("BUCKET_ENDPOINT_URL"):
        from runpod.serverless.utils import rp_upload
        url = rp_upload.upload_file_to_bucket(name, path, bucket_name=os.environ.get("BUCKET_NAME"))
        return {"url": url, "size_bytes": os.path.getsize(path)}
    with open(path, "rb") as f:
        data = f.read()
    size = len(data)
    if compress:
        data = gzip.compress(data, 6)
    return {"base64": base64.b64encode(data).decode("ascii"), "gzipped": bool(compress), "size_bytes": size}


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
def handler(job):
    inp = job.get("input") or {}
    timings, warnings = {}, []
    t_all = time.time()
    try:
        jaw = _jaw(inp.get("jaw"))
        if jaw is None:
            warnings.append("'jaw' not given: tooth names are generic (L1..R8), no FDI numbers")
        flip_z = _tri_bool(inp.get("flip_z"))
        flip_y = _tri_bool(inp.get("flip_y"))
        target_faces = int(inp.get("target_faces", 15999))
        if not (1000 <= target_faces <= NUM_POINTS):
            raise ValueError(f"target_faces must be between 1000 and {NUM_POINTS}")
        seed = int(inp.get("seed", SEED))
        n_runs = int(inp.get("n_runs", 5))
        if not (1 <= n_runs <= 20):
            raise ValueError("n_runs must be between 1 and 20")
        output_frame = str(inp.get("output_frame", "original")).lower()
        if output_frame not in ("original", "model", "rotated"):
            raise ValueError("output_frame must be 'original' or 'model'")
        ascii_ply = _bool(inp.get("ascii_ply"), False)
        compress = _bool(inp.get("compress_output"), False)
        return_decimated = _bool(inp.get("return_decimated"), False)

        with tempfile.TemporaryDirectory() as tmp:
            # 1. input ------------------------------------------------------
            t = time.time()
            in_path = _fetch_input(inp, tmp)
            pts, faces = pipe.load_mesh(in_path)
            timings["load"] = time.time() - t
            if len(faces) < 1000:
                warnings.append(f"only {len(faces)} faces: is this a full-arch scan?")

            # 2. orientation ("la vuelta") ------------------------------------
            t = time.time()
            T, orient = pipe.orientation_matrix(pts, faces, flip_z=flip_z, flip_y=flip_y)
            if "warning" in orient.get("flip_z_details", {}):
                warnings.append(orient["flip_z_details"]["warning"])
            p_model = pts @ T
            timings["orient"] = time.time() - t

            # 3. decimation ("la decimada") -----------------------------------
            t = time.time()
            dec = pipe.decimate_mesh(p_model, faces, target_faces=target_faces, max_faces=NUM_POINTS)
            dec_path = os.path.join(tmp, "decimated.ply")
            dec.write(dec_path)
            timings["decimate"] = time.time() - t

            # 4. CrossTooth inference ------------------------------------------
            t = time.time()
            labels_low, centers_low, faces_low, pts_low, agreement = predict_labels(dec_path, seed=seed, n_runs=n_runs)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            timings["inference"] = time.time() - t
            n_teeth = int(len(np.unique(labels_low[labels_low > 0])))
            if n_teeth < 5:
                warnings.append(f"only {n_teeth} teeth detected: check orientation (flip_z / flip_y) and jaw")

            # 5. back to full resolution (KNN k=3 on face centres) ----------------
            t = time.time()
            centers_hd = p_model[faces].mean(axis=1)
            labels_hd = pipe.project_labels(centers_low, labels_low, centers_hd, k=3)
            timings["knn"] = time.time() - t

            # 6. output PLY ----------------------------------------------------------
            t = time.time()
            out_pts = pts if output_frame == "original" else p_model
            base = os.path.splitext(os.path.basename(str(inp.get("name", "scan"))))[0] or "scan"
            ply_path = os.path.join(tmp, f"{base}_seg.ply")
            pipe.write_coloured_ply(ply_path, out_pts, faces, PALETTE[labels_hd], binary=not ascii_ply)
            packed = _package_file(ply_path, f"{base}_seg_{job.get('id', 'local')}.ply", compress, tmp)

            result = {
                "n_vertices": int(len(pts)),
                "n_faces": int(len(faces)),
                "n_faces_decimated": int(dec.ncells),
                "n_faces_predicted": int(len(labels_low)),
                "seed": seed,
                "n_runs": n_runs,
                "run_agreement": float(agreement.mean()),
                "output_frame": "original" if output_frame == "original" else "model",
                "ply_binary": not ascii_ply,
            }
            if "url" in packed:
                result["ply_url"] = packed["url"]
            else:
                result["ply_base64"] = packed["base64"]
                result["ply_gzipped"] = packed["gzipped"]
            result["ply_size_bytes"] = packed["size_bytes"]

            if return_decimated:
                dec_seg_path = os.path.join(tmp, f"{base}_16k_seg.ply")
                pipe.write_coloured_ply(dec_seg_path, pts_low, faces_low, PALETTE[labels_low], binary=not ascii_ply)
                packed_dec = _package_file(dec_seg_path, f"{base}_16k_seg_{job.get('id', 'local')}.ply", compress, tmp)
                result["decimated"] = {"n_faces": int(len(faces_low)), "frame": "model", **packed_dec}
            timings["write"] = time.time() - t

        # legend -----------------------------------------------------------------------
        names, fdi = _label_legend(jaw)
        counts = np.bincount(labels_hd, minlength=17)
        present = [int(i) for i in np.nonzero(counts)[0]]
        result["jaw"] = jaw
        result["labels"] = {
            "present": present,
            "n_teeth": int(len([i for i in present if i > 0])),
            "palette": {str(i): PALETTE[i].tolist() for i in range(17)},
            "names": {str(i): names[i] for i in range(17)},
            "fdi": {str(i): fdi[i] for i in fdi} if fdi else None,
            "face_counts": {str(i): int(counts[i]) for i in present},
            "run_agreement": {str(i): float(agreement[labels_low == i].mean()) for i in np.unique(labels_low)},
        }
        result["orientation"] = orient
        timings["total"] = time.time() - t_all
        result["timings_ms"] = {k: (int(v * 1000) if isinstance(v, float) else v) for k, v in timings.items()}
        result["warnings"] = warnings
        return result

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}", "timings_ms": {k: int(v * 1000) for k, v in timings.items()
                                                                   if isinstance(v, float)}}


# --------------------------------------------------------------------------- #
# Warm-up (also validates pointops + weights at boot, so a broken image fails fast)
# --------------------------------------------------------------------------- #
if os.environ.get("WARMUP", "1") == "1" and DEVICE.type == "cuda":
    sample = os.path.join(REPO_DIR, "YBSESUN6_upper.ply")
    if os.path.exists(sample):
        t_w = time.time()
        lab, _, _, _, _ = predict_labels(sample)
        print(f"[init] warm-up ok: {len(np.unique(lab[lab > 0]))} teeth on the sample scan "
              f"in {time.time() - t_w:.1f}s", flush=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})