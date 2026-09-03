"""
client_example.py -- send an STL to the CrossTooth endpoint and save the segmented PLY.

    export RUNPOD_API_KEY=...   RUNPOD_ENDPOINT_ID=...
    python client_example.py upper.stl upper
    python client_example.py https://mi-servidor/lower.stl lower

The STL is gzip-compressed + base64 (a 15 MB STL becomes ~12 MB, under the 20 MB /runsync limit).
For bigger scans pass a URL instead: the worker downloads it itself.
"""
import base64
import gzip
import os
import sys
import time

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = os.environ["RUNPOD_ENDPOINT_ID"]
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def main():
    src = sys.argv[1]
    jaw = sys.argv[2] if len(sys.argv) > 2 else "upper"
    payload = {"jaw": jaw}
    if src.startswith("http"):
        payload["stl_url"] = src
        name = os.path.basename(src)
    else:
        with open(src, "rb") as f:
            payload["stl_base64"] = base64.b64encode(gzip.compress(f.read(), 6)).decode()
        name = os.path.basename(src)
    payload["name"] = name

    t0 = time.time()
    r = requests.post(f"{BASE}/runsync?wait=300000", headers=HEADERS, json={"input": payload}, timeout=600)
    r.raise_for_status()
    job = r.json()
    # /runsync returns early if the job takes longer than `wait`: poll /status
    while job.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        time.sleep(2)
        job = requests.get(f"{BASE}/status/{job['id']}", headers=HEADERS, timeout=60).json()
    print(f"status={job.get('status')}  delay={job.get('delayTime')}ms  exec={job.get('executionTime')}ms  "
          f"wall={time.time() - t0:.1f}s")

    out = job.get("output") or {}
    if job.get("status") != "COMPLETED" or "error" in out:
        print("ERROR:", out.get("error") or job.get("error") or job)
        sys.exit(1)

    if "ply_url" in out:                       # bucket configured on the endpoint
        ply = requests.get(out["ply_url"], timeout=300).content
    else:
        ply = base64.b64decode(out["ply_base64"])
        if out.get("ply_gzipped"):
            ply = gzip.decompress(ply)
    dst = os.path.splitext(name)[0] + "_seg.ply"
    with open(dst, "wb") as f:
        f.write(ply)

    print(f"saved {dst}  ({out['n_faces']} faces, {out['labels']['n_teeth']} teeth)")
    print("orientation:", {k: out["orientation"][k] for k in ("flip_z", "flip_z_method", "flip_y", "mirror_fixed")})
    print("timings_ms:", out["timings_ms"])
    if out.get("warnings"):
        print("warnings:", out["warnings"])


if __name__ == "__main__":
    main()
