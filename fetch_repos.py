"""Download the two GitHub repos as tarballs (no git / apt needed at build time)."""
import io
import os
import sys
import tarfile
import urllib.request

REPOS = [
    ("https://codeload.github.com/XiShuFan/CrossTooth_CVPR2025/tar.gz/main", "CrossTooth_CVPR2025"),
    ("https://codeload.github.com/POSTECH-CVLab/point-transformer/tar.gz/master", "point-transformer"),
]

dest = sys.argv[1] if len(sys.argv) > 1 else "."
for url, name in REPOS:
    print("downloading", url, flush=True)
    data = urllib.request.urlopen(url, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        top = members[0].name.split("/")[0]           # e.g. CrossTooth_CVPR2025-main
        for m in members:                             # strip the top-level folder
            m.name = m.name[len(top) + 1:]
        members = [m for m in members if m.name]
        tar.extractall(os.path.join(dest, name), members=members)
    print(f"  -> {os.path.join(dest, name)}  ({len(data) / 1e6:.1f} MB)", flush=True)

w = os.path.join(dest, "CrossTooth_CVPR2025", "models", "PTv1", "point_best_model.pth")
assert os.path.getsize(w) > 10_000_000, "weights missing or truncated: " + w
print("weights OK", os.path.getsize(w) // 1_000_000, "MB")