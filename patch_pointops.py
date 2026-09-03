"""CrossTooth expects pointops.queryandgroup() to also return the neighbour indices (same patch as the notebook)."""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
old = """    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat"""
new = """    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1), idx # (m, nsample, 3+c), (m, nsample)
    else:
        return grouped_feat, idx"""
if new in src:
    print("pointops.py already patched")
else:
    assert old in src, "queryandgroup block not found in " + str(path)
    path.write_text(src.replace(old, new))
    print("pointops.py patched")
