"""Replace Git LFS pointers in a LIBERO-Pro2.0 checkout with SHA-checked payloads."""
import hashlib, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PIN = "e5500c690c1498eca1b7050350654964190cdf70"
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == PIN
names = subprocess.check_output(["git", "-C", str(root), "lfs", "ls-files", "-n"], text=True).split("\n")
def fetch(name):
    path = root / name
    head = path.read_bytes()[:200]
    if not head.startswith(b"version https://git-lfs"):
        return 0
    fields = dict(line.split(" ", 1) for line in path.read_text().splitlines())
    sha, size = fields["oid"].split(":")[1], int(fields["size"])
    url = f"https://media.githubusercontent.com/media/LIBERO-Pro-Team/LIBERO-Pro2.0/{PIN}/{name}"
    data = urllib.request.urlopen(url, timeout=120).read()
    assert len(data) == size and hashlib.sha256(data).hexdigest() == sha, name
    path.write_bytes(data)
    return size
with ThreadPoolExecutor(8) as pool:
    done = list(pool.map(fetch, [n for n in names if n]))
print("hydrated", sum(1 for d in done if d), "files", sum(done), "bytes")
