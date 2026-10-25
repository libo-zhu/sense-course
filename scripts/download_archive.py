"""Resume the public S2FL archive with verified HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
URL = "https://drive.usercontent.google.com/download?id=1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv&export=download&confirm=t"
# Public Drive metadata and Content-Range agree on this size.
TOTAL = 512951198


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "S2FL_benchmarks.rar"
    if target.exists() and target.stat().st_size == TOTAL:
        print(f"Already downloaded: {target}")
        return
    partials = sorted(RAW.glob("S2FL_benchmarks.zip*.part"))
    prefix = partials[0] if partials else None
    offset = prefix.stat().st_size if prefix else 0
    if offset > TOTAL:
        raise ValueError("Partial archive is longer than the expected source")
    parts = RAW / "archive-parts"
    parts.mkdir(exist_ok=True)
    chunk = 32 * 1024 * 1024
    jobs = [(start, min(start + chunk, TOTAL) - 1) for start in range(offset, TOTAL, chunk)]

    def fetch(job):
        start, end = job
        path = parts / f"{start}-{end}.part"
        if not path.exists() or path.stat().st_size != end - start + 1:
            headers = path.with_suffix(".headers")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "3", "--range", f"{start}-{end}", "-D", str(headers),
                            URL, "-o", str(path)], check=True)
            if path.stat().st_size != end - start + 1:
                raise RuntimeError(f"Server did not honor byte range {job}")
        print(f"Range {start:,}-{end:,} OK", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, jobs))
    assembly = target.with_suffix(".rar.assembling")
    with assembly.open("wb") as dst:
        if prefix:
            with prefix.open("rb") as src:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        for path in paths:
            with path.open("rb") as src:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    if assembly.stat().st_size != TOTAL:
        raise RuntimeError("Archive length mismatch")
    assembly.replace(target)
    h = hashlib.sha256()
    with target.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    (RAW / "archive-source.json").write_text(json.dumps({
        "source": "https://github.com/danfenghong/ISPRS_S2FL",
        "drive_id": "1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv", "bytes": TOTAL,
        "sha256": h.hexdigest()}, indent=2) + "\n")
    print(f"Complete: {target}", flush=True)


if __name__ == "__main__":
    main()
