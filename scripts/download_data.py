"""Download the author's public MAT files, with resumable downloads and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import gdown

ROOT = Path(__file__).resolve().parents[1]
# Each list is X_test, X_train, Y_test, Y_train, as listed in the author's Drive.
FILES = {
    "augsburg/hsi": ["1w4XsxEcMs9-FB4VJ1IR1qRGlwb68uLUZ", "1IRA42WPLtMM_JcqpT_WxaTXJ6lv-AVA3", "16bTWepZj1RlleE5lrb4XtGL8muu9HswV", "1NHo9lX4AlQSh3w1nZDpwpe3i31kOYkQN"],
    "augsburg/sar": ["1WFwX-ENMTsMGZghseOTcOR65PcinCGSD", "1sUN7K5U_iN_L-ZBDo2MAGrsm3sLTso73", "1edZeVEgHKSIHyzC8Kv4-Ykx3Gzk11q_Z", "1oWPZqB5c-uBNwt3EppEHZCQwj6prfK3a"],
    "augsburg/dsm": ["1Tmruxf8tumNlzMh26O2BDJMNfksML1HB", "1qZJYq3x3GTRF1gDe047gbzqUYWRRKr0W", "1olq-K2mQj0v1HqoJdoY_4uh27jQEU7M5", "1LNtxoxbuR3Xiue5mYHfakYRK1rHF85I5"],
    "houston2013/hsi": ["1zSHlG8tGO_HCCmFxjo82BmoT4Dv-DIzO", "1XtUYFv_dTMS8p5Bwva1lotmUZ36mOizL", "1xmL38KjjXL34IFLVvl9W70zNrIU0hTLV", "1PKN1iuAEMR5KNGLOBceCxTeAFFpXGYlu"],
    "houston2013/ms": ["1dqgAyzGr161_F0NgGEAHJrZ-r0tpEl4r", "1RrCzRiSWT0PBazarkMhJSbVpybM4gcbO", "1oxVVRQZNDDt0bXXsRyEJnBpVKwmK705-", "1gquxoCxjUJdNiFcVtkSRcqhC6i6AK_8z"],
    "houston2013/lidar": ["1XLGWfD6miw_CV-gCKhl_-KAibOsvSjWp", "1bNptuST-lbj8xWvsaotNA-ncx8f5TOLl", "1w-qZXeCjarAkbS7ckr1zw7-JNWQT4or4", "1tV1iwUJiG7kWX3gDmZKzXDI-Dwxp1dr4"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(task):
    rel, file_id = task
    path = ROOT / "data" / "processed" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        result = gdown.download(id=file_id, output=str(path), quiet=True, resume=True)
        if result is None:
            raise RuntimeError(f"Download failed: {rel}")
    # HTML error pages must never count as data.
    with path.open("rb") as f:
        header = f.read(128)
    if b"MATLAB" not in header and not header.startswith(b"\x89HDF"):
        raise ValueError(f"Not a MATLAB data file: {path}")
    entry = {"path": str(path.relative_to(ROOT)), "drive_id": file_id,
             "bytes": path.stat().st_size, "sha256": sha256(path)}
    print(f"OK {rel} ({entry['bytes']:,} bytes)", flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["train", "full", "small-test"], default="train")
    parser.add_argument("--dataset", choices=["all", "augsburg", "houston2013"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    tasks = []
    for folder, ids in FILES.items():
        if args.dataset != "all" and not folder.startswith(args.dataset + "/"):
            continue
        modal = folder.split("/")[1]
        for suffix, file_id in zip(["X_test", "X_train", "Y_test", "Y_train"], ids):
            if args.profile == "train" and suffix.endswith("test"):
                continue
            if args.profile == "small-test" and (not suffix.endswith("test") or folder == "augsburg/hsi"):
                continue
            tasks.append((f"{folder}/{modal}_{suffix}.mat", file_id))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(download, tasks))
    out = ROOT / "data" / f"download-{args.dataset}-{args.profile}.json"
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
    print(f"Manifest: {out}")


if __name__ == "__main__":
    main()
