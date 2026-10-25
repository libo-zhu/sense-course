"""Audit dataset completeness, official splits, and low-memory patch equivalence."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mac_data import AuthorPatches, AugsburgScenePatches, CHANNELS, MatMatrix


def main():
    (ROOT / "outputs").mkdir(exist_ok=True)
    report = {"status": "passed", "datasets": {}, "augsburg_lazy_patch_equivalence": {}}
    for name, channels in CHANNELS.items():
        ds = AuthorPatches(ROOT / "data/processed", name, list(channels))
        report["datasets"][name] = {"official_train_samples": len(ds), "channels": channels,
                                  "classes": np.unique(ds.labels).tolist(),
                                  "train_class_counts": np.bincount(ds.labels).tolist()}
        if name == "augsburg":
            lazy = AugsburgScenePatches(ROOT, list(channels), split="train")
            if not np.array_equal(ds.labels, lazy.labels):
                raise AssertionError("Augsburg original/preprocessed training label order differs")
            errors = {modal: 0.0 for modal in channels}
            for index in range(len(ds)):
                a, b = ds[index], lazy[index]
                for modal in channels:
                    error = float((a[modal] - b[modal]).abs().max())
                    errors[modal] = max(error, errors[modal])
                    np.testing.assert_allclose(a[modal].numpy(), b[modal].numpy(), atol=1e-7, rtol=1e-6)
            report["augsburg_lazy_patch_equivalence"] = {"checked_patches_per_modality": len(ds),
                                                         "maximum_absolute_float32_error": errors}
            lazy.close()
            test = AugsburgScenePatches(ROOT, list(channels), split="test")
            assert len(test) == 77533
            for modal in ["sar", "dsm"]:
                labels = MatMatrix(ROOT / "data/processed/augsburg" / modal / f"{modal}_Y_test.mat")
                np.testing.assert_array_equal(test.labels, labels.labels())
                labels.close()
            report["datasets"][name]["official_test_samples"] = len(test)
            report["datasets"][name]["test_class_counts"] = np.bincount(test.labels).tolist()
            test.close()
        else:
            folder = ROOT / "data/raw/Datasets/HS-MS Houston2013"
            for suffix, split in [("TR", "train"), ("VA", "test")]:
                image = cv2.imread(str(folder / f"2013_IEEE_GRSS_DF_Contest_Samples_{suffix}.tif"), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Cannot load official {split} label image")
                labels = image.T.reshape(-1)
                labels = labels[labels > 0].astype(np.int64) - 1
                matrix = MatMatrix(ROOT / f"data/processed/houston2013/hsi/hsi_Y_{split}.mat")
                np.testing.assert_array_equal(labels, matrix.labels())
                matrix.close()
                report["datasets"][name][f"official_{split}_samples"] = len(labels)
            test = AuthorPatches(ROOT / "data/processed", name, list(channels), split="test")
            report["datasets"][name]["test_class_counts"] = np.bincount(test.labels).tolist()
            test.close()
        ds.close()
    report["raw_mat_files"] = {str(p.relative_to(ROOT)): scipy.io.whosmat(p)
                               for p in (ROOT / "data/raw/Datasets").rglob("*.mat")}
    (ROOT / "outputs/data-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
