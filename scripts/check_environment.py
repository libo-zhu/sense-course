"""Validate native dependencies and the patched MMD against the original code."""
import importlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mac_models import JDA


def main():
    (ROOT / "outputs").mkdir(exist_ok=True)
    versions = {}
    for name in ["torch", "torchvision", "numpy", "scipy", "h5py", "mat73", "cv2",
                 "sklearn", "matplotlib", "tqdm", "gdown"]:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "import OK")
    original_path = ROOT / "references/original_jda.py"
    spec = importlib.util.spec_from_file_location("original_jda", original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(17)
    source = torch.randn(8, 64, dtype=torch.float64, requires_grad=True)
    target = torch.randn(8, 64, dtype=torch.float64)
    reference = original.MMD_loss()(source, target)
    ref_grad = torch.autograd.grad(reference, source)[0]
    source2 = source.detach().clone().requires_grad_()
    actual = JDA(1, 1).mmd_loss(source2, target)
    actual_grad = torch.autograd.grad(actual, source2)[0]
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual_grad, ref_grad, atol=1e-10, rtol=1e-10)
    for values in [torch.zeros(4, 16), torch.ones(4, 16)]:
        values.requires_grad_()
        loss = JDA(1, 1).mmd_loss(values, values.detach())
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(values.grad).all()
    report = {"status": "passed", "python": sys.version, "machine": platform.machine(),
              "versions": versions, "mps_built": torch.backends.mps.is_built(),
              "mps_available": torch.backends.mps.is_available(),
              "mmd_original_equivalence": "value and gradients passed (float64, 1e-10)",
              "mmd_zero_bandwidth": "finite loss and gradients passed"}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
