"""Real-data environment demo: teacher -> JAD student -> shared/specific fusion.

Training deliberately uses only a small subset of the official TRAINING partition.
Evaluation defaults to a held-out subset of the training split; --validation-source
official instead reads an independent official test subset. Either way, scores
only check execution; they are not a reproduction of the paper's accuracy.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from mac_data import AuthorPatches, CHANNELS, CLASSES, open_official_test
from mac_models import JDA, SharedSpecificFusion, create_model

ROOT = Path(__file__).resolve().parent


def select_device(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Use --device cpu explicitly; no silent fallback.")
    return torch.device(name)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_steps(name, model, loader, device, steps, loss_function):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.001, momentum=0.9)
    before = [p.detach().clone() for p in trainable]
    losses = []
    for step, batch in zip(range(steps), itertools.cycle(loader)):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non-finite loss")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError(f"{name}: missing or non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not any(not torch.equal(a, p.detach()) for a, p in zip(before, trainable)):
        raise RuntimeError(f"{name}: optimizer did not change any parameter")
    print(f"  {name}: {steps} updates, loss {losses[0]:.5f} -> {losses[-1]:.5f}", flush=True)
    return losses


def state_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assert_same_state(expected, model, context, exclude_fc=False):
    actual = model.state_dict()
    for name, tensor in expected.items():
        if exclude_fc and (name.startswith("fc.") or ".fc." in name):
            continue
        if not torch.equal(tensor, actual[name].detach().cpu()):
            raise AssertionError(f"{context}: frozen value changed: {name}")


def evaluate(model, loader, modalities, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(*(batch[m] for m in modalities))
            if isinstance(logits, tuple):
                logits = logits[0]
            if not torch.isfinite(logits).all():
                raise AssertionError("Inference produced non-finite logits")
            predictions = logits.argmax(1).cpu().tolist()
            rows.extend(zip(batch["label"].cpu().tolist(), predictions))
    return {"demo_accuracy": sum(y == p for y, p in rows) / len(rows), "samples": len(rows)}, rows


def run_dataset(args, name, device, out):
    modalities = args.teacher_modalities.split(",") if args.teacher_modalities else list(CHANNELS[name])
    if len(modalities) not in (2, 3) or len(set(modalities)) != len(modalities) or any(m not in CHANNELS[name] for m in modalities):
        raise ValueError(f"Invalid teacher modalities for {name}: {modalities}")
    dataset = AuthorPatches(ROOT / "data/processed", name, modalities)
    official_test = None
    try:
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=args.seed,
                                             stratify=dataset.labels)
        rng = np.random.default_rng(args.seed)
        train_idx = rng.permutation(train_idx)[:args.train_samples]
        val_idx = rng.permutation(val_idx)[:args.val_samples]
        if set(train_idx) & set(val_idx):
            raise AssertionError("Demo training and validation indices overlap")
        loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                            shuffle=True, num_workers=0, pin_memory=False)
        validation_dataset = dataset
        if args.validation_source == "official":
            official_test = open_official_test(ROOT, name, modalities)
            validation_dataset = official_test
            val_idx = rng.permutation(len(official_test))[:args.val_samples]
        val_loader = DataLoader(Subset(validation_dataset, val_idx), batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=False)
        classes = CLASSES[name]
        channels = [CHANNELS[name][m] for m in modalities]
        teacher = create_model(channels, classes).to(device)
        ce = nn.CrossEntropyLoss()
        print(f"{name}: {len(dataset)} official training patches; demo {len(train_idx)} train / {len(val_idx)} {args.validation_source}; {modalities}", flush=True)
        teacher_losses = train_steps("teacher", teacher, loader, device, args.steps,
                                     lambda b: ce(teacher(*(b[m] for m in modalities))[0], b["label"]))
        teacher.eval().requires_grad_(False)
        teacher_state = state_copy(teacher)
        results = {"modalities": modalities, "official_training_samples": len(dataset),
                   "demo_train_indices": train_idx.tolist(), "demo_evaluation_indices": val_idx.tolist(),
                   "evaluation_source": args.validation_source,
                   "teacher_losses": teacher_losses, "cases": []}
        for retained in (modalities if args.all_students else ["hsi"]):
            student = create_model([CHANNELS[name][retained]], classes).to(device)
            regular = create_model([CHANNELS[name][retained]], classes).to(device)
            kd = JDA(alpha=2.0 if retained == "hsi" else 1.0,
                     beta=1.0 if retained == "hsi" else 5.0).to(device)
            regular_losses = train_steps(f"{retained} regular", regular, loader, device, args.steps,
                                         lambda b: ce(regular(b[retained])[0], b["label"]))

            def student_loss(batch):
                with torch.no_grad():
                    teacher_logits, teacher_features = teacher(*(batch[m] for m in modalities))
                student_logits, student_features = student(batch[retained])
                return ce(student_logits, batch["label"]) + kd(student_features, teacher_features,
                                                              student_logits, teacher_logits)

            student_losses = train_steps(f"{retained} JAD student", student, loader, device,
                                         args.steps, student_loss)
            assert_same_state(teacher_state, teacher, "teacher during distillation")
            student_state, regular_state = state_copy(student), state_copy(regular)
            fusion = SharedSpecificFusion(student, regular).to(device)
            fusion_losses = train_steps(f"{retained} fusion", fusion, loader, device, args.steps,
                                        lambda b: ce(fusion(b[retained]), b["label"]))
            assert_same_state(student_state, student, "student features during fusion", exclude_fc=True)
            assert_same_state(regular_state, regular, "regular features during fusion", exclude_fc=True)
            metrics, rows = evaluate(fusion, val_loader, [retained], device)

            # The deployment checkpoint contains two single-input branches, not the teacher.
            checkpoint = out / f"{name}_{retained}_demo.pt"
            torch.save({"model": state_copy(fusion), "dataset": name,
                        "retained": retained, "purpose": "environment_smoke_test"}, checkpoint)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored = SharedSpecificFusion(create_model([CHANNELS[name][retained]], classes),
                                            create_model([CHANNELS[name][retained]], classes))
            restored.load_state_dict(loaded["model"])
            restored.to(device).eval()
            example = next(iter(val_loader))[retained].to(device)
            with torch.no_grad():
                before = fusion(example)
                after = restored(example)
            if not torch.allclose(before, after, atol=1e-5, rtol=1e-4):
                raise AssertionError("Checkpoint reload changed predictions")
            if any(k.startswith("teacher") for k in loaded["model"]):
                raise AssertionError("Teacher leaked into deployment checkpoint")
            with (out / f"{name}_{retained}_predictions.csv").open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["label_zero_based", "prediction_zero_based"])
                writer.writerows(rows)
            results["cases"].append({"retained": retained,
                                     "missing": [m for m in modalities if m != retained],
                                     "regular_losses": regular_losses, "student_losses": student_losses,
                                     "fusion_losses": fusion_losses,
                                     "eta": float(fusion.eta_logit.sigmoid().detach().cpu()),
                                     "checkpoint_reload": "passed", "frozen_features": "passed",
                                     "teacher_frozen": "passed", **metrics})
            print(f"  {retained}: missing-modality inference + checkpoint reload PASS; demo OA={metrics['demo_accuracy']:.3f} (not a benchmark)", flush=True)
        return results
    finally:
        dataset.close()
        if official_test is not None:
            official_test.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--dataset", choices=["all", "augsburg", "houston2013"], default="all")
    parser.add_argument("--teacher-modalities", help="Optional comma-separated modalities for one dataset, e.g. hsi,sar")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--validation-source", choices=["holdout", "official"], default="holdout")
    parser.add_argument("--all-students", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.train_samples < args.batch_size or args.val_samples < 1:
        parser.error("Need steps >= 1, batch >= 2, train_samples >= batch, val_samples >= 1")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"purpose": "Real-data environment smoke test, not paper accuracy reproduction",
              "platform": platform.platform(), "architecture": platform.machine(),
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "device": str(device), "mps_built": torch.backends.mps.is_built(),
              "mps_available": torch.backends.mps.is_available(), "datasets": {}}
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, ensure_ascii=False), flush=True)
    for name in (CHANNELS if args.dataset == "all" else [args.dataset]):
        report["datasets"][name] = run_dataset(args, name, device, args.output)
    if device.type == "mps":
        torch.mps.synchronize()
        report["mps_current_allocated_mib"] = torch.mps.current_allocated_memory() / 2**20
        report["mps_driver_allocated_mib"] = torch.mps.driver_allocated_memory() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mib"] = rss / (2**20 if platform.system() == "Darwin" else 1024)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "passed"
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {args.output / 'report.json'}; peak RSS={report['peak_rss_mib']:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
