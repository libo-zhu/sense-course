"""Mac experiment runner. Training/selection never reads official test labels.

This is a documented course-project protocol, not the author's exact protocol.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from mac_data import AuthorPatches, CHANNELS, CLASSES, open_official_test
from mac_demo import assert_same_state, select_device, state_copy
from mac_models import JDA, SharedSpecificFusion, create_model

ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    "aug-hs-sar-dsm": ("augsburg", ["hsi", "sar", "dsm"]),
    "aug-hs-sar": ("augsburg", ["hsi", "sar"]),
    "hou-hs-ms": ("houston2013", ["hsi", "ms"]),
    "hou-hs-lidar": ("houston2013", ["hsi", "lidar"]),
}


def experiment_names(modalities, extended=False):
    names = [*("single-" + m for m in modalities), "early-concat", "late-mean", "teacher"]
    for m in modalities:
        names.extend(["jad-" + m, "msh-" + m])
    if extended:
        names.append("moddrop")
        for m in modalities:
            names.extend(["kl-only-" + m, "mmd-only-" + m, "msh-fixed-eta-at-test-" + m])
    return names


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def stage_seed(seed, name):
    return (seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)) % 2**31


class EarlyConcat(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.network = create_model([sum(channels)], classes)

    def forward(self, *xs):
        return self.network(torch.cat(xs, dim=1))


class MeanProbabilities(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *xs):
        return torch.stack([m(x)[0].softmax(1) for m, x in zip(self.models, xs)]).mean(0)


def predict(model, batch, modalities):
    result = model(*(batch[m] for m in modalities))
    return result[0] if isinstance(result, tuple) else result


def score(labels, predictions, classes):
    cm = confusion_matrix(labels, predictions, labels=list(range(classes)))
    support = cm.sum(axis=1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(classes, dtype=float), where=support > 0)
    n = int(cm.sum())
    oa = float(cm.trace() / n)
    expected = float(np.dot(cm.sum(0).astype(float), support.astype(float)) / n**2)
    return {"oa": oa, "aa": float(recall[support > 0].mean()),
            "kappa": (oa - expected) / (1 - expected) if expected < 1 else None,
            "samples": n, "support": support.tolist(),
            "per_class_recall": [float(r) if s else None for r, s in zip(recall, support)],
            "confusion_matrix": cm.tolist()}


@torch.no_grad()
def evaluate(model, dataset, modalities, device, classes, batch_size,
             retained=None, noise=0.0, noise_seed=1234):
    model.eval()
    ys, ps = [], []
    rng = np.random.default_rng(noise_seed)
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        labels = batch.pop("label")
        for m in modalities:
            if retained is not None and m not in retained:
                batch[m] = torch.zeros_like(batch[m])
            elif noise:
                # Same CPU-generated noise for matching shapes/order across models/devices.
                perturbation = rng.normal(0, noise, size=tuple(batch[m].shape)).astype(np.float32)
                batch[m] = (batch[m] + torch.from_numpy(perturbation)).clamp(0, 1)
            batch[m] = batch[m].to(device)
        logits = predict(model, batch, modalities)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite evaluation output")
        ys.extend(labels.tolist())
        ps.extend(logits.argmax(1).cpu().tolist())
    return score(ys, ps, classes), list(zip(ys, ps))


class Runner:
    def __init__(self, args, scenario, seed):
        self.args, self.seed, self.scenario = args, seed, scenario
        self.dataset_name, self.modalities = SCENARIOS[scenario]
        self.classes = CLASSES[self.dataset_name]
        self.device = select_device(args.device)
        self.out = args.output / scenario / f"seed-{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.data = AuthorPatches(ROOT / "data/processed", self.dataset_name, self.modalities)
        train, val = train_test_split(np.arange(len(self.data)), test_size=args.val_fraction,
                                     stratify=self.data.labels, random_state=args.split_seed)
        if args.quick:
            train, val = train[:64], val[:32]
        self.train, self.val = Subset(self.data, train), Subset(self.data, val)
        config = {"protocol_version": 1, "dataset": self.dataset_name, "modalities": self.modalities,
                  "seed": seed, "split_seed": args.split_seed, "val_fraction": args.val_fraction,
                  "epochs": args.epochs, "fusion_epochs": args.fusion_epochs,
                  "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                  "quick": args.quick, "alpha_hsi": args.alpha_hsi,
                  "alpha_other": args.alpha_other, "beta_hsi": args.beta_hsi,
                  "beta_other": args.beta_other, "selection": "validation_oa_then_ce",
                  "train_indices": train.tolist(), "validation_indices": val.tolist(),
                  "normalization": "author scene-wide band min/max; not train-only normalization"}
        config_path = self.out / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            self.data.close()
            raise ValueError(f"Configuration changed: use a NEW --output directory: {config_path}")
        write_json(config_path, config)
        self.test = None  # Open only after all training/model selection has finished.
        results_path = self.out / "results.json"
        self.rows = json.loads(results_path.read_text()) if results_path.exists() else []

    def model(self, modalities, early=False):
        channels = [CHANNELS[self.dataset_name][m] for m in modalities]
        # Author constructors hard-code their initialization seeds. Keep those weights,
        # then reset the run RNG explicitly; seeds vary split-independent training order.
        model = EarlyConcat(channels, self.classes) if early else create_model(channels, self.classes)
        seed_all(self.seed)
        return model.to(self.device)

    def fit(self, name, model, modalities, teacher=None, kd=None, moddrop=False, fusion=False):
        folder = self.out / "models" / name
        folder.mkdir(parents=True, exist_ok=True)
        epochs = self.args.fusion_epochs if fusion else self.args.epochs
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        start, best, history = 0, (-1., -float("inf")), []
        if (folder / "last.pt").exists():
            saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start, best = saved["epoch"], tuple(saved["best"])
            history = saved["history"]
        if self.args.evaluate_only and start < epochs:
            raise RuntimeError(f"Incomplete training checkpoint for --evaluate-only: {folder}")
        if start >= epochs:
            print(f"REUSE {self.scenario} seed={self.seed} {name}: {folder / 'best.pt'}", flush=True)
        frozen = {k: v for k, v in state_copy(model).items()
                  if k not in {n for n, p in model.named_parameters() if p.requires_grad}}
        teacher_before = state_copy(teacher) if teacher is not None else None
        ce = nn.CrossEntropyLoss()
        for epoch in range(start, epochs):
            epoch_seed = stage_seed(self.seed, name) + epoch
            seed_all(epoch_seed)
            generator = torch.Generator().manual_seed(epoch_seed)
            loader = DataLoader(self.train, batch_size=self.args.batch_size, shuffle=True,
                                generator=generator, num_workers=0, drop_last=False)
            model.train()
            total, seen = 0., 0
            tick = time.perf_counter()
            # Linear warmup (5 epochs) followed by cosine; explicit course protocol.
            warmup = min(5, epochs)
            factor = ((epoch + 1) / warmup if epoch < warmup else
                      0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(1, epochs - warmup))))
            for group in optimizer.param_groups:
                group["lr"] = float(self.args.lr * factor)
            for cpu_batch in loader:
                # A singleton tail is valid: the author's BN tensors retain spatial axes.
                batch = {k: v.to(self.device) for k, v in cpu_batch.items()}
                inputs = [batch[m] for m in modalities]
                if moddrop:
                    inputs = [x.clone() for x in inputs]
                    chosen = torch.randint(len(inputs), (len(batch["label"]),))
                    dropped = torch.rand(len(chosen)) < 0.5
                    for i, x in enumerate(inputs):
                        x[(chosen.eq(i) & dropped).to(self.device)] = 0
                optimizer.zero_grad(set_to_none=True)
                output = model(*inputs)
                logits = output[0] if isinstance(output, tuple) else output
                loss = ce(logits, batch["label"])
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, t_features = teacher(*(batch[m] for m in self.modalities))
                    loss = loss + kd(output[1], t_features, logits, t_logits)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{name} epoch {epoch + 1}: non-finite loss")
                loss.backward()
                if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError(f"{name}: non-finite gradient")
                optimizer.step()
                seen += len(batch["label"])
                total += loss.item() * len(batch["label"])
            val_metrics, _ = evaluate(model, self.val, modalities, self.device, self.classes,
                                      self.args.eval_batch_size)
            # CE breaks equal-OA ties; validation is clean and never uses distillation targets.
            model.eval()
            val_ce, val_seen = 0., 0
            with torch.no_grad():
                for batch in DataLoader(self.val, batch_size=self.args.eval_batch_size):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    val_ce += ce(predict(model, batch, modalities), batch["label"]).item() * len(batch["label"])
                    val_seen += len(batch["label"])
            key = (val_metrics["oa"], -val_ce / val_seen)
            record = {"epoch": epoch + 1, "train_loss": total / seen, "train_samples_seen": seen,
                      "val_oa": key[0], "val_ce": -key[1], "lr": optimizer.param_groups[0]["lr"],
                      "seconds": time.perf_counter() - tick}
            if key > best:
                best = key
                save_checkpoint(folder / "best.pt", {"model": state_copy(model), "epoch": epoch + 1,
                                                      "validation": val_metrics, "name": name})
            history.append(record)
            save_checkpoint(folder / "last.pt", {"model": state_copy(model), "optimizer": optimizer.state_dict(),
                                                  "epoch": epoch + 1, "best": list(best), "history": history})
            write_json(folder / "history.json", history)
            if epoch == start or (epoch + 1) % self.args.log_every == 0 or epoch + 1 == epochs:
                print(f"{self.scenario} seed={self.seed} {name} {epoch+1}/{epochs}: "
                      f"loss={record['train_loss']:.4f} val_OA={key[0]:.4f}", flush=True)
        if fusion:
            actual = model.state_dict()
            for key, value in frozen.items():
                if not torch.equal(value, actual[key].detach().cpu()):
                    raise AssertionError(f"Frozen fusion feature/BN changed: {key}")
        if teacher is not None:
            assert_same_state(teacher_before, teacher, "frozen teacher")
        saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        model.eval()
        return model

    def record(self, name, model, modalities, retained=None, noise=0.0):
        retained = list(modalities) if retained is None else retained
        case = name + "__keep-" + "-".join(retained) + f"__noise-{noise:g}"
        if self.device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        metrics, predictions = evaluate(model, self.test, modalities, self.device, self.classes,
                                        self.args.eval_batch_size, retained, noise)
        if self.device.type == "mps":
            torch.mps.synchronize()
        folder = self.out / "evaluation" / case
        folder.mkdir(parents=True, exist_ok=True)
        info = {"experiment": name, "retained": retained,
                "missing": [m for m in self.modalities if m not in retained],
                "noise_std": noise, "seed": self.seed, "scenario": self.scenario,
                "quick": self.args.quick, "test_partition": "official_test",
                "prediction_labels": "zero-based", "evaluation_seconds_including_data": time.perf_counter() - started,
                "parameters": sum(p.numel() for p in model.parameters()), **metrics}
        if isinstance(model, SharedSpecificFusion):
            info["eta"] = model.eta_logit.sigmoid().item()
        write_json(folder / "metrics.json", info)
        with (folder / "predictions.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["official_test_index", "label_zero_based", "prediction_zero_based"])
            indices = self.test.indices if isinstance(self.test, Subset) else range(len(self.test))
            writer.writerows((int(i), y, p) for i, (y, p) in zip(indices, predictions))
        np.savetxt(folder / "confusion_matrix.csv", metrics["confusion_matrix"], fmt="%d", delimiter=",")
        identity = (name, retained, noise)
        self.rows = [r for r in self.rows if (r["experiment"], r["retained"], r["noise_std"]) != identity]
        self.rows.append(info)
        write_json(self.out / "results.json", self.rows)
        print(f"TEST {case}: OA={metrics['oa']:.4f} AA={metrics['aa']:.4f} n={metrics['samples']}", flush=True)

    def run(self):
        allowed = experiment_names(self.modalities, self.args.preset == "extended")
        requested = self.args.only or allowed
        if any(name not in allowed for name in requested) or len(set(requested)) != len(requested):
            raise ValueError(f"--only must contain unique names from {allowed}")
        (self.out / "completed.json").unlink(missing_ok=True)
        cache = {}

        def get(name):
            if name in cache:
                return cache[name]
            if name == "teacher":
                model = self.fit(name, self.model(self.modalities), self.modalities)
                model.eval().requires_grad_(False)
                mods = self.modalities
            elif name in ("early-concat", "moddrop"):
                mods = self.modalities
                model = self.fit(name, self.model(mods, early=name == "early-concat"), mods,
                                 moddrop=name == "moddrop")
            elif name == "late-mean":
                mods = self.modalities
                model = MeanProbabilities([get("single-" + m)[0] for m in mods]).to(self.device)
            else:
                m = name.rsplit("-", 1)[1]
                mods = [m]
                if name.startswith("single-"):
                    model = self.fit(name, self.model(mods), mods)
                elif name.startswith("msh-fixed-eta-at-test-"):
                    model = copy.deepcopy(get("msh-" + m)[0])
                    model.eta_logit.data.zero_()
                elif name.startswith("msh-"):
                    student, regular = get("jad-" + m)[0], get("single-" + m)[0]
                    model = SharedSpecificFusion(copy.deepcopy(student), copy.deepcopy(regular)).to(self.device)
                    model = self.fit(name, model, mods, fusion=True)
                else:
                    teacher = get("teacher")[0]
                    a = self.args.alpha_hsi if m == "hsi" else self.args.alpha_other
                    b = self.args.beta_hsi if m == "hsi" else self.args.beta_other
                    if name.startswith("kl-only-"):
                        a = 0.
                    if name.startswith("mmd-only-"):
                        b = 0.
                    model = self.fit(name, self.model(mods), mods, teacher, JDA(a, b).to(self.device))
            cache[name] = (model, mods)
            return cache[name]

        models = [(name, *get(name)) for name in requested]
        # Official test is not even opened until every stage has finished selection.
        official = open_official_test(ROOT, self.dataset_name, self.modalities)
        try:
            if self.args.quick:
                idx = np.random.default_rng(self.args.split_seed).permutation(len(official))[:64]
                self.test = Subset(official, idx)
            else:
                self.test = official
            write_json(self.out / "runtime.json", {"device": str(self.device), "torch": torch.__version__,
                                                   "mps_available": torch.backends.mps.is_available()})
            for name, model, modalities in models:
                self.record(name, model, modalities)
                if len(modalities) > 1:
                    for count in range(1, len(modalities)):
                        for retained in itertools.combinations(modalities, count):
                            self.record(name, model, modalities, list(retained))
                if self.args.preset == "extended" and name.startswith(("single-", "jad-", "msh-")):
                    for noise in [0.05, 0.1, 0.2]:
                        self.record(name, model, modalities, noise=noise)
            write_json(self.out / "completed.json", {"status": "complete", "quick": self.args.quick,
                                                      "last_requested": requested,
                                                      "evaluation_rows": len(self.rows)})
        finally:
            official.close()
            self.data.close()


def aggregate(output):
    rows = []
    for path in sorted(output.glob("*/seed-*/results.json")):
        if (path.parent / "completed.json").exists():
            rows.extend(json.loads(path.read_text()))
    fields = ["scenario", "seed", "experiment", "retained", "missing", "noise_std", "oa", "aa", "kappa", "samples", "quick"]
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "retained": "+".join(row["retained"]), "missing": "+".join(row["missing"])})
    groups = {}
    for row in rows:
        key = (row["scenario"], row["experiment"], "+".join(row["retained"]), row["noise_std"])
        groups.setdefault(key, []).append(row)
    stats = []
    for key, group in groups.items():
        stat = dict(zip(["scenario", "experiment", "retained", "noise_std"], key))
        stat["n_seeds"], stat["seeds"] = len(group), [r["seed"] for r in group]
        for metric in ["oa", "aa", "kappa"]:
            values = [r[metric] for r in group if r[metric] is not None]
            stat[metric + "_mean"] = float(np.mean(values)) if values else None
            stat[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        stats.append(stat)
    write_json(output / "summary-mean-std.json", stats)
    if stats:
        with (output / "summary-mean-std.csv").open("w") as f:
            writer = csv.DictWriter(f, list(stats[0]))
            writer.writeheader()
            writer.writerows(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="aug-hs-sar-dsm")
    parser.add_argument("--preset", choices=["core", "extended"], default="core")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--fusion-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--alpha-hsi", type=float, default=2.)
    parser.add_argument("--beta-hsi", type=float, default=1.)
    parser.add_argument("--alpha-other", type=float, default=1.)
    parser.add_argument("--beta-other", type=float, default=5.)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="1 epoch/stage, 64 train / 32 validation / 64 test; NOT a benchmark")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--only", nargs="+", help="Run named experiments, automatically train/load their dependencies")
    parser.add_argument("--list", action="store_true", help="Print experiment scenarios without loading data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        print(json.dumps({name: {"dataset": d, "modalities": ms,
                                "experiments": experiment_names(ms, args.preset == "extended")}
                          for name, (d, ms) in SCENARIOS.items()}, ensure_ascii=False, indent=2))
        return
    if args.quick:
        args.epochs = args.fusion_epochs = 1
    if min(args.epochs, args.fusion_epochs, args.log_every, args.eval_batch_size) < 1 or args.batch_size < 2:
        parser.error("Epochs/log/eval batch must be positive; train batch >= 2")
    if not 0 < args.val_fraction < 0.5 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Need 0 < val_fraction < .5, lr > 0 and weight_decay >= 0")
    if min(args.alpha_hsi, args.alpha_other, args.beta_hsi, args.beta_other) < 0:
        parser.error("Distillation weights cannot be negative")
    if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error("Seeds must be unique nonnegative integers")
    args.output = (args.output or ROOT / "outputs" / ("experiments-quick" if args.quick else "experiments")).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for scenario in SCENARIOS if args.scenario == "all" else [args.scenario]:
        for seed in args.seeds:
            runner = Runner(args, scenario, seed)
            try:
                runner.run()
            finally:
                runner.data.close()
            aggregate(args.output)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print(f"DONE: {args.output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
