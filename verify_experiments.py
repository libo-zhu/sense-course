"""Focused checks for scientific metrics and exact epoch-boundary resume on CPU."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import run_experiments as exp


def main():
    torch.set_num_threads(4)
    m = exp.score([0, 0, 1, 1], [0, 1, 1, 1], 3)
    assert m["oa"] == .75 and m["aa"] == .75 and m["kappa"] == .5
    assert m["per_class_recall"] == [.5, 1., None]
    assert np.array_equal(m["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [0, 0, 0]])
    with tempfile.TemporaryDirectory(dir=exp.ROOT / "outputs") as tmp:
        args = SimpleNamespace(device="cpu", output=Path(tmp) / "resume", split_seed=2026,
                               val_fraction=.2, quick=True, epochs=2, fusion_epochs=2,
                               batch_size=21, eval_batch_size=32, lr=.001, weight_decay=.001,
                               preset="core", alpha_hsi=2., beta_hsi=1., alpha_other=1., beta_other=5.,
                               evaluate_only=False, log_every=1)
        runner = exp.Runner(args, "aug-hs-sar-dsm", 42)
        assert not set(runner.train.indices) & set(runner.val.indices)
        original_save = exp.save_checkpoint

        def interrupt_after_first_epoch(path, value):
            original_save(path, value)
            if path.name == "last.pt" and value["epoch"] == 1:
                raise InterruptedError("Simulated interruption after a complete epoch")

        try:
            with patch.object(exp, "save_checkpoint", interrupt_after_first_epoch):
                runner.fit("single-hsi", runner.model(["hsi"]), ["hsi"])
        except InterruptedError:
            pass
        else:
            raise AssertionError("Interruption was not exercised")
        resumed = runner.fit("single-hsi", runner.model(["hsi"]), ["hsi"])
        state = torch.load(runner.out / "models/single-hsi/last.pt", weights_only=True)
        assert [x["epoch"] for x in state["history"]] == [1, 2]
        assert all(x["train_samples_seen"] == 64 for x in state["history"])
        args.output = Path(tmp) / "continuous"
        continuous = exp.Runner(args, "aug-hs-sar-dsm", 42)
        expected = continuous.fit("single-hsi", continuous.model(["hsi"]), ["hsi"])
        exp.assert_same_state(exp.state_copy(expected), resumed, "CPU resumed vs uninterrupted best")
        # Inference consumes one input for the independently loaded single-modality model.
        metrics, _ = exp.evaluate(resumed, runner.val, ["hsi"], torch.device("cpu"), 7, 32)
        assert metrics["samples"] == 32
        # Exercise the post-warmup cosine path, whose LR must stay a Python float
        # so safe weights_only=True checkpoint loading works without NumPy globals.
        args.output = Path(tmp) / "cosine"
        args.epochs = 7
        cosine = exp.Runner(args, "aug-hs-sar-dsm", 42)
        cosine.fit("single-dsm", cosine.model(["dsm"]), ["dsm"])
        last = torch.load(cosine.out / "models/single-dsm/last.pt", weights_only=True)
        assert last["epoch"] == 7 and type(last["optimizer"]["param_groups"][0]["lr"]) is float
        runner.data.close()
        continuous.data.close()
        cosine.data.close()
    print("PASS: OA/AA/kappa, absent-class handling, disjoint split, singleton tail batch, "
          "epoch-boundary optimizer resume identical to uninterrupted CPU training, checkpoint inference, "
          "post-warmup cosine checkpoint safe loading")


if __name__ == "__main__":
    main()
