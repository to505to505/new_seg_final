import os
import lightning
from lightning.pytorch.callbacks import Callback


class BestMetricWriter(Callback):
    """Writes best.txt in the run directory with the best monitored metric value."""

    def __init__(self, monitor: str = "metrics/val_ap_all", mode: str = "max"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1

    def _is_better(self, current, best):
        if self.mode == "max":
            return current > best
        return current < best

    def _get_run_dir(self, trainer):
        if trainer.logger is not None:
            return trainer.logger.log_dir
        return trainer.default_root_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        current = float(metrics[self.monitor])
        if self._is_better(current, self.best_value):
            self.best_value = current
            self.best_epoch = trainer.current_epoch

            run_dir = self._get_run_dir(trainer)
            os.makedirs(run_dir, exist_ok=True)
            best_path = os.path.join(run_dir, "best.txt")
            with open(best_path, "w") as f:
                f.write(f"epoch: {self.best_epoch}\n")
                f.write(f"{self.monitor}: {self.best_value:.6f}\n")
                for k, v in sorted(metrics.items()):
                    if k != self.monitor and k.startswith("metrics/"):
                        f.write(f"{k}: {float(v):.6f}\n")
