import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import os
from utils import save_json


class MetricSaver(pl.Callback):
    def __init__(self, exp_name):
        self.odir = "SCST"
        self.exp_name = exp_name

    @rank_zero_only
    def on_test_epoch_end(self, trainer, module):
        metrics = module.test_results
        if not os.path.exists(f"{self.odir}/{trainer.logger.version}"):
            os.mkdir(f"{self.odir}/{trainer.logger.version}")
        fname = f"{self.odir}/{trainer.logger.version}/Epoch-{trainer.current_epoch}-{self.exp_name}-test-op.json"
        save_json(metrics, fname)
