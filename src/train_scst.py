import torch.nn.functional

from scst import RLTrainer
import pytorch_lightning as pl
from input_parser import create_parser
from scst_constants import ModelParams, NodeParams, GenerateParams, ULParams
from deepspeed_config import DEEPSPEED_CONFIG, DEEPSPEED_CONFIG_TRIAL
from scst_callbacks import MetricSaver
from utils import create_weight_vector
from scoring import (
    calculate_rouge_reward,
    fkgl_reward,
    lexical_reward,
    cosine_similarity_reward,
)
from pytorch_lightning.loggers import TensorBoardLogger

# torch.autograd.set_detect_anomaly(True)

# from hf_data import

LIGHTNING_MODULE_PATH = "scst_models/bart-large-xsum-rl-1689463.ckpt"
parser = create_parser()
args = parser.parse_args()

use_ul = True if int(args.use_ul) == 1 else False
only_rl = True if int(args.only_rl) == 1 else False
use_deepspeed = True if int(args.use_deepspeed) == 1 else False
fast_run = True if int(args.fast_run) == 1 else False
# Create mode config
model_cfg = ModelParams(
    model_name=args.model_path,
    max_epochs=args.epochs,
    start=args.start,
    stop=args.stop,
    train_batch_size=args.train_batch_size,
    use_rl=only_rl,
    use_ul=use_ul,
    use_deepspeed=use_deepspeed,
    generate_strat=args.generate_strat,
)
# print(args.n_samples)

# Create Node Config
node_cfg = NodeParams(max_gpus=args.max_gpus, max_cpus=args.max_cpus)

# Create sentence generation Config here
generate_cfg = GenerateParams(
    top_p=args.top_p,
    top_k=args.top_k,
    no_copy_ngram=args.no_copy_ngram,
    no_repeat_ngram=args.no_repeat_ngram,
)
ul_config = ULParams(use_ul)
rewards = args.rewards.split(",")

REWARD_MAP = {
    "fkgl": {"func": fkgl_reward, "w": args.fkgl_w},
    "simi": {"func": cosine_similarity_reward, "w": args.simi_w},
    "lexical": {"func": lexical_reward, "w": args.lexical_w},
}
REWARD_LIST = [REWARD_MAP[r] for r in rewards]

# Create lightning module
model = RLTrainer(
    model_cfg=model_cfg,
    node_cfg=node_cfg,
    generate_cfg=generate_cfg,
    ul_config=ul_config,
    reward_list=REWARD_LIST,
)
model = model.load_from_checkpoint(LIGHTNING_MODULE_PATH)

if use_deepspeed is True:
    # Update deepspeed config to take account n_gpus
    DEEPSPEED_CONFIG_TRIAL["scheduler"]["params"]["total_num_steps"] = (
        len(model.train_ds["ds"]) // model_cfg.train_batch_size * args.max_gpus
    )

    # Create deepspeed plugin
    plugin = pl.plugins.DeepSpeedPlugin(config=DEEPSPEED_CONFIG_TRIAL)
else:
    plugin = None

# Create logger
logger = TensorBoardLogger("tb_logs", name="bart-large-xsum-2000-3000")

# Checkpoint callback
mckpt = pl.callbacks.ModelCheckpoint(
    monitor="val_sample_rewards",
    mode="max",
    verbose=True,
    dirpath="scst_models/",
    filename="bart-large-xsum-rl-from-finetuned-1000-samples-{epoch}-{val_sample_rewards}",
)
progbar = pl.callbacks.RichProgressBar()
# Create metric saver
ops = MetricSaver(exp_name=args.exp_name)

# print("Starting Training")
# Create lightning trainer

trainer = pl.Trainer(
    gpus=args.max_gpus,
    precision=16,
    max_epochs=int(args.epochs),
    auto_select_gpus=True,
    strategy=plugin,
    callbacks=[mckpt, ops, progbar],
    # callbacks = [progbar, ops],
    fast_dev_run=fast_run,
    detect_anomaly=False,
    logger=logger,
)

# Fit trainer
trainer.fit(model)

# Test model
trainer.test(ckpt_path="best")
