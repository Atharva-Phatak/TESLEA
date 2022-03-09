from dataclasses import dataclass
from utils import create_weight_vector
import torch


@dataclass
class ModelParams:
    model_name: str
    max_epochs: int
    train_batch_size: int
    start: int
    stop: int
    generate_strat: str
    use_deepspeed: bool
    use_rl: bool
    use_ul: bool
    use_deepspeed: bool
    max_target_len: int = 512
    max_input_len: int = 940
    gamma: int = 0.95
    learning_rate: float = 2e-5
    accum_steps: int = 1


@dataclass
class NodeParams:
    max_gpus: int
    max_cpus: int


@dataclass
class GenerateParams:
    top_p: float
    top_k: int
    no_repeat_ngram: int
    no_copy_ngram: int


@dataclass
class ULParams:
    use_ul: bool
    ul_softmax: bool = True
    ul_temprature: int = 2
    ul_file_path: str = "data/bart_freq_normalized_ids.txt"
    exclude_tokens: str = "4,6"
    ul_n_weights: int = -1
    ul_selective_penalty: bool = True
    ul_alpha: int = 100
