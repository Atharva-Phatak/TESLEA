import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", "--epochs", type=int, required=True)
    parser.add_argument("-experiment_name", "--exp_name", type=str, required=True)
    parser.add_argument("-bs", "--train_batch_size", type=int, default=1)
    parser.add_argument("-model_path", "--model_path", type=str, required=True)
    parser.add_argument("-start", "--n_start", type=int, default=-1)
    parser.add_argument("-stop", "--n_stop", type=int, default=-1)
    parser.add_argument("-gpus", "--max_gpus", type=int, required=True)
    parser.add_argument("-cpus", "--max_cpus", type=int, required=True)
    parser.add_argument("-top_p", "--top_p", type=float, default=1.0)
    parser.add_argument("-top_k", "--top_k", type=int, default=0)
    parser.add_argument("-no_repeat_ngram", "--no_repeat_ngram", type=int, default=3)
    parser.add_argument("-no_copy_ngram", "--no_copy_ngram", type=int, default=4)
    parser.add_argument("-only_rl", "--only_rl", type=int)
    parser.add_argument("-use_ul", "--use_ul", type=int)
    parser.add_argument("-rewards", "--rewards", type=str, default="simi,fkgl,lexical")
    parser.add_argument(
        "-generate_strat", "--generate_strat", type=str, default="sample"
    )
    parser.add_argument("-use_deepspeed", "--use_deepspeed", type=int, default=0)
    parser.add_argument("-fast_run", "--fast_run", type=int)
    parser.add_argument("-simi-w", "--simi_w", type=int, default=1)
    parser.add_argument("-fkgl_w", "--fkgl_w", type=int, default=1)
    parser.add_argument("-lexical_w", "--lexical_w", type=int, default=1)
    return parser
