# TESLEA : Medical Text Simplification using Reinforcement Learning

To run the models use the below command

` python train_scst.py -h`

To reproduce results run

`python train_scst.py --epochs 30 -experiment_name <experiment-name> -bs 1 -model_path <path-to-finetuned-bart-model> -gpus 1 -cpus 6 -no_repeat_ngram 5 -no_copy_ngram 7 -only_rl 1 -generate_strat sample_scst -use_deepspeed 0 -fast_run 0 --n_start -1 --n_stop -1`

Note : Model will be released soon on huggingface hub once the paper is published. 
Note : Training script supports DDP and DeepSpeed with some minor modifications.
