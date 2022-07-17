from datasets import load_dataset
import torch
from scst_constants import ModelParams
from transformers import DataCollatorForSeq2Seq, AutoTokenizer

DATA = load_dataset(
    "csv",
    data_files={
        "train": ["data/train.csv"],
        "valid": ["data/val.csv"],
        "test": ["data/test.csv"],
    },
)
TOKENIZER = AutoTokenizer.from_pretrained("models/bart-base")


def preprocess_text(examples):
    inputs = TOKENIZER(
        examples["inputs"],
        max_length=ModelParams.max_input_len,
        truncation=True,
        padding="max_length",
    )
    model_inputs = {
        "attention_mask": inputs["attention_mask"],
        "input_ids": inputs["input_ids"],
    }
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(
            examples["targets"],
            max_length=ModelParams.max_target_len,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["target_attention_mask"] = labels["attention_mask"]
    return model_inputs


def create_data(model, split, start, stop):
    if start != -1 and stop != -1:
        data = DATA[split].select(list(range(start, stop)))
    else:
        data = DATA[split]
    tok_data = data.map(preprocess_text, batched=True)
    tok_data = tok_data.remove_columns(["inputs", "targets", "doi"])
    tok_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "target_attention_mask"],
    )
    collate_fn = DataCollatorForSeq2Seq(TOKENIZER, model=model)
    return {"ds": tok_data, "collate_fn": collate_fn}
