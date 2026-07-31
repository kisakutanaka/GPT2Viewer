import json

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "rinna/japanese-gpt2-xsmall"
CORPUS_PATH = "/Users/tanakakisaku/Documents/dev/ScienseMuseum/AI/GPT2Viewer/training-data/aozora/corpus.json"
OUT_DIR = "/private/tmp/claude-501/-Users-tanakakisaku-Documents-dev-ScienseMuseum-AI-GPT2Viewer/d141fa56-e564-4ce8-b0c7-a5335841e87e/scratchpad/aozora-finetuned"
BLOCK_SIZE = 512
VAL_TITLES = {"走れメロス", "藪の中"}
NUM_EPOCHS = 5

torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

data = json.load(open(CORPUS_PATH, encoding="utf-8"))
train_works = [d for d in data if d["title"] not in VAL_TITLES]
val_works = [d for d in data if d["title"] in VAL_TITLES]
print(f"train works: {len(train_works)}, val works: {len(val_works)}")


def to_blocks(works):
    ids = []
    for w in works:
        ids.extend(tokenizer.encode(w["text"]))
        ids.append(tokenizer.eos_token_id)
    blocks = [ids[i : i + BLOCK_SIZE] for i in range(0, len(ids) - BLOCK_SIZE + 1, BLOCK_SIZE)]
    return blocks


class BlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}


train_blocks = to_blocks(train_works)
val_blocks = to_blocks(val_works)
print(f"train blocks: {len(train_blocks)}, val blocks: {len(val_blocks)}")

train_dataset = BlockDataset(train_blocks)
val_dataset = BlockDataset(val_blocks)

args = TrainingArguments(
    output_dir=OUT_DIR + "-checkpoints",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_only_model=True,
    save_total_limit=NUM_EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_strategy="epoch",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"saved fine-tuned model to {OUT_DIR}")
