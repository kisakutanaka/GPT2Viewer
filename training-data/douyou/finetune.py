import json
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "rinna/japanese-gpt2-xsmall"
CORPUS_PATH = "/Users/tanakakisaku/Documents/dev/ScienseMuseum/AI/GPT2Viewer/training-data/douyou/corpus.json"
OUT_DIR = "/private/tmp/claude-501/-Users-tanakakisaku-Documents-dev-ScienseMuseum-AI-GPT2Viewer/d141fa56-e564-4ce8-b0c7-a5335841e87e/scratchpad/douyou-extreme"
MAX_LENGTH = 256
VAL_SONGS = 5
SEED = 42
NUM_EPOCHS = 80

random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

data = json.load(open(CORPUS_PATH, encoding="utf-8"))
random.shuffle(data)
val_data = data[:VAL_SONGS]
train_data = data[VAL_SONGS:]
print(f"train songs: {len(train_data)}, val songs: {len(val_data)}")


class LyricsDataset(Dataset):
    def __init__(self, songs):
        self.examples = []
        for song in songs:
            ids = tokenizer.encode(song["lyrics"]) + [tokenizer.eos_token_id]
            ids = ids[:MAX_LENGTH]
            pad_len = MAX_LENGTH - len(ids)
            input_ids = ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * len(ids) + [0] * pad_len
            labels = ids + [-100] * pad_len
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(labels),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


train_dataset = LyricsDataset(train_data)
val_dataset = LyricsDataset(val_data)

# Extending the schedule to 80 epochs (vs the earlier 40) changes the LR trajectory at every
# epoch mark, since LR decays linearly to 0 over the *whole* schedule — "epoch 35" here is not
# the same weights as "epoch 35" in a 40-epoch run. Saving every epoch from 40 onward so we can
# check whether pushing further trades coherence for a *stronger* 童謡 flavor, or just breaks
# down further the way the earlier 25-40 sweep did.
args = TrainingArguments(
    output_dir=OUT_DIR + "-checkpoints",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_only_model=True,
    save_total_limit=40,
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
