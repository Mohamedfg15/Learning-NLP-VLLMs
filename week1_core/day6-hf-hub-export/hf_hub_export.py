import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from huggingface_hub import create_repo

CONFIG = {
    "base_model": "distilbert-base-uncased",
    "dataset": "nyu-mll/glue",
    "dataset_config": "sst2",
    "num_labels": 2,
    "max_length": 128,
    "batch_size": 16,
    "epochs": 3,
    "lr_full": 2e-5,
    "lr_lora": 2e-4,
    "weight_decay": 0.01,
    "seed": 42,
    "full_dir": "./exports/full_model",
    "lora_dir": "./exports/lora_r16",
    "full_repo": "MohamedSayed112/sst2-distilbert-full-ft",
    "lora_repo": "MohamedSayed112/sst2-distilbert-LoRA",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_lin", "v_lin"],
    "modules_to_save": ["classifier"],
}


def save_metadata():
    full_meta = {
        "method": "full_finetuning",
        "base_model": CONFIG["base_model"],
        "dataset": f"{CONFIG['dataset']}/{CONFIG['dataset_config']}",
        "epochs": CONFIG["epochs"],
        "lr": CONFIG["lr_full"],
        "metrics": {"accuracy": 0.9197, "macro_f1": 0.9197},
    }
    lora_meta = {
        "method": "lora_r16",
        "base_model": CONFIG["base_model"],
        "dataset": f"{CONFIG['dataset']}/{CONFIG['dataset_config']}",
        "lora_r": CONFIG["lora_r"],
        "lora_alpha": CONFIG["lora_alpha"],
        "target_modules": CONFIG["target_modules"],
        "epochs": CONFIG["epochs"],
        "lr": CONFIG["lr_lora"],
        "metrics": {"accuracy": 0.9002, "macro_f1": 0.9001},
    }

    for folder, payload in [(CONFIG["full_dir"], full_meta), (CONFIG["lora_dir"], lora_meta)]:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)


def push_full():
    create_repo(repo_id=CONFIG["full_repo"], repo_type="model", private=False, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["full_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["full_dir"])
    model.push_to_hub(CONFIG["full_repo"], commit_message="Upload full fine-tuned DistilBERT")
    tokenizer.push_to_hub(CONFIG["full_repo"], commit_message="Upload tokenizer")


def push_lora():
    create_repo(repo_id=CONFIG["lora_repo"], repo_type="model", private=False, exist_ok=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"], num_labels=CONFIG["num_labels"]
    )
    lora_model = PeftModel.from_pretrained(base_model, CONFIG["lora_dir"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["lora_dir"])
    lora_model.push_to_hub(CONFIG["lora_repo"], commit_message="Upload LoRA adapter")
    tokenizer.push_to_hub(CONFIG["lora_repo"], commit_message="Upload tokenizer")


if __name__ == "__main__":
    save_metadata()
    # push_full()
    # push_lora()