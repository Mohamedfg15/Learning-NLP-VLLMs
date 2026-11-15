

import os
import time
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 1. Configuration
# =========================================================
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "dataset_name": "nyu-mll/glue",
    "dataset_config": "sst2",
    "num_labels": 2,
    "max_length": 128,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate_full": 2e-5,
    "learning_rate_lora": 2e-4,
    "weight_decay": 0.01,
    "seed": 42,
    "output_dir": "./results_day5",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_lin", "v_lin"],   
}

# =========================================================
# 2. Helper Functions
# =========================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_percentage": 100 * trainable / total if total > 0 else 0
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1
    }

# =========================================================
# 3. Data Preparation
# =========================================================
def load_and_prepare_data(tokenizer, config):
    dataset = load_dataset(config["dataset_name"], config["dataset_config"])

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=config["max_length"],
            padding=False
        )

    tokenized = dataset.map(tokenize_function, batched=True)

    # Rename label column if needed
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    tokenized = tokenized.remove_columns(
        [col for col in tokenized["train"].column_names if col not in ["input_ids", "attention_mask", "labels"]]
    )

    tokenized.set_format("torch")
    return tokenized

# =========================================================
# 4. Model Setup
# =========================================================
def create_full_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"]
    )
    return model

def create_lora_model(config):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"]
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        modules_to_save=["classifier"]
    )

    model = get_peft_model(base_model, lora_config)
    return model

# =========================================================
# 5. Training Function
# =========================================================
def train_and_evaluate(model, tokenized_datasets, tokenizer, config, method_name: str, learning_rate: float):
    print(f"\n{'='*60}")
    print(f"Starting {method_name}")
    print(f"{'='*60}")

    # Count parameters
    params_info = count_parameters(model)
    print(f"Total Parameters     : {params_info['total_params']:,}")
    print(f"Trainable Parameters : {params_info['trainable_params']:,}")
    print(f"Trainable Percentage : {params_info['trainable_percentage']:.4f}%")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(config["output_dir"], method_name.lower().replace(" ", "_")),
        learning_rate=learning_rate,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=config["seed"],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train with timing
    start_time = time.perf_counter()
    trainer.train()
    training_time = time.perf_counter() - start_time

    # Final evaluation
    eval_results = trainer.evaluate()

    results = {
        "method": method_name,
        "trainable_params": params_info["trainable_params"],
        "total_params": params_info["total_params"],
        "trainable_percentage": params_info["trainable_percentage"],
        "accuracy": eval_results["eval_accuracy"],
        "macro_f1": eval_results["eval_macro_f1"],
        "training_time_seconds": round(training_time, 2),
        "learning_rate": learning_rate,
    }

    print(f"\nResults for {method_name}:")
    print(f"Accuracy     : {results['accuracy']:.4f}")
    print(f"Macro F1     : {results['macro_f1']:.4f}")
    print(f"Training Time: {results['training_time_seconds']:.2f} seconds")

    return results, trainer

# =========================================================
# 6. Main Comparison + Hyperparameter Experiments
# =========================================================
def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenized_datasets = load_and_prepare_data(tokenizer, CONFIG)

    all_results = []

    # -----------------------------------------
    # 1. Full Fine-Tuning
    # -----------------------------------------
    full_model = create_full_model(CONFIG)
    full_results, _ = train_and_evaluate(
        model=full_model,
        tokenized_datasets=tokenized_datasets,
        tokenizer=tokenizer,
        config=CONFIG,
        method_name="Full Fine-Tuning",
        learning_rate=CONFIG["learning_rate_full"]
    )
    all_results.append(full_results)

    # -----------------------------------------
    # 2. LoRA (Base Rank = 8)
    # -----------------------------------------
    lora_model = create_lora_model(CONFIG)
    lora_results, _ = train_and_evaluate(
        model=lora_model,
        tokenized_datasets=tokenized_datasets,
        tokenizer=tokenizer,
        config=CONFIG,
        method_name="LoRA (r=8)",
        learning_rate=CONFIG["learning_rate_lora"]
    )
    all_results.append(lora_results)

    # -----------------------------------------
    # 3. Hyperparameter Experiments - Different Ranks
    # -----------------------------------------
    print("\n\n>>> Starting Rank Experiments...")
    for rank in [4, 16]:
        CONFIG_RANK = CONFIG.copy()
        CONFIG_RANK["lora_r"] = rank

        model = create_lora_model(CONFIG_RANK)
        results, _ = train_and_evaluate(
            model=model,
            tokenized_datasets=tokenized_datasets,
            tokenizer=tokenizer,
            config=CONFIG_RANK,
            method_name=f"LoRA (r={rank})",
            learning_rate=CONFIG["learning_rate_lora"]
        )
        all_results.append(results)

    # -----------------------------------------
    # 4. Print Final Comparison Table
    # -----------------------------------------
    print("\n\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(f"{'Method':<20} {'Trainable Params':>18} {'Accuracy':>10} {'Macro F1':>10} {'Time (s)':>10}")
    print("-"*80)

    for res in all_results:
        print(f"{res['method']:<20} {res['trainable_params']:>18,} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} {res['training_time_seconds']:>10.2f}")

    # Save results
    results_path = os.path.join(CONFIG["output_dir"], "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nResults saved to: {results_path}")
    print("Done.")

if __name__ == "__main__":
    main()