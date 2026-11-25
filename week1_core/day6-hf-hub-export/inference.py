import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODEL_TYPE = "full"  # "full" or "lora"
FULL_MODEL_ID = "MohamedSayed112/sst2-distilbert-full-ft"
BASE_MODEL_ID = "distilbert-base-uncased"
LORA_ADAPTER_ID = "MohamedSayed112/sst2-distilbert-LoRA"
MAX_LENGTH = 128
BATCH_SIZE = 32
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def load_model(model_type: str = MODEL_TYPE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "full":
        tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(FULL_MODEL_ID)
    elif model_type == "lora":
        tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_ID)
        base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID, num_labels=2)
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
    else:
        raise ValueError("MODEL_TYPE must be 'full' or 'lora'")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def preprocess_input(texts, tokenizer, device):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        raise ValueError("Input list cannot be empty.")
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("All inputs must be non-empty strings.")

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


def predict_batch(texts, model, tokenizer, device, batch_size: int = BATCH_SIZE):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = preprocess_input(batch, tokenizer, device)

        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            confs, preds = probs.max(dim=-1)

        for pred, conf in zip(preds.cpu().tolist(), confs.cpu().tolist()):
            results.append({"label": LABELS[pred], "confidence": round(conf, 4)})

    return results


def predict(text, model, tokenizer, device):
    return predict_batch([text], model, tokenizer, device, batch_size=1)[0]


if __name__ == "__main__":
    t0 = time.perf_counter()
    model, tokenizer, device = load_model(MODEL_TYPE)
    load_time = time.perf_counter() - t0

    print(f"Device: {device} | Mode: {MODEL_TYPE} | Load Time: {load_time:.4f}s")
    print("Single:", predict("This movie was absolutely amazing!", model, tokenizer, device))

    samples = [
        "This movie was absolutely amazing!",
        "The story was boring and disappointing.",
    ]
    for text, res in zip(samples, predict_batch(samples, model, tokenizer, device)):
        print(text, "->", res)