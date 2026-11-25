# Day 06 — HF Hub Export & Inference

- `hf_hub_export.py`: Saves metadata and pushes Full / LoRA artifacts to Hugging Face Hub.
- `inference.py`: Runs single and batch inference using `model.eval()`, `torch.inference_mode()`, and input validation.
- `model_card.md` & `error_analysis.md`: Evaluation metrics and error breakdown.

## Run
Set `MODEL_TYPE = "full"` or `"lora"` in `inference.py`, then run:

```bash
python inference.py