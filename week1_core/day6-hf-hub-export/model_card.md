# DistilBERT SST-2 Sentiment Classifier

- **Base Model:** `distilbert-base-uncased`
- **Dataset:** GLUE `sst2` (Validation set: 872 samples)
- **HF Hub:** `MohamedSayed112/sst2-distilbert-full-ft` | `MohamedSayed112/sst2-distilbert-LoRA`

| Method | Trainable Params | Accuracy | Macro F1 | Train Time (s) | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full Fine-Tuning | 66,955,010 | 0.9197 | 0.9197 | 629.38 | `[[389, 39], [31, 413]]` |
| LoRA (r=16) | 887,042 (~1.32%) | 0.9002 | 0.9001 | 452.24 | `[[378, 50], [37, 407]]` |