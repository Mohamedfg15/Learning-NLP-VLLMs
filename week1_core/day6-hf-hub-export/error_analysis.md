# Error Analysis — SST-2 Validation

- **Full Fine-Tuning:** 70 / 872 errors (39 FP, 31 FN)
- **LoRA (r=16):** 87 / 872 errors (50 FP, 37 FN)

## Failure Modes
- **Sarcasm / Contrast:** Positive words paired with negative intent or vice versa.
- **Short / Ambiguous Phrases:** Fragments without broader context.
- **Truncation:** Sequences truncated at `max_length=128`.