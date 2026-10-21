# Experiment result summaries

This directory stores aggregate result tables and protocol notes only. Dataset images, model answers tied to individual images, checkpoints, and runtime logs are intentionally omitted.

## Main method

- `main_method_0to4shot_refprompt_youden_table_20260818.csv/.md`: 0–4-shot results with same-category normal reference images inserted into the prompt; score source is `E_raw_hdec` and thresholds are selected by Youden.
- `main_method_0to4shot_youden_table_20260817.csv/.md`: earlier 0–4-shot table. Read its protocol note before comparing it with the reference-prompt experiment.
- Other main-method CSV/Markdown tables preserve the historical aggregate values and comparisons that were already in the working results folder.
- `main_method_0shot_vs_1shot_20260817` is a historical exploratory subset where the one-shot setting did not use a strict disjoint support/test split; it is retained for provenance only and must not be cited as a strict few-shot result.

## VisA-D&R

- `visa_dr/detection_summary.csv`: ACC, precision, recall, and F1 copied from completed runs. The Qwen3 rows are HDEC-based; the Qwen2 rows are based on hard QA classification. The source run is recorded per row, so these protocol differences remain visible.
- `visa_dr/metrics_summary.csv`: language-generation metrics aggregated over the 1,000-image VisA-D&R QA export.
- `visa_dr/truncation_summary.csv` and `visa_dr/truncation_by_question.csv`: generation truncation counts for those runs.

The current DeepSeek judge evaluation is a separate post-hoc quality assessment over the existing answers. It does not change detection labels or ACC/F1. Its summary, category-level results, per-image metric table, token usage, and validated judge outputs will be added after the run finishes.
