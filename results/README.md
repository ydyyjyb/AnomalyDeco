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

## VisA-D&R quality-dimension judge evaluation

- `visa_dr/deepseek_quality_dimensions_20261021/`: post-hoc evaluation of existing answer files for Qwen3-VL-4B, Qwen2-VL-2B/7B, and LLaVA-OV-0.5B/7B, using DeepSeek Flash as a blind LLM judge. This does not change detection labels or ACC/F1.
- The source contained 1,000 rows; 24 images were excluded because at least one candidate answer was missing/failed, leaving 976 common-complete images. The judge produced complete, parseable results for 936 images (936 per model); 40 image records remain incomplete. A resume attempt used the same evaluation settings, but the API returned HTTP 402, so missing judgments were not imputed or treated as valid.
- `quality_metrics_summary.csv`, `quality_metrics_by_category.csv`, and `quality_metrics_per_image.csv` contain the overall, category-level, and per-image metrics. `judge_outputs.valid.jsonl` preserves the 936 complete judge records, `judge_parts.jsonl` preserves dimension-level outputs/attempts, and `judge_failures.jsonl` records the 40 incomplete records. `summary_meta.json`, `run_config.json`, `data_quality.json`, and `api_usage_summary.json` document denominators, protocol, source completeness, and successful API usage.
- Metrics: micro defect-fact precision/recall/F1 using deterministic one-to-one slot matching; visual-claim support/refutation rates among classifiable image claims (with uncertainty separately reported); and 1–5 LLM rubric means for anomaly rationale, cause plausibility, and prevention actionability. These are exploratory automated-judge metrics, not human-annotated ground truth or official Anomaly-OV metrics. Interpret rubric/grounding scores as judge outputs, not verified facts.
