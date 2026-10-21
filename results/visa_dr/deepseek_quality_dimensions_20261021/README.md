# VisA-D&R quality-dimension evaluation (2026-10-21)

This is a post-hoc evaluation of the already-generated VisA-D&R answers. It does not rerun the vision-language models and does not modify their detection labels or ACC/F1 results.

## Evaluation set and completion

- Five candidate systems: Qwen3-VL-4B, Qwen2-VL-2B, Qwen2-VL-7B, LLaVA-OV-0.5B, and LLaVA-OV-7B.
- Of 1,000 source rows, 24 were excluded because one or more candidate answers were unavailable/failed. The judge input therefore contained 976 common-complete images.
- Three separate, blinded judge stages were used: reference defect-fact extraction, candidate-answer text/rubric extraction, and image-grounding review. Candidate model identities were shuffled per image. The judge was DeepSeek Flash with temperature 0, JSON output, and concurrency 3.
- Complete valid judgments were obtained for 936/976 images. The remaining 40 image records have no complete three-stage judgment and are excluded from the metric tables. A resume attempt retried only failed stages with the same settings; the API returned HTTP 402, so no missing score was fabricated or imputed.

## Metrics

- `fact_precision`, `fact_recall`, `fact_f1`: micro-aggregated defect-fact counts using deterministic one-to-one matching by slot, fixed slot thresholds, and a fixed synonym list.
- `image_support_rate` / `image_refuted_rate`: proportions of classifiable visual claims judged supported/refuted by the image. `image_uncertain_rate` is separately computed over all visual claims, including uncertain claims; non-visual claims are excluded from these denominators.
- `anomaly_rationale_mean`, `cause_mean`, `prevention_mean`: judge rubric means on a 1–5 scale for rationale quality, cause plausibility, and prevention actionability. These are automated judge scores, not expert-verified causal labels.
- Confidence intervals in the CSVs use 1,000 stratified paired bootstrap samples by category at 95% percentile intervals.

## Aggregate results

Values below are rounded; CSV files retain full precision and confidence intervals.

| Model | N images | Fact P / R / F1 | Image supported / refuted / uncertain | Rationale / cause / prevention (1–5) |
|---|---:|---:|---:|---:|
| Qwen3-VL-4B | 936 | 0.327 / 0.079 / 0.128 | 0.896 / 0.104 / 0.228 | 3.330 / 3.353 / 3.752 |
| Qwen2-VL-2B | 936 | 0.429 / 0.064 / 0.112 | 0.864 / 0.136 / 0.227 | 2.174 / 2.441 / 2.717 |
| Qwen2-VL-7B | 936 | 0.511 / 0.107 / 0.177 | 0.917 / 0.083 / 0.257 | 2.725 / 2.657 / 2.956 |
| LLaVA-OV-0.5B | 936 | 0.755 / 0.139 / 0.234 | 0.854 / 0.146 / 0.361 | 2.119 / 2.225 / 2.106 |
| LLaVA-OV-7B | 936 | 0.618 / 0.109 / 0.185 | 0.900 / 0.100 / 0.234 | 2.752 / 2.520 / 3.071 |

The image-support rate is conditional on a claim being classifiable as supported or refuted; uncertainty is not silently counted as support. Scores are exploratory comparisons from one automated judge and should not be presented as human-validated results without a manual audit.

## Files

- `quality_metrics_summary.csv`: overall model-level metrics and confidence intervals.
- `quality_metrics_by_category.csv`: category-level metrics.
- `quality_metrics_per_image.csv`: per-image, per-model score rows.
- `judge_outputs.valid.jsonl`: merged outputs for the 936 complete images.
- `judge_parts.jsonl`: dimension-level judge outputs and attempts, including invalid attempts for audit.
- `judge_failures.jsonl`: the 40 records without complete judgments; failures are kept explicit.
- `run_config.json`, `summary_meta.json`, `data_quality.json`, `api_usage_summary.json`: protocol, denominators, source quality, and successful API usage summary. No API credential is stored here.
