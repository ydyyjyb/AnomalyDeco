import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs" / "qwen_visa_dr_1000_analysis"
OUT = SOURCE_ROOT / "export"

MODELS = {
    "Qwen2-VL-2B": SOURCE_ROOT / "qwen2vl_paper_official_dataset_20260901_023412" / "visa_dr_qwen2vl-2b_official_dataset",
    "Qwen2-VL-7B": SOURCE_ROOT / "qwen2vl_paper_official_dataset_20260901_023412" / "visa_dr_qwen2vl-7b_official_dataset",
    "Qwen3-VL-4B": SOURCE_ROOT / "visa_dr_reasoning_paper_ready_banbang_full_20260909_154349" / "qwen3_original",
    "Qwen3-VL-4B+DeCo": SOURCE_ROOT / "visa_dr_reasoning_deco_p10_k200_full_20260910_033822" / "qwen3_deco",
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else None


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qa_rows = []
    image_rows = []
    summaries = []
    validation = []
    category_rows = []

    for model, folder in MODELS.items():
        predictions = read_jsonl(folder / "reasoning_predictions.jsonl")
        low_pairs = read_jsonl(folder / "low_level_pairs.jsonl")
        complex_pairs = read_jsonl(folder / "complex_pairs.jsonl")
        pair_lookup = {
            (str(p["index"]), int(p["question_index"])): p
            for p in low_pairs + complex_pairs
        }
        model_qa = []
        model_images = []

        for row in predictions:
            idx = str(row["index"])
            questions = row.get("questions") or []
            answers = row.get("answers") or []
            responses = row.get("responses") or []
            hits = row.get("hit_max_new_tokens") or [False] * len(questions)
            per_image = []
            for q_idx, (question, reference, response) in enumerate(zip(questions, answers, responses)):
                pair = pair_lookup[(idx, q_idx)]
                record = {
                    "model": model,
                    "index": idx,
                    "category": row.get("category", ""),
                    "image": row.get("image", ""),
                    "question_index": q_idx,
                    "reasoning_level": "Low-level" if q_idx <= 1 else "Complex",
                    "question": question,
                    "reference_answer": reference,
                    "model_answer": response,
                    "rouge_l": pair.get("rouge_l"),
                    "sbert": pair.get("sbert"),
                    "hit_max_new_tokens": bool(hits[q_idx]) if q_idx < len(hits) else False,
                    "output_tokens": (row.get("raw_output_token_counts") or [None] * len(questions))[q_idx],
                }
                qa_rows.append(record)
                model_qa.append(record)
                per_image.append(record)

            low = [x for x in per_image if x["reasoning_level"] == "Low-level"]
            complex_ = [x for x in per_image if x["reasoning_level"] == "Complex"]
            image_record = {
                "model": model,
                "index": idx,
                "category": row.get("category", ""),
                "image": row.get("image", ""),
                "question_count": len(per_image),
                "low_question_count": len(low),
                "complex_question_count": len(complex_),
                "low_rouge_l": mean([x["rouge_l"] for x in low]),
                "low_sbert": mean([x["sbert"] for x in low]),
                "complex_sbert": mean([x["sbert"] for x in complex_]),
                "truncated_question_count": sum(x["hit_max_new_tokens"] for x in per_image),
            }
            image_rows.append(image_record)
            model_images.append(image_record)

        summary = {
            "model": model,
            "images": len(predictions),
            "questions": len(model_qa),
            "low_pairs": sum(x["reasoning_level"] == "Low-level" for x in model_qa),
            "complex_pairs": sum(x["reasoning_level"] == "Complex" for x in model_qa),
            "low_rouge_l": mean([x["rouge_l"] for x in model_qa if x["reasoning_level"] == "Low-level"]),
            "low_sbert": mean([x["sbert"] for x in model_qa if x["reasoning_level"] == "Low-level"]),
            "complex_sbert": mean([x["sbert"] for x in model_qa if x["reasoning_level"] == "Complex"]),
            "truncated_questions": sum(x["hit_max_new_tokens"] for x in model_qa),
        }
        summaries.append(summary)
        by_category = defaultdict(list)
        for item in model_images:
            by_category[item["category"]].append(item)
        for category, items in sorted(by_category.items()):
            category_rows.append({
                "model": model,
                "category": category,
                "images": len(items),
                "low_rouge_l": mean([x["low_rouge_l"] for x in items]),
                "low_sbert": mean([x["low_sbert"] for x in items]),
                "complex_sbert": mean([x["complex_sbert"] for x in items]),
                "truncated_questions": sum(x["truncated_question_count"] for x in items),
            })
        validation.append({
            "model": model,
            "rows_are_1000": len(predictions) == 1000,
            "unique_images": len({r["image"] for r in predictions}),
            "pair_lookup_complete": len(model_qa) == len(pair_lookup),
            "questions_equal_4029": len(model_qa) == 4029,
        })

    qa_fields = [
        "model", "index", "category", "image", "question_index", "reasoning_level",
        "question", "reference_answer", "model_answer", "rouge_l", "sbert",
        "hit_max_new_tokens", "output_tokens",
    ]
    image_fields = [
        "model", "index", "category", "image", "question_count", "low_question_count",
        "complex_question_count", "low_rouge_l", "low_sbert", "complex_sbert",
        "truncated_question_count",
    ]
    summary_fields = [
        "model", "images", "questions", "low_pairs", "complex_pairs", "low_rouge_l",
        "low_sbert", "complex_sbert", "truncated_questions",
    ]
    category_fields = [
        "model", "category", "images", "low_rouge_l", "low_sbert", "complex_sbert",
        "truncated_questions",
    ]
    write_csv(OUT / "all_models_qa_metrics.csv", qa_rows, qa_fields)
    write_csv(OUT / "all_models_image_metrics.csv", image_rows, image_fields)
    write_csv(OUT / "model_summary.csv", summaries, summary_fields)
    write_csv(OUT / "category_summary.csv", category_rows, category_fields)
    with (OUT / "all_models_image_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in image_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "model_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    with (OUT / "category_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(category_rows, handle, ensure_ascii=False, indent=2)
    with (OUT / "all_models_qa_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in qa_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"summaries": summaries, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
