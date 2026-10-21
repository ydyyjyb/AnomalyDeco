import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "qwen_visa_dr_1000_analysis" / "export" / "all_models_qa_metrics.jsonl"
OUT = ROOT / "outputs" / "qwen_visa_dr_1000_analysis" / "category_examples.json"
MODEL_ORDER = ["Qwen2-VL-2B", "Qwen2-VL-7B", "Qwen3-VL-4B", "Qwen3-VL-4B+DeCo"]


def main():
    rows = [json.loads(line) for line in DATA.open(encoding="utf-8") if line.strip()]
    selected = {}
    for row in rows:
        category = row["category"]
        key = (int(row["index"]), row["image"])
        if category not in selected or key < selected[category]:
            selected[category] = key

    examples = []
    for category in sorted(selected):
        index_int, image = selected[category]
        subset = [
            row for row in rows
            if row["category"] == category and int(row["index"]) == index_int and row["image"] == image
        ]
        subset.sort(key=lambda r: (MODEL_ORDER.index(r["model"]), int(r["question_index"])))
        for row in subset:
            q_idx = int(row["question_index"])
            examples.append({
                **row,
                "low_rouge_l_applicable": row["rouge_l"] if q_idx <= 1 else None,
                "low_sbert_applicable": row["sbert"] if q_idx <= 1 else None,
                "complex_sbert_applicable": row["sbert"] if q_idx >= 2 else None,
            })

    payload = {
        "selection": [
            {"category": category, "index": f"{index_int:08d}", "image": image}
            for category, (index_int, image) in sorted(selected.items())
        ],
        "examples": examples,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"categories": len(selected), "rows": len(examples), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
