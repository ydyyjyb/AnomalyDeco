#!/usr/bin/env bash
set -euo pipefail

# ===================== 需要修改 =====================
PROJECT_DIR="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/transformers-4.57.6"
SCRIPT="examples/deco/eval_mvtec_qwen3_vl_thinking.py"
DATASET="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec"
MODEL="/home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl"

# 15类全量搜索（MVTec AD）
CATEGORY=""  # 留空表示用 CATEGORIES
CATEGORIES="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

SHOTS="0"
METRIC_SHOT="mean"      # mean / 0 / 1 / 3
SEED=42
MAX_NEW_TOKENS=16000

GPU_IDS=(0 1 2)         # 三张卡
MAX_SAMPLES_STAGE1="5"  # 每个小类（含good）取5张
MAX_SAMPLES_STAGE2="5"  # 每个小类（含good）取5张
PHASE2_MAX_CONFIGS=18

# defect-aware 词表
# 注意：已按你要求改过 eval 脚本后，这里只需要固定“缺陷语义基词”，会自动拼接当前类 defect_types
ANCHOR_DEFECT_BASE_WORDS="${ANCHOR_DEFECT_BASE_WORDS:-defect,anomaly,abnormal}"
ANCHOR_NORM_WORDS="${ANCHOR_NORM_WORDS:-normal,uniform,undamaged,no defect,good}"
ANCHOR_HEDGE_WORDS="${ANCHOR_HEDGE_WORDS:-maybe,seems,likely,appears,possibly,could,might}"

RUN_DIR="$PROJECT_DIR/../runs/deco_2stage_3gpu_mvtec15_$(date +%Y%m%d_%H%M%S)"
# ===================================================

mkdir -p "$RUN_DIR/json/phase1" "$RUN_DIR/json/phase2" "$RUN_DIR/logs/phase1" "$RUN_DIR/logs/phase2"
cd "$PROJECT_DIR"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Script not found: $PROJECT_DIR/$SCRIPT"
  exit 1
fi

run_one() {
  local stage="$1" gpu="$2" id="$3" a="$4" p="$5" k="$6" s="$7" e="$8" ln="$9" eh="${10}" en="${11}" lse="${12}" g="${13}" max_samples="${14}"

  local out_json="$RUN_DIR/json/$stage/${id}_a${a}_p${p}_k${k}_l${s}-${e}_ln${ln}_eh${eh}_en${en}_lse${lse}_g${g}.json"
  local log_file="$RUN_DIR/logs/$stage/${id}.log"

  local extra_args=()
  [[ -n "$max_samples" ]] && extra_args+=(--max_samples_per_defect "$max_samples")
  [[ -n "$CATEGORY" ]] && extra_args+=(--category "$CATEGORY")
  [[ -z "$CATEGORY" && -n "$CATEGORIES" ]] && extra_args+=(--categories "$CATEGORIES")

  local deco_extra=()
  [[ "$en" == "1" ]] && deco_extra+=(--enable_hedge)
  [[ "$lse" == "1" ]] && deco_extra+=(--use_logsumexp)

  echo "[${stage}] [GPU${gpu}] start $id: a=$a p=$p k=$k layers=[$s,$e) ln=$ln eh=$eh en=$en lse=$lse g=$g"

  if PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --dataset_root "$DATASET" \
    --model_name_or_path "$MODEL" \
    --output_json "$out_json" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    "${extra_args[@]}" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_beams 1 \
    --attn_implementation flash_attention_2 \
    --allow_attn_fallback \
    --use_deco \
    --anchor_strategy defect_aware \
    --alpha "$a" \
    --threshold_top_p "$p" \
    --threshold_top_k "$k" \
    --start_layer "$s" \
    --end_layer "$e" \
    --lambda_norm "$ln" \
    --eta_hedge "$eh" \
    --gamma "$g" \
    --anchor_defect_words "$ANCHOR_DEFECT_BASE_WORDS" \
    --anchor_norm_words "$ANCHOR_NORM_WORDS" \
    --anchor_hedge_words "$ANCHOR_HEDGE_WORDS" \
    "${deco_extra[@]}" \
    > "$log_file" 2>&1; then
    echo "[${stage}] [GPU${gpu}] done  $id"
  else
    echo "[${stage}] [GPU${gpu}] FAIL  $id (see $log_file)"
  fi
}

run_stage() {
  local stage="$1" config_file="$2" max_samples="$3"
  local n_gpu="${#GPU_IDS[@]}"
  local i=0

  while IFS= read -r line || [[ -n "${line:-}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line// /}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    # id alpha top_p top_k start end lambda_norm eta_hedge enable_hedge use_logsumexp gamma
    read -r id a p k s e ln eh en lse g <<< "$line"

    local gpu="${GPU_IDS[$((i % n_gpu))]}"
    run_one "$stage" "$gpu" "$id" "$a" "$p" "$k" "$s" "$e" "$ln" "$eh" "$en" "$lse" "$g" "$max_samples" &

    i=$((i + 1))
    while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$n_gpu" ]]; do
      wait -n || true
    done
  done < "$config_file"

  wait || true
}

rank_stage() {
  local stage="$1"
  local best_tsv="$RUN_DIR/${stage}_best.tsv"

  RUN_DIR="$RUN_DIR" STAGE="$stage" BEST_TSV="$best_tsv" METRIC_SHOT="$METRIC_SHOT" python - <<'PY'
import os, glob, json, re, math

run_dir = os.environ["RUN_DIR"]
stage = os.environ["STAGE"]
best_tsv = os.environ["BEST_TSV"]
metric_shot = os.environ.get("METRIC_SHOT", "mean").strip().lower()

if stage == "global":
    files = glob.glob(f"{run_dir}/json/*/*.json")
else:
    files = glob.glob(f"{run_dir}/json/{stage}/*.json")

pat = re.compile(
    r"_a([-0-9.]+)_p([-0-9.]+)_k([0-9]+)_l([0-9]+)-([0-9]+)_ln([-0-9.]+)_eh([-0-9.]+)_en([01])_lse([01])_g([-0-9.]+)\.json$"
)

def pick_binary(summary):
    vals = []
    for _, m in summary.items():
        b = m.get("binary_acc")
        if b is not None:
            vals.append(float(b))
    if not vals:
        return float("nan")
    if metric_shot == "mean":
        return sum(vals) / len(vals)
    if metric_shot in summary and isinstance(summary[metric_shot], dict):
        b = summary[metric_shot].get("binary_acc")
        if b is not None:
            return float(b)
    return sum(vals) / len(vals)

rows = []
for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        continue

    summary = j.get("summary", {})
    binary = pick_binary(summary)
    if not math.isfinite(binary):
        continue

    ext_vals, parse_vals = [], []
    for _, m in summary.items():
        e = m.get("extraction_success_rate")
        p = m.get("json_parse_success_rate")
        if e is not None: ext_vals.append(float(e))
        if p is not None: parse_vals.append(float(p))
    extract = sum(ext_vals) / len(ext_vals) if ext_vals else 0.0
    parse_ok = sum(parse_vals) / len(parse_vals) if parse_vals else 0.0

    meta = j.get("meta", {})
    a = meta.get("deco_alpha")
    p = meta.get("deco_threshold_top_p")
    k = meta.get("deco_threshold_top_k")
    layers = meta.get("deco_early_exit_layers")
    ln = meta.get("deco_lambda_norm")
    eh = meta.get("deco_eta_hedge")
    en = meta.get("deco_enable_hedge")
    lse = meta.get("deco_use_logsumexp")
    g = meta.get("deco_gamma")

    s, e = None, None
    if isinstance(layers, list) and len(layers) > 0:
        s = int(min(layers))
        e = int(max(layers)) + 1

    if any(x is None for x in [a, p, k, s, e, ln, eh, en, lse, g]):
        m = pat.search(os.path.basename(fp))
        if m:
            a, p, k, s, e = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            ln, eh = float(m.group(6)), float(m.group(7))
            en, lse = int(m.group(8)), int(m.group(9))
            g = float(m.group(10))

    if any(x is None for x in [a, p, k, s, e, ln, eh, en, lse, g]):
        continue

    rows.append({
        "file": os.path.basename(fp),
        "binary": float(binary),
        "extract": float(extract),
        "parse": float(parse_ok),
        "a": float(a), "p": float(p), "k": int(k), "s": int(s), "e": int(e),
        "ln": float(ln), "eh": float(eh), "en": int(bool(en)), "lse": int(bool(lse)), "g": float(g),
    })

if not rows:
    raise RuntimeError(f"No valid result json found for stage={stage}")

# binary_acc为核心，extract/parse仅用于同分时排序
rows.sort(key=lambda x: (x["binary"], x["extract"], x["parse"]), reverse=True)

print(f"\n===== {stage.upper()} TOP (metric_shot={metric_shot}) =====")
for i, r in enumerate(rows[:10], 1):
    print(
        f"{i:>2}. binary={r['binary']:.4f} extract={r['extract']:.4f} parse={r['parse']:.4f} "
        f"a={r['a']} p={r['p']} k={r['k']} layers=[{r['s']},{r['e']}) "
        f"ln={r['ln']} eh={r['eh']} en={r['en']} lse={r['lse']} g={r['g']}  {r['file']}"
    )

best = rows[0]
with open(best_tsv, "w", encoding="utf-8") as f:
    f.write(
        f"{best['file']}\t{best['a']}\t{best['p']}\t{best['k']}\t{best['s']}\t{best['e']}\t"
        f"{best['ln']}\t{best['eh']}\t{best['en']}\t{best['lse']}\t{best['g']}\t"
        f"{best['binary']}\t{best['extract']}\t{best['parse']}\n"
    )

print(f"\nBest ({stage}): {best['file']}")
print(
    f"params: alpha={best['a']}, top_p={best['p']}, top_k={best['k']}, layers=[{best['s']},{best['e']}), "
    f"lambda_norm={best['ln']}, eta_hedge={best['eh']}, enable_hedge={best['en']}, "
    f"use_logsumexp={best['lse']}, gamma={best['g']}"
)
PY
}

# ---------------- 阶段1：粗筛 ----------------
PHASE1_CFG="$RUN_DIR/phase1_configs.tsv"
cat > "$PHASE1_CFG" <<'EOF'
# id alpha top_p top_k start end lambda_norm eta_hedge enable_hedge use_logsumexp gamma
c1  0.60 0.92 24 20 29 1.00 0.00 0 0 8.0
c2  0.60 0.92 24 20 29 1.00 0.10 1 0 8.0
c3  0.60 0.92 24 20 29 1.00 0.15 1 1 8.0
c4  0.60 0.92 24 20 29 0.80 0.15 1 1 8.0
c5  0.60 0.92 24 20 29 1.20 0.15 1 1 8.0
c6  0.50 0.92 24 20 29 1.00 0.15 1 1 8.0
c7  0.70 0.92 24 20 29 1.00 0.15 1 1 8.0
c8  0.60 0.88 24 20 29 1.00 0.15 1 1 8.0
c9  0.60 0.96 24 20 29 1.00 0.15 1 1 8.0
c10 0.60 0.92 16 20 29 1.00 0.15 1 1 8.0
c11 0.60 0.92 32 20 29 1.00 0.15 1 1 8.0
c12 0.60 0.92 24 19 28 1.00 0.15 1 1 6.0
c13 0.60 0.92 24 21 30 1.00 0.15 1 1 10.0
c14 0.55 0.90 20 20 29 1.00 0.10 1 1 8.0
c15 0.65 0.94 28 20 29 1.00 0.20 1 1 8.0
EOF

echo ">>> Phase-1 coarse search"
run_stage "phase1" "$PHASE1_CFG" "$MAX_SAMPLES_STAGE1"
rank_stage "phase1"

IFS=$'\t' read -r BEST1_FILE A0 P0 K0 S0 E0 LN0 EH0 EN0 LSE0 G0 BIN0 EXT0 PARSE0 < "$RUN_DIR/phase1_best.tsv"

# ---------------- 阶段2：围绕 best 细搜 ----------------
PHASE2_CFG="$RUN_DIR/phase2_configs.tsv"
A0="$A0" P0="$P0" K0="$K0" S0="$S0" E0="$E0" LN0="$LN0" EH0="$EH0" EN0="$EN0" LSE0="$LSE0" G0="$G0" \
OUT="$PHASE2_CFG" PHASE2_MAX_CONFIGS="$PHASE2_MAX_CONFIGS" python - <<'PY'
import os

a0 = float(os.environ["A0"])
p0 = float(os.environ["P0"])
k0 = int(float(os.environ["K0"]))
s0 = int(float(os.environ["S0"]))
e0 = int(float(os.environ["E0"]))
ln0 = float(os.environ["LN0"])
eh0 = float(os.environ["EH0"])
en0 = int(float(os.environ["EN0"]))
lse0 = int(float(os.environ["LSE0"]))
g0 = float(os.environ["G0"])
out = os.environ["OUT"]
max_cfg = int(os.environ.get("PHASE2_MAX_CONFIGS", "18"))

def clamp_p(x): return max(0.05, min(1.0, round(x, 4)))
def clamp_k(x): return max(1, int(round(x)))
def clamp_s(x): return max(0, int(x))
def clamp_e(s, e): return max(s + 1, int(e))
def clamp_ln(x): return max(0.0, round(x, 4))
def clamp_eh(x): return max(0.0, round(x, 4))
def clamp_g(x): return max(0.1, round(x, 4))

cands = [
    ("f0",  a0,               p0,               k0,              s0,                e0,                ln0,               eh0,               en0,      lse0,      g0),
    ("f1",  round(a0-0.10,4), p0,               k0,              s0,                e0,                ln0,               eh0,               en0,      lse0,      g0),
    ("f2",  round(a0+0.10,4), p0,               k0,              s0,                e0,                ln0,               eh0,               en0,      lse0,      g0),
    ("f3",  a0,               p0,               k0,              s0,                e0,                clamp_ln(ln0-0.2), eh0,               en0,      lse0,      g0),
    ("f4",  a0,               p0,               k0,              s0,                e0,                clamp_ln(ln0+0.2), eh0,               en0,      lse0,      g0),
    ("f5",  a0,               p0,               k0,              s0,                e0,                ln0,               clamp_eh(eh0-0.1), en0,      lse0,      g0),
    ("f6",  a0,               p0,               k0,              s0,                e0,                ln0,               clamp_eh(eh0+0.1), en0,      lse0,      g0),
    ("f7",  a0,               clamp_p(p0-0.02), k0,              s0,                e0,                ln0,               eh0,               en0,      lse0,      g0),
    ("f8",  a0,               clamp_p(p0+0.02), k0,              s0,                e0,                ln0,               eh0,               en0,      lse0,      g0),
    ("f9",  a0,               p0,               clamp_k(k0-8),   s0,                e0,                ln0,               eh0,               en0,      lse0,      g0),
    ("f10", a0,               p0,               clamp_k(k0+8),   s0,                e0,                ln0,               eh0,               en0,      lse0,      g0),
    ("f11", a0,               p0,               k0,              clamp_s(s0-1),     clamp_e(clamp_s(s0-1), e0-1), ln0, eh0, en0, lse0, g0),
    ("f12", a0,               p0,               k0,              clamp_s(s0+1),     clamp_e(clamp_s(s0+1), e0+1), ln0, eh0, en0, lse0, g0),
    ("f13", a0,               p0,               k0,              s0,                e0,                ln0,               eh0,               1-en0,    lse0,      g0),
    ("f14", a0,               p0,               k0,              s0,                e0,                ln0,               eh0,               en0,      1-lse0,    g0),
    ("f15", a0,               p0,               k0,              s0,                e0,                ln0,               eh0,               en0,      lse0,      clamp_g(g0+2.0)),
    ("f16", a0,               p0,               k0,              s0,                e0,                ln0,               eh0,               en0,      lse0,      clamp_g(g0-2.0)),
    ("f17", 0.60,             p0,               k0,              s0,                e0,                1.00,              eh0,               1,        1,         8.0),
]

seen = set()
uniq = []
for row in cands:
    key = tuple(row[1:])
    if key in seen:
        continue
    seen.add(key)
    uniq.append(row)

uniq = uniq[:max_cfg]

with open(out, "w", encoding="utf-8") as f:
    f.write("# id alpha top_p top_k start end lambda_norm eta_hedge enable_hedge use_logsumexp gamma\n")
    for r in uniq:
        f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]} {r[7]} {r[8]} {r[9]} {r[10]}\n")
PY

echo ">>> Phase-2 fine search"
run_stage "phase2" "$PHASE2_CFG" "$MAX_SAMPLES_STAGE2"
rank_stage "phase2"

# ---------------- 全局最优 ----------------
rank_stage "global"

echo
echo "Run dir: $RUN_DIR"
echo "Phase1 best: $(cat "$RUN_DIR/phase1_best.tsv")"
echo "Phase2 best: $(cat "$RUN_DIR/phase2_best.tsv")"
echo "Global best: $(cat "$RUN_DIR/global_best.tsv")"
