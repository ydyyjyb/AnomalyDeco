# AnomalyDeco

Research code and experiment summaries for training-free multimodal industrial anomaly detection and anomaly QA with Qwen-VL and DeCo.

## Repository layout

- `Deco-main/`: DeCo implementation, Transformers integration, and experiment entry points.
- `scripts/`: reproducible evaluation, analysis, validation, and export scripts.
- `JUDO_scripts/`: JUDO-style evaluation scripts used in this project.
- `evaluation/`: dataset loaders and benchmark setup. Dataset payloads are intentionally not included; placeholder directories preserve expected paths.
- `results/`: aggregate tables for the main method, VisA-D&R, prior comparisons, and the DeepSeek quality-dimension evaluation (including per-image judgments and stage outputs).
- `PADA/`, `sam2-main/`, `qwen-vl-finetune/`, `qwen-vl-utils/`, and related directories: supporting source code. See retained upstream license files.
- `third_party_licenses/`: license text for upstream materials whose original top-level README/license files were not project-root files.

Jupyter notebooks are included as code-only notebooks; saved cell outputs and embedded attachments were removed so generated data or media are not accidentally published.

## Data and model weights

Raw datasets and model weights are not stored in this public repository. Empty placeholder folders marked with `.gitkeep` preserve the expected names. Download each dataset/model from its official source and place it in the corresponding directory before running scripts. Do not commit dataset images, checkpoints, or generated run directories.

The expected dataset folders include `evaluation/ODinW-13/data/{VisA,MVTec,AITEX,BTAD,ELPV,MPDD,BrainMRI,HeadCT,Br35H}`, `evaluation/VisA-DR-official/VisA`, and `evaluation/MMAD`.

The expected model folders include `Deco-main/qwen3_4b_vl_instruct`, `Deco-main/qwen3_4b_vl_thinking`, `Deco-main/qwen2_vl_2b_instruct`, `Deco-main/qwen2_vl_7b_instruct`, and `Deco-main/weights/visa_dr_open_models/`.

## Results

See [results/README.md](results/README.md) for which tables are included and the evaluation protocols they represent. In particular, the main-method tables use `E_raw_hdec + Youden`; the VisA-D&R detection table retains each run's source and notes protocol differences rather than implying every row is directly comparable.

## Reproducibility and secrets

Scripts expect dependencies and model/data paths to be configured for the local machine. API credentials must be supplied through environment variables (for example, `DEEPSEEK_API_KEY`) and must never be committed. Run outputs, logs, dataset payloads, and model checkpoints are ignored by `.gitignore`.

This repository intentionally has no blanket root license: the workspace combines code with different upstream licenses. Check each component's retained license before redistributing or reusing it.
