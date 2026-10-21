#!/bin/bash

# ODinW Inference Script (Thinking Model)
# This script runs inference on the ODinW dataset using vLLM with thinking mode parameters

# CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 python run_odinw.py infer \
#     --model-path /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/qwen3_4b_vl \
#     --data-dir ./data \
#     --output-file ./results/bottle1.jsonl \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 32768 \
#     --max-images-per-prompt 10 \
#     --max-new-tokens 32768 \
#     --temperature 0.6 \
#     --top-p 0.95 \
#     --top-k 20 \
#     --repetition-penalty 1.0 \
#     --presence-penalty 0.0\
#     --ref-img-dir /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec/bottle/train/good \
#     --num-shots 1\
#     --use-deco \
#     --deco-alpha 0.2 \
#     --deco-layer -6

 CUDA_VISIBLE_DEVICES=0 python run_odinw.py infer \
    --model-path /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/qwen3_4b_vl \
    --data-dir ./data \
    --output-file ./results/bottle1.jsonl \
    --num-shots 0 \
    --ref-img-dir /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec/bottle/train/good \
    --num-shots 1
# python ./results/jsonl2json.py

# python ../../sam2-main/notebooks/infer_with_sam_v1.py