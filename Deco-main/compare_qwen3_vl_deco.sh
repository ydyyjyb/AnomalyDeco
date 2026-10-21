cd /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/transformers-4.57.6
PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES=1 python examples/deco/compare_qwen3_vl_deco.py \
  --model_name_or_path /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl \
  --image_path /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/bottle1/images/test_contamination_003.png \
  --prompt "请详细描述图像内容" \
  --max_new_tokens 4000 \
  --alpha 0.6 \
  --threshold_top_p 0.9 \
  --threshold_top_k 20 \
  --start_layer 20 \
  --end_layer 29
