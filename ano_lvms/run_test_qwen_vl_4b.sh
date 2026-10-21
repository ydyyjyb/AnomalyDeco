CUDA_VISIBLE_DEVICES=0 python test_qwen_vl_withdeco_4b.py  \
  --model-path qwen3_4b_vl \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 4000
