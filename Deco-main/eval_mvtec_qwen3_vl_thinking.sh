cd /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/transformers-4.57.6

PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES=2 python examples/deco/eval_mvtec_qwen3_vl_thinking.py \
  --dataset_root /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec \
  --model_name_or_path /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl \
  --output_json /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/res_bottle_0shot_640.json \
  --shots 0 \
  --category bottle \
  --max_new_tokens 16000
# f9_a0.45_p0.88_k16_l21-30.json	0.45	0.88	16	21	30	0.5623254924459744
# PYTHONPATH="$(pwd)/src" CUDA_VISIBLE_DEVICES=2 python examples/deco/eval_mvtec_qwen3_vl_thinking.py \
#     --dataset_root /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/evaluation/ODinW-13/data/mvtec \
#     --model_name_or_path /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/qwen3_4b_vl \
#     --output_json /home/s414f/s414f1/414f/ano_vlms/Qwen3-VL-main/Deco-main/res_bottle_deco_0shot.json \
#     --category bottle \
#     --shots 0 \
#     --use_deco \
#     --alpha 0.45 \
#     --threshold_top_p 0.88 \
#     --threshold_top_k 16 \
#     --start_layer 21 \
#     --end_layer 30



# do_sample 控制“下一 token 如何选”。

# do_sample=False
# 不采样，按分数最高 token 走（贪心/beam 的确定性路径）。
# 特点：更稳定、可复现，输出更“保守”。

# do_sample=True
# 按概率分布随机采样 token。
# 特点：多样性更高，但结果波动更大、可复现性更差（除非固定 seed 且环境一致）。

# 在你这个脚本里还对应：

# 开启 do_sample 时 temperature 才生效。
# 关闭 do_sample 时 temperature 会被忽略。
# 做评测（尤其要比较指标）通常建议先用 do_sample=False，保证结果稳定。
# num_beams 是束搜索（Beam Search）的宽度参数。

# num_beams=1：不使用束搜索，走贪心/采样路径。
# num_beams>1：启用束搜索。每一步保留多条候选序列，最后选整体分数更高的输出。
