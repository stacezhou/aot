exp="bi_eval"
gpu_num="6"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_aotl"
model="swinb_aotl"

# stage="pre"
stage="pre_ytb_dav"
dataset="youtubevos2019"
split="val"  # or "val_all_frames"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} ${@:1}