exp=${1:-'eval'}
gpu_num=4
model="swinb_aotl"
stage="pre_ytb_dav"
dataset="youtubevos2019"
split="test"  
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} \
    --ckpt_path ./pretrain_models/AOTv2_85.1_80000.pth --ms 0.75 1 1.4 --flip