exp=${1:-'finetune'}
gpu_num=4
model="swinb_aotl"
stage="finetune"

python tools/train.py --amp \
	--lstt_v2 \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num} \
	--batch_size 12 