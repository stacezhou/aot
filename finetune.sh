exp='finetune'
gpu_num=8
model="swinb_aotl"
stage="finetune"
dataset="youtubevos2019"
split="val"  

python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num} \
	--batch_size 24