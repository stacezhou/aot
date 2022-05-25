GPU=${1:-6}
## start unit
stage=0
# GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./pretrain_models/AOTv2_${GPU}5.1_${GPU}0000.pth &&
# python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&


## unit
stage=1 &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
GPU_NEED=${GPU} cuda mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=2 &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=3 &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=4 &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=5 &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=6 &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=${GPU} &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=${GPU} &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval &&

## unit
stage=9 &&
GPU_NEED=${GPU} cuda python tools/eval.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} --ckpt_path ./results/boost_SwinB_AOTL__BOOST_FT/ckpt/save_step_1000.pth &&
mv results/boost_SwinB_AOTL__BOOST_FT/ckpt/ results/boost_SwinB_AOTL__BOOST_FT/ckpt${stage} &&
python ytb_compare.py datasets/BOOST results/boost_SwinB_AOTL__BOOST_FT/eval/vos_test_val_boost_SwinB_AOTL_BOOST_FT_ckpt_unknown_ema && 
GPU_NEED=${GPU} cuda python tools/train.py --exp_name boost --stage boost --model swinb_aotl --gpu_num ${GPU} &&
mv results/boost_SwinB_AOTL__BOOST_FT/eval/ results/boost_SwinB_AOTL__BOOST_FT/eval${stage} &&
mkdir results/boost_SwinB_AOTL__BOOST_FT/eval &&
cp results/boost_SwinB_AOTL__BOOST_FT/eval${stage}/*.csv results/boost_SwinB_AOTL__BOOST_FT/eval