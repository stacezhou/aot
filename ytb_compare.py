from metric import read_video_pair,metric_video_JF
import pandas as pd 
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool
import mmcv
import numpy as np

def metric_a_video(args):
    print('.',end='',flush=True)
    name, pred_video, gt_video = args
    pred_masks, gt_masks = read_video_pair(pred_video, gt_video)
    JF = metric_video_JF(pred_masks, gt_masks)
    return name,(JF['J']+JF['F']) / 2


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('gt_path')
    parser.add_argument('pred_path')
    parser.add_argument('--anew',action='store_true')
    arg = parser.parse_args()
    
    gt_path = Path(arg.gt_path)
    pred_path = Path(arg.pred_path)
    out_path = Path(arg.pred_path).parent / f'{pred_path.name}_vs_{gt_path.name}.pkl'

    if out_path.exists() and arg.anew == False:
        result = mmcv.load(out_path)
    else:
        gt_path = gt_path / 'Annotations'
        pred_path = pred_path / 'Annotations'

        pred_videos = [v for v in pred_path.iterdir() if v.is_dir()]
        videos_name = [v.name for v in pred_videos]
        gt_videos = [v for v in gt_path.iterdir() if v.is_dir() and v.name in videos_name]
        videos_name = [v.name for v in gt_videos]
        pred_videos = [v for v in pred_path.iterdir() if v.name in videos_name]

        args = list(zip(videos_name,pred_videos,gt_videos))
        p = Pool(32)
        result = p.map(metric_a_video, args)
        p.close()
        p.join()
        mmcv.dump(result, out_path)

    result_dict = {
        (v,i) : JF[i]
        for v,JF in result
        for i in range(JF.shape[0])
    }
    align_result = dict()
    max_length = max([v.shape[0] for v in result_dict.values()])
    for k,v in result_dict.items():
        # for i in range(v.shape[0]):
        #     if v[i] >= 0.9987: # 忽略 gt obj
        #         v[i] = np.nan
        #     else:
        #         break
        v = np.pad(v,(0,max_length - v.shape[0]), constant_values=np.nan)
        align_result[k]=v
    
    df = pd.DataFrame(align_result)
    df.to_csv(out_path.parent / f'df_{out_path.stem}.csv', float_format='%.3f')
    video_df = (
        df.mean()
        .reset_index()
        .rename(columns={'level_0':'name','level_1':'num_obj',0:'JF'})
        .groupby('name')
        .mean()
        .drop(columns='num_obj')
        .sort_values('JF')
    )
    video_df.to_csv(out_path.parent / f'video_df_{out_path.stem}.csv', float_format='%.3f')

    frame_df = (
        df.transpose()
        .reset_index()
        .rename(columns={'level_0':'name','level_1':'obj'})
        .groupby('name')
        .mean()
        .drop(columns='obj')
    )
    frame_df.to_csv(out_path.parent / f'frame_df_{out_path.stem}.csv', float_format='%.3f')
    print(video_df.head(10))
    
