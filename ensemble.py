import os
import mmcv
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

pipe_stcn = '/tmp/stcn.pkl'
pipe_aot = '/tmp/aot.pkl'

def mean_merge_back(aot, stcn):
    get_aot = torch.from_numpy(aot)
    get_stcn = torch.from_numpy(stcn)
    
    N, _, h, w = get_stcn.shape
    *_, H, W = get_aot.shape

    get_stcn = F.interpolate(get_stcn, (H, W), mode='bilinear', align_corners=True)
    merge_logit = (get_stcn + get_aot[:N]) / 2
    send_aot = torch.ones_like(get_aot) * (-1e+4)
    send_aot[:N] = merge_logit
    send_stcn = F.interpolate(merge_logit, (h, w), mode='bilinear', align_corners=True)

    return send_aot.numpy(), send_stcn.numpy()

# def just_get_merge_output(aot, stcn, output_dir):
#     get_aot = torch.from_numpy(aot)
#     get_stcn = torch.from_numpy(stcn)
    
#     N, _, h, w = get_stcn.shape
#     *_, H, W = get_aot.shape

#     get_stcn = F.interpolate(get_stcn, (H, W), mode='bilinear', align_corners=True)
#     merge_logit = (get_stcn + get_aot[:N]) / 2

#     return aot, stcn


def main():
    parser = ArgumentParser()
    parser.add_argument('--off', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if  args.off:
        try:
            os.remove(pipe_stcn)
        except:
            pass
        try:
            os.remove(pipe_aot)
        except:
            pass
        return
    else:
        try:
            os.mkfifo(pipe_stcn)
        except FileExistsError:
            os.remove(pipe_stcn)
            os.mkfifo(pipe_stcn)

        try:
            os.mkfifo(pipe_aot)
        except FileExistsError:
            os.remove(pipe_aot)
            os.mkfifo(pipe_aot)


    get_stcn = mmcv.load(pipe_stcn) 
    get_aot = mmcv.load(pipe_aot)  
    for frame_s in get_stcn: pass
    for frame_a in get_aot: pass
    f_last = ''
    while True:
        if frame_a == f_last and frame_s == f_last:
            print(f'stcn repeat at {frame_s}', flush=True)
            mmcv.dump(get_stcn[frame_s], pipe_stcn)
            get_stcn = mmcv.load(pipe_stcn) 
            for frame_s in get_stcn: pass

            print(f'aot repeat at {frame_a}', flush=True)
            mmcv.dump(get_aot[frame_a], pipe_aot)
            get_aot = mmcv.load(pipe_aot)
            for frame_a in get_aot: pass

        elif frame_a == frame_s:
            f_last = frame_a
            send_aot, send_stcn = mean_merge_back(get_aot[f_last], get_stcn[f_last])
            mmcv.dump(send_aot, pipe_aot)
            mmcv.dump(send_stcn, pipe_stcn)
            get_stcn = mmcv.load(pipe_stcn) 
            get_aot = mmcv.load(pipe_aot)  
            for frame_s in get_stcn: pass
            for frame_a in get_aot: pass

        elif frame_a > frame_s:
            print(f'stcn repeat at {frame_s}', flush=True)
            mmcv.dump(get_stcn[frame_s], pipe_stcn)
            get_stcn = mmcv.load(pipe_stcn) 
            for frame_s in get_stcn: pass

        else:
            print(f'aot repeat at {frame_a}', flush=True)
            mmcv.dump(get_aot[frame_a], pipe_aot)
            get_aot = mmcv.load(pipe_aot)
            for frame_a in get_aot: pass


if __name__ == '__main__':
    main()