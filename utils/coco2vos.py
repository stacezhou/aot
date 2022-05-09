from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
from pycocotools import mask as m
from PIL import Image
from collections import defaultdict
from argparse import ArgumentParser

def save2mask(objs_at_frame,filename,shape,palette):
    
    objs = []
    for i,x in enumerate(objs_at_frame):
        if x is not None:
            bin_obj = m.decode(x)
        else:
            bin_obj = np.zeros(shape)
        objs.append(bin_obj)
    objs = np.stack(objs)
    

    if objs.sum(axis=0).max() > 1:
        print(filename)
    for i in range(objs.shape[0]):
        objs[i] *= (i+1) 
        
    array = objs.sum(axis=0).astype(np.uint8)
    
    im = Image.fromarray(array)
    im.putpalette(palette)
    # print(filename)
    im.save(filename)

def main():
    parser = ArgumentParser()
    parser.add_argument('anno_file')
    parser.add_argument('output_dir')
    parser.add_argument('--palette')
    args = parser.parse_args()
    anno_file = args.anno_file
    root = args.output_dir
    palette = Image.open(args.palette).getpalette()
    with open(anno_file,'r') as f:
        anno = json.load(f)
    breakpoint()
    video_names = {x['id']:Path(x['file_names'][0]).parent.__str__() for x in anno['videos']}

    objs_at_video = defaultdict(list)
    for obj in anno['annotations']:
        objs_at_video[video_names[obj['video_id']]].append(obj)
        
    vi = 0
    va = len(video_names)
    for k,video_name in video_names.items():
        nums_frame = len(objs_at_video[video_name][0]['segmentations'])
        (Path(root) / video_name).mkdir(parents=True, exist_ok=True)
        height = objs_at_video[video_name][0]['height']
        width = objs_at_video[video_name][0]['width']
        shape = (height,width)
        vi += 1
        print(f'{vi} / {va}')
        for i in tqdm(range(nums_frame)):
            objs_at_frame = [x['segmentations'][i] for x in objs_at_video[video_name]]
            filename = Path(root) / video_name / f'img_{i:07d}.png'
            if filename.exists():
                continue
            save2mask(objs_at_frame,str(filename),shape,palette)


if __name__ == '__main__':
    main()