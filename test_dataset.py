from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool
import cv2


def collect_imglistdic(image_root, label_root, suffix='.jpg', **kw):
    from pathlib import Path
    videos = [v.name for v in Path(image_root).iterdir() if v.is_dir()]
    label_videos = [v.name for v in Path(label_root).iterdir() if v.is_dir()]
    assert set(videos) == set(label_videos)
    imglistdict = dict()
    for v in videos:
        imglist = [f.name for f in (Path(image_root)/v).glob(f'*{suffix}')]
        labellist = [f.name for f in (Path(label_root)/v).glob(f'*.png')]
        assert len(imglist) == len(labellist)
        imglistdict[v] = [sorted(imglist), sorted(labellist)]
    return imglistdict


def main(img_root, mask_root):
    imglistdict = collect_imglistdic(img_root, mask_root)
    bad_image = []
    bad_mask = []
    for v, (img, labels) in tqdm(imglistdict.items()):
        for im, mask in zip(img, labels):
        #     try:
        #         cv2.imread(str(img_root / v / im))
        #     except:
        #         bad_image.append(str(img_root / v / im))
            try:
                Image.open(str(mask_root / v / mask))
            except:
                bad_mask.append(str(mask_root / v / mask))
#     print(bad_image)
    print(bad_mask)
#     print(len(bad_image))
    print(len(bad_mask))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_root')
    arg = parser.parse_args()
    img_root = Path(arg.data_root) / 'JPEGImages'
    mask_root = Path(arg.data_root) / 'Annotations'
    main(img_root, mask_root)
