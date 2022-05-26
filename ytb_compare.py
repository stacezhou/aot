import pandas as pd 
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image
from tqdm import tqdm

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    if inters == union == 0:
        return 1
    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def metric_two_masks(m1,m2,labels):
    return  [
        [l, db_eval_iou(m1==l,m2==l)]
        for l in labels 
    ]

def read_mask(root, name, fid):
    return Image.open(Path(root) / 'Annotations' / name / f'{fid:05d}.png').__array__()

def metric_a_frame(root1, root2, name, fid, labels):
    m1 = read_mask(root1, name, fid)
    m2 = read_mask(root2, name, fid)
    res =  metric_two_masks(m1, m2, labels)
    return [
        [name, fid, *label_iou]
        for label_iou in res
    ]

def metric_frames(root1, root2 ,nfls):
    results = []
    if len(nfls) > 4000:
        nfls = tqdm(nfls)
    for name,fid,labels in nfls:
        results.extend(metric_a_frame(root1, root2, name, fid, labels))
    return results

def metric_frames_unit(arg):
    root1, root2, nfls = arg
    return metric_frames(*arg)

def mp_metric_frames(root1, root2, nfls, nprocs = 48):
    step = (len(nfls) + nprocs) // nprocs
    args = []
    for i in range(0,len(nfls),step):
        args.append([root1,root2,nfls[i:i+step]])
    p = Pool(nprocs)
    result = p.map(metric_frames_unit, args)
    p.close()
    p.join()
    flat = []
    for res in result:
        flat.extend(res)
    return flat