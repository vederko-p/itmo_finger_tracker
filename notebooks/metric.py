
import numpy as np
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm


def oks_metric(gt_kps: np.array, st_kps: np.array):
    """
    gt_kps: [[xi, yi]]
    st_kps: [[xi, yi]]
    """
    eps = 1e-5
    d = cdist(gt_kps, st_kps).diagonal()**2
    s = (
        (gt_kps[0].max() - gt_kps[0].min())
        * (gt_kps[1].max() - gt_kps[1].min())
    )
    return np.exp((-d / (s + eps) / 2).sum())


def oks_over_dataset(dataset, model):
    oks_total = 0
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        array_img, array_lbl = dataset[i]
        st_kps = model(array_img)
        oks_total += oks_metric(array_lbl, st_kps)
    return oks_total / len(dataset)
