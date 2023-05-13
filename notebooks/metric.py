
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
        tens_img, tens_lbl = dataset[i]
        gt_kps = tens_lbl.numpy()[:, :-1].astype(int)
        array_img = (
            tens_img.numpy().transpose(1, 2, 0) * 255
        ).astype('uint8')
        st_kps = model(array_img)
        oks_total += oks_metric(gt_kps, st_kps)
    return oks_total
