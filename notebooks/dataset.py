
import os
import yaml
from PIL import Image

from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def read_yaml(filepath: str):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data


class DatasetKeypoints(Dataset):
    def __init__(self, ds_path: str, img_size):

        self.ds_path = ds_path
        self.img_size = img_size

        self.images = None
        self.images_p = None
        self.keypoints = None
        self.transform = None

        self._collect_keypoints()
        self._build_transform()

    def _build_transform(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
        ])

    def _collect_keypoints(self):
        self.images = [
            f for f in os.listdir(self.ds_path) if '.jpg' in f
        ]
        self.images_p = [
            os.path.join(self.ds_path, img) for img in self.images
        ]
        self.keypoints = []
        for img in tqdm(self.images, total=len(self.images)):
            f_name = '.'.join(img.split('.')[:-1])[:-2]
            # labels paths:
            left_lp = os.path.join(
                self.ds_path, f_name + '_l.json'
            )
            right_lp = os.path.join(
                self.ds_path, f_name + '_r.json'
            )
            # labels left data:
            if os.path.isfile(left_lp):
                left_data = read_yaml(left_lp)['hand_pts']
            else:
                left_data = [[0, 0, 0]] * 21
            # labels right data:
            if os.path.isfile(right_lp):
                right_data = read_yaml(right_lp)['hand_pts']
            else:
                right_data = [[0, 0, 0]] * 21
            # result data:
            self.keypoints.append(left_data + right_data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, indx):
        pil_img = Image.open(self.images_p[indx])
        tens_img = self.transform(pil_img)
        r = self.img_size / torch.tensor(list(pil_img.size) + [1])
        keypoints = torch.tensor(self.keypoints[indx]) * r
        return tens_img, keypoints
