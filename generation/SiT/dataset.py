# dataset.py
import os
import io
import json
import pickle
from pathlib import Path
from typing import Tuple, List

import lmdb
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader



class ImageNetLMDB(Dataset):
    """ImageNet LMDB dataset compatible with PyTorch DataLoader & DDP."""

    def __init__(self, lmdb_path: str, transform):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            subdir=False
        )
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.class_to_idx = pickle.loads(txn.get(b'__classes__'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        key = f"{idx:08d}".encode()
        with self.env.begin() as txn:
            buf = txn.get(key)
        img_bytes, label = pickle.loads(buf)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class FFHQDataset(Dataset):
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def __init__(self, root: str, transform, recursive: bool = True):
        self.root = str(root)
        self.transform = transform
        root_p = Path(self.root)

        self.label_value = 0
        self._label_tensor = torch.tensor(self.label_value, dtype=torch.long)

        if recursive:
            files = [p for p in root_p.rglob("*") if p.suffix.lower() in self.IMG_EXTS]
        else:
            files = [p for p in root_p.iterdir() if p.suffix.lower() in self.IMG_EXTS]

        # 排序保证多卡一致
        self.files: List[Path] = sorted(files)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self._label_tensor


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None, category='name',
                 loader=default_loader):
        super().__init__(root, transform, target_transform, loader)
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year

        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


def _looks_like_imagenet_lmdb(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "imagenet_train.lmdb"))

def _has_image_subdirs(root: str) -> bool:

    root_p = Path(root)
    if not root_p.is_dir():
        return False
    subdirs = [d for d in root_p.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        return False

    for d in subdirs:
        for p in d.rglob("*"):
            if p.suffix.lower() in FFHQDataset.IMG_EXTS:
                return True
    return False

def build_dataset(transform, args):

    root = args.data_path

    if args.num_classes == 1:
        return FFHQDataset(root=root, transform=transform, recursive=True)

    lmdb_path = os.path.join(root, "imagenet_train.lmdb")
    if os.path.isfile(lmdb_path):
        return ImageNetLMDB(lmdb_path=lmdb_path, transform=transform)

    if _has_image_subdirs(root):
        return ImageFolder(root=root, transform=transform)

    alt = os.path.join(root, "train")
    if _has_image_subdirs(alt):
        return ImageFolder(root=alt, transform=transform)

    raise FileNotFoundError(
        f"Cannot build dataset from '{root}'. "
        f"Expected one of: (1) imagenet_train.lmdb file; "
        f"(2) ImageFolder-style directory; "
        f"(3) num_classes==0 for unlabeled images like FFHQ."
    )
