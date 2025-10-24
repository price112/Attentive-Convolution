import os
import json
import random
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None

# repa_dataset_lmdb.py
import os, io, json
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
from PIL import Image
try:
    import pyspng
except ImportError:
    pyspng = None

import lmdb, msgpack

def _file_ext(fname):
    return os.path.splitext(fname)[1].lower()

def _to_chw_uint8(x: np.ndarray):
    # HxWxC or HxW -> CxHxW
    if x.ndim == 2:
        x = x[:, :, None]
    return x.reshape(*x.shape[:2], -1).transpose(2, 0, 1)

class CustomDataset(Dataset):

    def __init__(self, data_dir, lmdb_path=None):
        PIL.Image.init()
        self.data_dir = data_dir
        self.lmdb_path = lmdb_path
        self.use_lmdb = False

        if lmdb_path is not None and os.path.isdir(lmdb_path):
            # 尝试打开 LMDB
            try:
                self.env = lmdb.open(lmdb_path, subdir=True, readonly=True, lock=False,
                                     readahead=True, meminit=False, max_readers=2048)
                with self.env.begin(write=False) as txn:
                    _len = txn.get(b'__len__')
                    assert _len is not None
                    self._length = int(_len.decode('utf-8'))
                self.use_lmdb = True
            except Exception as e:
                self.use_lmdb = False

        if not self.use_lmdb:
            
            supported_ext = set(PIL.Image.EXTENSION.keys()) | {'.npy'}
            self.images_dir = os.path.join(self.data_dir, 'images')
            self.features_dir = os.path.join(self.data_dir, 'vae-sd')

            _image_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self.images_dir)
                for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
            self.image_fnames = sorted(fn for fn in _image_fnames if _file_ext(fn) in supported_ext)

            _feature_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self.features_dir)
                for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
            self.feature_fnames = sorted(fn for fn in _feature_fnames if _file_ext(fn) in supported_ext)

            with open(os.path.join(self.features_dir, 'dataset.json'), 'rb') as f:
                labels_map = dict(json.load(f)['labels'])
            labels = [labels_map[fname.replace('\\', '/')] for fname in self.feature_fnames]
            labels = np.asarray(labels)
            self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

            assert len(self.image_fnames) == len(self.feature_fnames), \
                "Number of feature files and label files should be same"

    def __len__(self):
        return self._length if self.use_lmdb else len(self.feature_fnames)

    def _decode_image_from_bytes(self, img_bytes, img_ext):
        if img_ext == '.npy':

            arr = np.load(io.BytesIO(img_bytes))
            if arr.ndim == 2:  # HxW
                arr = arr[:, :, None]
            # (C,H,W)
            arr = arr.reshape(-1, *arr.shape[-2:])
            return arr
        else:

            with Image.open(io.BytesIO(img_bytes)) as im:
                im = im.convert('RGB')
                arr = np.array(im, dtype=np.uint8)
            arr = _to_chw_uint8(arr)
            return arr

    def _decode_feature_from_bytes(self, feat_bytes):
        return np.load(io.BytesIO(feat_bytes), allow_pickle=False)

    def __getitem__(self, idx):
        if self.use_lmdb:

            with self.env.begin(write=False) as txn:
                rec_raw = txn.get(f'{idx:08d}'.encode('ascii'))
                if rec_raw is None:
                    raise IndexError(idx)
                rec = msgpack.unpackb(rec_raw, raw=False)
            img = self._decode_image_from_bytes(rec['img_bytes'], rec['img_ext'])
            feat = self._decode_feature_from_bytes(rec['feat_npy_bytes'])
            label = rec['label']
            return torch.from_numpy(img), torch.from_numpy(feat), torch.as_tensor(label)
        else:

            image_fname = self.image_fnames[idx]
            feature_fname = self.feature_fnames[idx]
            image_ext = _file_ext(image_fname)
            # image
            with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
                if image_ext == '.npy':
                    image = np.load(f)
                    image = image.reshape(-1, *image.shape[-2:])
                else:

                    with Image.open(f) as im:
                        im = im.convert('RGB')
                        image = np.array(im, dtype=np.uint8)
                    image = _to_chw_uint8(image)
            # feature
            feat = np.load(os.path.join(self.features_dir, feature_fname))
            return torch.from_numpy(image), torch.from_numpy(feat), torch.tensor(self.labels[idx])


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset #if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError

class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        with open(os.path.join(self.root, f'{index}.png'), 'rb') as f:
            x = np.array(PIL.Image.open(f))
            x = x.reshape(*x.shape[:2], -1).transpose(2, 0, 1)

        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return x, z, c


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, z, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, z, y

class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=True, p_uncond=0.1, mode='train'):
        super().__init__()
        print('Prepare dataset...')
        if mode == 'val':
            self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
            assert len(self.test) == 40504
            self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))
        else:
            self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
            assert len(self.train) == 82783
            self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

            if cfg:  # classifier free guidance
                assert p_uncond is not None
                print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
                self.train = CFGDataset(self.train, p_uncond, self.empty_context)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'