import os
from dataset.base_dataset import BaseDataset, get_transform
from dataset.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ColorizationDataset(BaseDataset):
    def __init__(self, config):
        BaseDataset.__init__(self, config)
        self.dir = os.path.join(config.dataroot, config.phase)
        self.AB_paths = sorted(make_dataset(self.dir, config.max_dataset_size))
        assert(config.input_nc == 1 and config.output_nc == 2 and config.direction == 'AtoB')
        self.transform = get_transform(self.config, convert=False)

    def __getitem__(self, index):
        path = self.AB_paths[index]
        im = Image.open(path).convert('RGB')
        im = self.transform(im)
        im = np.array(im)
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        A = lab_t[[0], ...] / 50.0 - 1.0
        B = lab_t[[1, 2], ...] / 110.0
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        return len(self.AB_paths)
