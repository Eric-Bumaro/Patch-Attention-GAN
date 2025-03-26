import os
from dataset.base_dataset import BaseDataset, get_params, get_transform
from dataset.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def __init__(self, config):
        BaseDataset.__init__(self, config)
        self.dir_AB = os.path.join(config.dataroot, config.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths
        assert(self.config.load_size >= self.config.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.config.output_nc if self.config.direction == 'BtoA' else self.config.input_nc
        self.output_nc = self.config.input_nc if self.config.direction == 'BtoA' else self.config.output_nc

    def __getitem__(self, index):
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.config, A.size)
        A_transform = get_transform(self.config, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.config, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)
