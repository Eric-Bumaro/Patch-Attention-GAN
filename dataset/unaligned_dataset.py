import os
from dataset.base_dataset import BaseDataset, get_transform
from dataset.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    def __init__(self, config):
        BaseDataset.__init__(self, config)
        self.dir_A = os.path.join(config.dataroot, config.phase + 'A')  # create a path '/path/to/dataset/trainA'
        self.dir_B = os.path.join(config.dataroot, config.phase + 'B')  # create a path '/path/to/dataset/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/dataset/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/dataset/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.config.direction == 'BtoA'
        input_nc = self.config.output_nc if btoA else self.config.input_nc       # get the number of channels of input image
        output_nc = self.config.input_nc if btoA else self.config.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.config, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.config, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.config.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
