from dataset.base_dataset import BaseDataset, get_transform
from dataset.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def __init__(self, config):
        BaseDataset.__init__(self, config)
        self.A_paths = sorted(make_dataset(config.dataroot, config.max_dataset_size))
        input_nc = self.config.output_nc if self.config.direction == 'BtoA' else self.config.input_nc
        self.transform = get_transform(config, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)
