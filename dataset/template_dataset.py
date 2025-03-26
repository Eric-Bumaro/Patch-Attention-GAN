from dataset.base_dataset import BaseDataset, get_transform


class TemplateDataset(BaseDataset):
    def __init__(self, config):
        # save the option and dataset root
        BaseDataset.__init__(self, config)
        # get the image paths of your dataset;
        self.image_paths = []
        self.transform = get_transform(config)

    def __getitem__(self, index):
        path = 'temp'    # needs to be a string
        data_A = None    # needs to be a tensor
        data_B = None    # needs to be a tensor
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        return len(self.image_paths)
