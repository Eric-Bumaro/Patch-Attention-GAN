from model.attention_gan_model import CycleGANModel
from config import TestConfig
from dataset import create_dataset
from utils import save_images

if __name__ == "__main__":
    config = TestConfig()
    dataset = create_dataset(config)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of test images = %d' % dataset_size)

    model = CycleGANModel(config)  # create a model given opt.model and other options
    model.setup()

    model.eval()

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(config, visuals, img_path)
