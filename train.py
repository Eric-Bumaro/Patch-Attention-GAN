import os
import time
import numpy as np
from matplotlib import pyplot as plt

from dataset import create_dataset
from config import TrainConfig
from model.attention_gan_model import CycleGANModel
from utils import print_options, set_gpu_id

if __name__ == '__main__':
    config = TrainConfig()
    print_options(config)
    set_gpu_id(config)
    dataset = create_dataset(config)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = CycleGANModel(config)  # create a model given opt.model and other options
    model.setup()  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations

    total_loss_D = []
    total_loss_G = []
    total_loss_G_A2B = []
    total_loss_G_B2A = []
    total_loss_A_Cycle = []
    total_loss_B_Cycle = []
    total_loss_background_mask = []
    for epoch in range(config.epoch, config.n_epochs):
        epoch_start_time = time.time()
        epoch_iter = 0
        model.update_learning_rate()

        epoch_loss_D = []
        epoch_loss_G = []
        epoch_loss_G_A2B = []
        epoch_loss_G_B2A = []
        epoch_loss_A_Cycle = []
        epoch_loss_B_Cycle = []
        epoch_loss_background_mask = []
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += config.batch_size
            epoch_iter += config.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            loss_G_list, loss_D = model.optimize_parameters()
            # calculate loss functions, get gradients, update network weights
            epoch_loss_D.append(loss_D)
            epoch_loss_G.append(loss_G_list[0])
            epoch_loss_G_A2B.append(loss_G_list[1])
            epoch_loss_G_B2A.append(loss_G_list[2])
            epoch_loss_A_Cycle.append(loss_G_list[3])
            epoch_loss_B_Cycle.append(loss_G_list[4])
            epoch_loss_background_mask.append(loss_G_list[5])

        if (epoch + 1) % config.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        epoch_mean_loss_D = np.mean(epoch_loss_D).item()
        epoch_mean_loss_G = np.mean(epoch_loss_G).item()
        epoch_mean_loss_G_A2B = np.mean(epoch_loss_G_A2B).item()
        epoch_mean_loss_G_B2A = np.mean(epoch_loss_G_B2A).item()
        epoch_mean_loss_A_Cycle = np.mean(epoch_loss_A_Cycle).item()
        epoch_mean_loss_B_Cycle = np.mean(epoch_loss_B_Cycle).item()
        epoch_mean_loss_background_mask = np.mean(epoch_loss_background_mask).item()

        total_loss_D.append(epoch_mean_loss_D)
        total_loss_G.append(epoch_mean_loss_G)
        total_loss_G_A2B.append(epoch_mean_loss_G_A2B)
        total_loss_G_B2A.append(epoch_mean_loss_G_B2A)
        total_loss_A_Cycle.append(epoch_mean_loss_A_Cycle)
        total_loss_B_Cycle.append(epoch_mean_loss_B_Cycle)
        total_loss_background_mask.append(epoch_mean_loss_background_mask)

        print('End of epoch %d / %d \t Time Taken: %d sec \t Discriminator  Loss: %.4f \t Generator  Loss: '
              '%.4f \t  A2B Loss: %.4f \t B2A Loss: %.4f \t A Cycle Loss: %.4f \t B Cycle Loss: %.4f \t '
              'Background Mask Loss: %.4f' %
              (epoch, config.n_epochs, time.time() - epoch_start_time,
               epoch_mean_loss_D, epoch_mean_loss_G, epoch_mean_loss_G_A2B, epoch_mean_loss_G_B2A,
               epoch_mean_loss_A_Cycle, epoch_mean_loss_B_Cycle, epoch_mean_loss_background_mask))

    save_suffix = os.path.join(config.result_dir, config.experiment_name)
    D_loss_graph = os.path.join(save_suffix, "loss_D.png")
    G_loss_graph = os.path.join(save_suffix, "loss_G.png")
    G_loss_detail_graph = os.path.join(save_suffix, "loss_G_detail.png")

    plt.Figure(figsize=(12, 8))
    plt.plot(total_loss_D)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("D_Loss")
    plt.savefig(D_loss_graph)
    plt.close()

    plt.Figure(figsize=(12, 8))
    plt.plot(total_loss_G)
    plt.xlabel("Epoch")
    plt.ylabel("G_Loss")
    plt.savefig(G_loss_graph)
    plt.close()

    plt.Figure(figsize=(12, 8))
    plt.plot(total_loss_G_A2B, label="A2B GAN", color="blue")
    plt.plot(total_loss_G_B2A, label="B2A GAN", color="green")
    plt.plot(total_loss_A_Cycle, label="A Cycle", color="orange")
    plt.plot(total_loss_B_Cycle, label="B Cycle", color="yellow")
    plt.plot(total_loss_background_mask, label="Background Mask", color="red")
    plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
    plt.tight_layout()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(G_loss_detail_graph)
    plt.close()
