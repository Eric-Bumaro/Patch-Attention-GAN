import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def get_ssim(fake_img, real_img):
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2GRAY)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
    ssim_score, _ = ssim(fake_img, real_img, full=True)
    return ssim_score


def get_MSE(fake_img, real_img):
    size = real_img.shape[0] * real_img.shape[1] * real_img.shape[2]
    MSE = np.sum((real_img - fake_img) ** 2) / size
    return MSE
