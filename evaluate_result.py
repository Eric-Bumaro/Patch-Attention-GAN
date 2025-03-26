import argparse
import numpy as np
import cv2
import os
import glob

from tqdm import tqdm

from evaluate.FID import get_FID
from evaluate.IS import inception_model
from evaluate.KID import get_KID
from evaluate.LPIPS import get_LPIPS
from evaluate.QS import get_QS
from evaluate.SWD import get_SWD
from evaluate.historical import get_ssim, get_MSE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', required=True)
    args = parser.parse_args()

    image_root_dir = args.imgs_path
    mean_MSE = 0
    SSIM = 0
    LPIPS = 0
    count = 0
    generate_file = glob.glob(os.path.join(image_root_dir, "*fake_B.png"))

    images1 = []
    images2 = []
    for file_path in tqdm(generate_file):
        truth_file_path = file_path.replace("fake", "real")
        fake_img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
        real_img = cv2.imread(truth_file_path, cv2.COLOR_BGR2RGB)
        SSIM += get_ssim(fake_img, real_img)
        mean_MSE += get_MSE(fake_img, real_img)
        LPIPS += get_LPIPS(fake_img, real_img).item()
        count += 1
        images1.append(fake_img)
        images2.append(real_img)

    mean_MSE = mean_MSE / count
    SSIM = SSIM / count
    PSNR = 10 * np.log10(255 * 255 / mean_MSE)
    print("\nNow get FID")
    FID = get_FID(images1, images2)
    print("\nNow get IS")
    IS = inception_model.judge(images1)
    print("\nNow get KID")
    KID = get_KID(images1, images2)
    LPIPS = LPIPS / count
    # print("\nNow get QS")
    # QS = get_QS(images1, images2)
    print("\nNow get SWD\n")
    SWD = get_SWD(images1, images2)

    print("SSIM: ", SSIM)
    print("PSNR: ", PSNR)
    print("FID: ", FID)
    print("IS: ", IS)
    print("KID: ", KID)
    print("LPIPS: ", LPIPS)
    # print("QS: ", QS)
    print("SWD: ", SWD)
