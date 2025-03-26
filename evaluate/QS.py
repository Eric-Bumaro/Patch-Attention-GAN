import numpy as np
import torch
import torchvision.models as models
import lpips


def calculate_quality_score(real_images, generated_images):
    device = torch.device("cpu")
    # Load VGG model
    vgg_model = models.vgg16(pretrained=True).features.to(device)
    vgg_model.eval()
    # Load LPIPS loss model
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    # Preprocess images
    real_images = torch.from_numpy(real_images).to(device).permute(0, 3, 1, 2)
    generated_images = torch.from_numpy(generated_images).to(device).permute(0, 3, 1, 2)
    # Calculate LPIPS distance between real and generated images
    lpips_distance = lpips_loss(real_images, generated_images).mean()
    # Calculate Quality Score
    quality_score = 1.0 - lpips_distance.item()
    return quality_score


def get_QS(images1, images2):
    images1 = np.asarray(images1)
    images2 = np.asarray(images2)
    return calculate_quality_score(images1, images2)
