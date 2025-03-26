from torchmetrics.image.kid import KernelInceptionDistance
import torch

kid = KernelInceptionDistance(subset_size=4)


def get_KID(images1, images2):
    images1 = torch.as_tensor(images1).permute(0, 3, 1, 2)
    images2 = torch.as_tensor(images2).permute(0, 3, 1, 2)
    kid.update(images1, real=False)
    kid.update(images2, real=True)
    return kid.compute()[0].item()
