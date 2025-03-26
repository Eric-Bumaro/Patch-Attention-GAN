import lpips
import torch

loss_fn_alex = lpips.LPIPS(net='alex')


def get_LPIPS(image1, image2):
    image1 = torch.as_tensor([image1]).permute(0, 3, 1, 2)
    image2 = torch.as_tensor([image2]).permute(0, 3, 1, 2)
    return loss_fn_alex(image1, image2)
