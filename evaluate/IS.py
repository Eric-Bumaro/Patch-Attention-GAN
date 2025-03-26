import cv2
import torch
import torch.utils.data
from torch import nn
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm


class InceptionScore:
    def __init__(self, split=4):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.up = nn.Upsample(size=(299, 299), mode='bilinear')
        self.split = split

    def judge(self, img_list):
        N = len(img_list)
        preds = np.zeros((N, 1000))
        length = len(img_list)
        for index in tqdm(range(length)):
            img = img_list[index]
            img = Image.fromarray(img)
            img = self.transforms(img)
            img = img.unsqueeze(0)
            img = self.up(img)
            img = img.to(self.device)
            pred = F.softmax(self.inception_model(img)).cpu().detach().numpy()
            preds[index] = pred
        split_scores = []
        for k in range(self.split):
            part = preds[k * (N // self.split): (k + 1) * (N // self.split), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores)


inception_model = InceptionScore()

if __name__ == "__main__":
    image = cv2.imread("../ALIKE/alike_test.jpg")
    print(inception_model.judge([image]))
