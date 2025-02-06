import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import random
from PIL import ImageFilter
from torch.nn import functional as F
    
from PIL import Image, ImageDraw
import random

# Code taken from https://github.com/autonomousvision/akorn
# specifically: https://github.com/autonomousvision/akorn/blob/main/source/data/augs.py

def gauss_noise_tensor(sigma=0.1):
    def fn(img):
        out = img + sigma * torch.randn_like(img)     
        out = torch.clamp(out, 0, 1) #  pixel space is [0, 1] 
        return out
    return fn


def my_aug():
    transform_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.75, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
        ]
    )
    return transform_aug

def my_aug_strong(noise=0.):
    transform_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.AugMix(),
            transforms.ToTensor(),
            gauss_noise_tensor(noise) if noise > 0 else lambda x: x
        ]
    )
    return transform_aug


def my_aug_noise(noise=0.3):
    transform_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32),
            get_color_distortion(),
            transforms.ToTensor(),
            gauss_noise_tensor(noise)

        ]
    )
    return transform_aug

def augmix():
    transform_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.75, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.AugMix(),
            transforms.ToTensor(),
        ]
    )
    return transform_aug


def random_Linf_noise(trnsfms: transforms.Compose = None, epsilon=64 / 255):
    if trnsfms is None:
        trnsfms = transforms.Compose([transforms.ToTensor()])

    randeps = torch.rand(1).item() * epsilon
    def fn(x):
        x = x + randeps * torch.randn_like(x).sign()
        return torch.clamp(x, 0, 1)

    trnsfms.transforms.append(fn)
    return trnsfms


def get_color_distortion(s=0.5):
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    


def random_erase(image, min_erase_ratio=0.1, max_erase_ratio=0.3, fill_color=(0, 0, 0)):
    """
    Randomly erases (fills with a color) a rectangular portion of the image.
    
    Args:
        image (PIL.Image): The input image.
        min_erase_ratio (float): Minimum ratio of the erase size compared to the original size.
        max_erase_ratio (float): Maximum ratio of the erase size compared to the original size.
        fill_color (tuple): RGB color to fill the erased area.
    
    Returns:
        PIL.Image: The image with a random part erased.
    """
    # Get original image dimensions
    width, height = image.size
    
    # Determine the erase size
    erase_width = random.randint(int(min_erase_ratio * width), int(max_erase_ratio * width))
    erase_height = random.randint(int(min_erase_ratio * height), int(max_erase_ratio * height))
    
    # Choose a random position to start the erase
    left = random.randint(0, width - erase_width)
    top = random.randint(0, height - erase_height)
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Define the area to be erased and fill it with the specified color
    erase_box = (left, top, left + erase_width, top + erase_height)
    draw.rectangle(erase_box, fill=fill_color)
    
    return image