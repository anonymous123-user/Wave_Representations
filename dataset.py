import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
from itertools import combinations
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torchvision import datasets

from torch.utils.data import random_split

import pdb

from augs import get_color_distortion

# Some dataset code taken from https://github.com/autonomousvision/akorn/blob/main/source/data/augs.py

class MNISTSegmentationDataset(Dataset):

    def __init__(self, mnist_dataset, image_size):
        super().__init__()
        # Load MNIST dataset
        self.image_size = image_size
        #self.mnist = datasets.MNIST(data_path, train=train, download=False,
        #                          transform=transforms.ToTensor())
        self.mnist = mnist_dataset
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # Upsample image by 2x using bilinear interpolation
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), 
                          mode='bilinear', align_corners=False).squeeze(0)
        
        # Convert image to binary mask where 1=digit, 0=background
        mask = (img[0] > 0.5).long()  # Shape: (56, 56)
        
        # Set digit pixels to label+1 (so background=0, digit1=1, digit2=2, etc)
        mask[mask == 1] = label + 1
        
        return img, mask


class ShapeDataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


#def load_data(dataset, data_config, num_train, num_test, scale_min=0.7, transform_set='set1', normalize=True):
def load_data(dataset, data_config, num_train=None, num_test=None, scale_min=0.7, transform_set='set1', normalize=True):
    if dataset == 'two-shapes':
        x_train, y_train = load_two_shapes(data_config['train_path'], num_train)
        x_test, y_test = load_two_shapes(data_config['test_path'], num_test)
        # y: batch x n x n, values 0 1 2
    elif dataset == 'tetrominoes':
        x_train, y_train, x_test, y_test = load_tetrominoes(data_config['x_path'], 
                                                            data_config['y_path'],
                                                            data_config['z_path'],
                                                            num_train, 
                                                            num_test)
        return ShapeDataset(x_train, y_train), ShapeDataset(x_test, y_test)
    elif dataset == '2-4Shapes': # code for 2-4 Shapes taken from akorn repo
        trainset = get_2_4shapes_pair(
            fpath=data_config["train_path"],
            imsize=data_config['img_size'],
            scale_min=scale_min,
            transform_set=transform_set,
        )
        testset = get_2_4shapes(
            fpath=data_config["test_path"],
            imsize=data_config['img_size'],
        )
        return trainset, testset
    elif dataset == '2-4Shapes_supervised':
        trainset = get_2_4shapes(
            fpath=data_config["train_path"],
            imsize=data_config['img_size'],
        )
        testset = get_2_4shapes(
            fpath=data_config["test_path"],
            imsize=data_config['img_size'],
        )
        return trainset, testset
    elif dataset == 'pascal-voc':
        trainset, testset = load_pascal_voc(data_config['train_path'], 
                                            data_config['test_path'], 
                                            data_config['img_size'],
                                            normalize=normalize,
                                            )
        return trainset, testset
    elif dataset == 'sbd':
        trainset, testset = load_sbd(data_config['train_path'], 
                                     data_config['test_path'], 
                                     data_config['img_size']
                                     )
        return trainset, testset
    elif dataset == 'new_tetronimoes':
        x_train, y_train = load_new_tetrominoes(data_config['x_train_path'], 
                                                data_config['y_train_path'])
        x_test, y_test = load_new_tetrominoes(data_config['x_val_path'], 
                                              data_config['y_val_path'])
        return ShapeDataset(x_train, y_train), ShapeDataset(x_test, y_test)
    elif dataset == 'mnist':
        torch.manual_seed(42)
        trainset = datasets.MNIST(data_config['train_path'], train=True, download=False,
                                  transform=transforms.ToTensor())
        total_size = len(trainset)          # Total number of samples in the dataset
        train_size = int(0.85 * total_size)  # 85% for training
        val_size = total_size - train_size
        trainset, valset = random_split(trainset, [train_size, val_size])
        trainset = MNISTSegmentationDataset(trainset, data_config['img_size'])
        valset = MNISTSegmentationDataset(valset, data_config['img_size'])
        return trainset, valset
        #trainset = MNISTSegmentationDataset(data_config['train_path'], data_config['img_size'], train=True)
        #testset = MNISTSegmentationDataset(data_config['test_path'], data_config['img_size'], train=False)
        #return trainset, testset
    else:
        print(f"ERROR: {dataset} is not a valid dataset.")
        exit()
    return x_train, y_train, x_test, y_test


def load_new_tetrominoes(x_path, y_path):
    x, y = np.load(x_path), np.load(y_path)
    x = np.transpose(x, (0, 3, 1, 2)) / 255.0
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    return x.type(torch.float), y


def load_sbd(train_path, test_path, imsize):

    class CustomTransform:
        def __init__(self, size):
            self.to_tensor = transforms.ToTensor()

            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.target_transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor(),
            ])

        def __call__(self, image, target):
            # Apply ToTensor separately to image and target
            image = self.transform(image)
            if isinstance(target, np.ndarray):
                # Convert numpy array to PIL Image before transforming
                target = Image.fromarray(target)
            target = self.target_transform(target).squeeze(0) * 255
            return image, target
        
    
    # Instantiate the SBDataset with the custom transform
    trainset = datasets.SBDataset(
        root=train_path,  # Replace with the path where VOC dataset is stored    image_set='train',  # Options: 'train', 'val', 'train_noval'
        image_set='train',
        mode='segmentation',  # Options: 'boundaries', 'segmentation'
        download=False,  # Set to True if you need to download the dataset
        transforms=CustomTransform(size=imsize)  # Use the custom transform
    )

    # Instantiate the SBDataset with the custom transform
    testset = datasets.SBDataset(
        root=test_path,  # Replace with the path where VOC dataset is stored    image_set='train',  # Options: 'train', 'val', 'train_noval'
        image_set='val',
        mode='segmentation',  # Options: 'boundaries', 'segmentation'
        download=False,  # Set to True if you need to download the dataset
        transforms=CustomTransform(size=imsize)  # Use the custom transform
    )

    return trainset, testset


def load_pascal_voc(train_path, test_path, imsize, normalize):
    # Define the transformation for image resizing
    if normalize:
        image_transform = transforms.Compose([
            transforms.Resize((imsize, imsize)),  # Resize images to 64x64
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize((imsize, imsize)),  # Resize images to 64x64
            transforms.ToTensor()  # Convert image to a PyTorch tensor
        ])

    def target_transform(target):
        # Resize with NEAREST interpolation and convert to array
        target = target.resize((imsize, imsize), resample=Image.NEAREST)
        target = np.array(target)
        return torch.from_numpy(target)

    # Load the VOCSegmentation dataset with separate transforms for images and targets
    trainset = datasets.VOCSegmentation(
        root=train_path,  # Replace with the path where VOC dataset is stored
        year='2012',  # Specify the year of the dataset you want to use
        image_set='train',  # Can be 'train', 'val', 'trainval', or 'test'
        download=False,  # Set True to download if the dataset is not present
        transform=image_transform,  # Apply the transformation to images
        target_transform=target_transform  # Apply the transformation to targets
    )
    testset = datasets.VOCSegmentation(
        root=test_path,  # Replace with the path where VOC dataset is stored
        year='2012',  # Specify the year of the dataset you want to use
        image_set='val',  # Can be 'train', 'val', 'trainval', or 'test'
        download=False,  # Set True to download if the dataset is not present
        transform=image_transform,  # Apply the transformation to images
        target_transform=target_transform  # Apply the transformation to targets
    )
    return trainset, testset


class NumpyDataset(Dataset):
    """NpzDataset: loads a npz file as dataset."""

    def __init__(self, filename, transform=torchvision.transforms.ToTensor(), num=None):
        super().__init__()

        dataset = np.load(filename)
        self.images = dataset["images"].astype(np.float32)
        if self.images.shape[1] == 1:
            self.images = np.repeat(self.images, 3, axis=1) 
        self.pixelwise_instance_labels = dataset["labels"]

        if "class_labels" in dataset:
        #if "labels" in dataset:
            self.class_labels = dataset["class_labels"]
            #self.class_labels = dataset["labels"]
        else:
            self.class_labels = None

        if "pixelwise_class_labels" in dataset:
            self.pixelwise_class_labels = dataset["pixelwise_class_labels"]
        else:
            self.pixelwise_class_labels = None

        self.transform = transform


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # {"input_images": self.images[idx]}
        img = np.transpose(img, (1, 2, 0))
        img = (255 * img).astype(np.uint8)
        img = Image.fromarray(img)  # .convert('RGB')
        labels = {"pixelwise_instance_labels": self.pixelwise_instance_labels[idx]}

        if self.class_labels is not None:
            #labels["class_labels"] = self.class_labels[idx]
            labels["labels"] = self.class_labels[idx]
        if self.pixelwise_class_labels is not None:
            labels["pixelwise_class_labels"] = self.pixelwise_class_labels[idx]
        return self.transform(img), labels
    
class PairDataset(NumpyDataset):
    """Generate mini-batche pairs on CIFAR10 training set."""

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.transpose(img, (1, 2, 0))
        img = (255 * img).astype(np.uint8)
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs)  # stack a positive pair

def get_2_4shapes_pair(fpath, imsize=32, scale_min=0.7, transform_set='set1'):
    if transform_set == 'set1':
        transform = transforms.Compose(
            [
                transforms.Resize((imsize, imsize), interpolation=InterpolationMode.NEAREST),
                transforms.RandomResizedCrop(imsize, scale=(scale_min, 1.0)),
                get_color_distortion(s=0.5),
                transforms.ToTensor(),
            ]
        )
    elif transform_set == 'set2':
        transform = transforms.Compose(
            [
                transforms.Resize((imsize, imsize), interpolation=InterpolationMode.NEAREST),
                transforms.RandomRotation(180),
                transforms.ToTensor(),
            ]
        )
    else:
        print(f"ERROR: {transform_set} is not a valid transform set.")
        exit()
    return PairDataset(fpath, transform=transform)

def get_2_4shapes(fpath, imsize=32):
    return NumpyDataset(fpath, transform=transforms.Compose(
        [
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
        ]
    ))






















"""
x: num x 1 x 40 x 40
y: num x 40 x 40, with values
    0=background, -1=overlap, 1=triangle, 2=square, 3=circle, 4=diamond
"""
def load_24shapes_old(fpath, num=None):
    x = np.load(fpath)
    x, y = torch.from_numpy(x['images']), torch.from_numpy(x['labels'])
    x = x.type(torch.float)
    y = y.type(torch.LongTensor)
    if num is not None:
        x = x[0:num]
        y = y[0:num]
    #labels[labels == -1] = 2
    return x, y

"""
x: num x 1 x 32 x 32
y: num x 32 x 32, with values
    0=background, 1=square, 2=triangle
    intersection (originally -1) removed and replaced with triangle (2)
"""
def load_two_shapes(fpath, num=None):
    x = np.load(fpath)
    x, labels = torch.from_numpy(x['images']), torch.from_numpy(x['labels'])
    x = x.type(torch.float)
    labels = labels.type(torch.LongTensor).squeeze(1)
    if num is not None:
        x = x[0:num]
        labels = labels[0:num]
    labels[labels == -1] = 2
    return x, labels


"""
x: num x 3 x 32 x 32
y: num x 32 x 32, with values
    0, 1, 2, 3
"""
def load_tetrominoes(x_path, y_path, z_path, num_train, num_test):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    class_map = {0:0,
                    1:0,
                    2:1,
                    3:1,
                    4:1,
                    5:1,
                    6:1,
                    7:1,
                    8:1,
                    9:1,
                    10:2,
                    11:2,
                    12:2,
                    13:2,
                    14:3,
                    15:3,
                    16:3,
                    17:3,
                    18:4}
    combos = np.array(list(combinations(classes, 3)))

    x, y, z = np.load(x_path), np.load(y_path), np.load(z_path)
    x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
    x = x.type(torch.float)
    y = (y - 1).type(torch.LongTensor)
    z = z.type(torch.LongTensor)
    # z: num x 4 x 35 x 35 x 1
    z = z.squeeze(4)
    z = torch.argmax(z, dim=1) # num x 35 x 35

    x_result = []
    y_result = []
    z_result = []
    for combo in combos:
        found = np.equal(y, combo).sum(1)
        idx = np.where(found == 3)[0]
        x_result.append(x[idx])
        y_result.append(y[idx])
        z_result.append(z[idx])
    x = np.concatenate(x_result, axis=0)
    y = np.concatenate(y_result, axis=0)
    z = np.concatenate(z_result, axis=0)

    # Map from original indices
    y_mapped = []
    for i in range(len(y)):
        curr = [class_map[y_curr] for y_curr in y[i]]
        y_mapped.append(curr)
    y = np.array(y_mapped)

    # Conver to one-hot
    y = F.one_hot(torch.from_numpy(y), num_classes=5).sum(1)
    y = y.type(torch.float)

    # Remove repeats
    idx = torch.all((y != 2) & (y != 3), dim=1) # no repeats (each object should be unique)
    x = x[idx]
    y = y[idx]
    z = z[idx]

    # Randomize before splitting into train / test
    idx = np.random.choice(len(x), size=len(x), replace=False)
    x = torch.from_numpy(x[idx]).type(torch.float)
    z = torch.from_numpy(z[idx]).type(torch.LongTensor)
    y = y[idx]

    # Split into train / test
    x = torch.permute(x, (0, 3, 1, 2))
    #pdb.set_trace()
    #x = x.mean(1).unsqueeze(1)
    #x[x > 0] = 1.0
    x = x / 255.0
    x_train = x[0:num_train]
    y_train = y[0:num_train]
    x_test = x[num_train:num_train + num_test]
    y_test = y[num_train:num_train + num_test]
    z_train = z[0:num_train]
    z_test = z[num_train:num_train + num_test]
    return x_train, z_train, x_test, z_test