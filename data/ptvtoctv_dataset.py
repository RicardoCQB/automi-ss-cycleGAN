import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from torchvision.transforms import transforms
import numpy as np
import torch


class ptvtoctvdataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + 'A1')  # create a path '/path/to/data/trainA1'
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A2')  # create a path '/path/to/data/trainA2'
        self.dir_B1 = os.path.join(opt.dataroot, opt.phase + 'B1')  # create a path '/path/to/data/trainB1'
        self.dir_B2 = os.path.join(opt.dataroot, opt.phase + 'B2')  # create a path '/path/to/data/trainB2'

        self.A1_paths = sorted(
            make_dataset(self.dir_A1, opt.max_dataset_size))  # load images from '/path/to/data/trainA1'
        self.A2_paths = sorted(
            make_dataset(self.dir_A2, opt.max_dataset_size))  # load images from '/path/to/data/trainA2'
        self.B1_paths = sorted(
            make_dataset(self.dir_B1, opt.max_dataset_size))  # load images from '/path/to/data/trainB1'
        self.B2_paths = sorted(
            make_dataset(self.dir_B2, opt.max_dataset_size))  # load images from '/path/to/data/trainB2'

        self.A_size = len(self.A1_paths)  # get the size of dataset A
        self.B_size = len(self.B1_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=True)
        self.binary_transform = get_transform(self.opt, grayscale=False, binary=True)
        self.transform_B = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A1_path = self.A1_paths[index % self.A_size]  # make sure index is within then range
        A2_path = self.A2_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B1_path = self.B1_paths[index_B]
        B2_path = self.B2_paths[index_B]

        A1_img = Image.open(A1_path).convert('L')  # Load as grayscale
        A2_img = Image.open(A2_path).convert('L')  # Load as grayscale
        B1_img = Image.open(B1_path).convert('L')  # Load as grayscale
        B2_img = Image.open(B2_path).convert('L')  # Load as grayscale

        # Transform the images and binary images
        A1_img = self.transform_A(A1_img)
        A2_img = self.binary_transform(A2_img)
        B1_img = self.transform_B(B1_img)
        B2_img = self.binary_transform(B2_img)

        # Stack the tensors in a two channel tensor
        A = torch.cat((A1_img, A2_img), 0)
        B = torch.cat((B1_img, B2_img), 0)

        return {'A': A, 'B': B, 'A_paths': A1_path, 'B_paths': B1_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
