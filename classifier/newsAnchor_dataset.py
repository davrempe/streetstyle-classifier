import os
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import random
import pickle


class NewsAnchorDataset(object):
    '''
    Handles loading and stratified sampling from the NewsAnchor dataset.
    '''
    attributes = [
        {'Solid': 0, 'Graphics' : 1, 'Striped' : 2, 'Floral' : 3, 'Plaid' : 4, 'Spotted' : 5}, # clothing_pattern
        {'Black' : 0, 'White' : 1, 'More than 1 color' : 2, 'Blue' : 3, 'Gray' : 4, 'Red' : 5,
                'Pink' : 6, 'Green' : 7, 'Yellow' : 8, 'Brown' : 9, 'Purple' : 10, 'Orange' : 11,
                'Cyan' : 12}, # major_color
        {'No': 0, 'Yes' : 1}, # wearing_necktie
        {'No': 0, 'Yes' : 1}, # collar_presence
        {'No': 0, 'Yes' : 1}, # wearing_scarf
        {'Long sleeve' : 0, 'Short sleeve' : 1, 'No sleeve' : 2}, # sleeve_length
        {'Round' : 0, 'Folded' : 1, 'V-shape' : 2}, # neckline_shape
        {'Shirt' : 0, 'Outerwear' : 1, 'T-shirt' : 2, 'Dress' : 3,
            'Tank top' : 4, 'Suit' : 5, 'Sweater' : 6}, # clothing_category
        {'No': 0, 'Yes' : 1}, # wearing_jacket
        {'No': 0, 'Yes' : 1}, # wearing_hat
        {'No': 0, 'Yes' : 1}, # wearing_glasses
        {'One layer': 0, 'Multiple layers' : 1} # multiple_layers
    ]

    def __init__(self, image_data_dir, manifest_data_path, batch_size=32, transform=None):
        '''
        - image_data_dir: directory holding the image data structure
        - manifest_data_path: file path holding the manifest file
        - batch_size: mini-batch size for sampling
        - transform: torchvision transforms to apply to each image
        '''
        self.image_data_dir = '../data/' + image_data_dir + '/'
        self.manifest_data_path = '../data/' + manifest_data_path
        self.batch_size = batch_size
        self.transform = transform
        # state variables for mini-batch sampling
        self.cur_infer_start_idx = 0
        # loaded necessary data
        self.preload()

    '''
    Performs necessary precomputation like loading manifest file information,
    and building structure for minibatch sampling.
    '''
    def preload(self):
        # load in file name for images of each anchor
        manifest_data = pickle.load(open(manifest_data_path, 'rb'))
        self.img_name_data = []
        self.img_meta_data = []
        for video_name, img_group in manifest_data.items():
            for pid, img_list in enumerate(img_group):
                for img_name in img_list:
                    self.img_name_data.append(img_name)
                    self.img_meta_data.append((video_name, pid))
        ## For new manifest format
        for meta in manifest_data:
            self.img_name_data.append(meta[0])
            self.img_meta_data.append((meta[1], meta[2]))
        
        self.num_images = len(self.img_name_data)
        self.infer_inds = range(0, self.num_images)

    def next_infer(self):
        '''
        Returns the next mini-batch from the the whole dataset or None if no
        more data is available. If None, resets so next time called will get the
        first test batch and so on.
        '''
        if self.cur_infer_start_idx == len(self.infer_inds):
            self.cur_infer_start_idx = 0
            return None, None

        images = None
        # start at cur infer start idx
        start_idx = self.cur_infer_start_idx
        end_idx = self.cur_infer_start_idx + self.batch_size
        # make sure doesn't go past end
        end_idx = end_idx if end_idx <= len(self.infer_inds) else len(self.infer_inds)
        self.cur_infer_start_idx = end_idx
        img_indices = self.infer_inds[start_idx:end_idx]
        manifest = self.img_meta_data[start_idx:end_idx]
        for i, img_idx in enumerate(img_indices):
            # load the image
            img = self.load_img(img_idx)
            if images is None:
                images = torch.Tensor(len(img_indices), img.shape[0], img.shape[1], img.shape[2])
            images[i] = img
        return images, manifest


    def load_img(self, img_idx):
        '''
        Loads the given image with corresponding crop and transform applied.
        '''
        img_name = self.img_name_data[img_idx]
        # load the image (located in path a/b/c/abcimagename.jpg)
        img_path = self.image_data_dir + img_name
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img