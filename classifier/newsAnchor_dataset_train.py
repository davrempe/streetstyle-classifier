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
    attri_dict = [
        {'solid': 0, 'graphics' : 1, 'striped' : 2, 'floral' : 3, 'plaid' : 4, 'spotted' : 5}, # clothing_pattern
        {'black' : 0, 'white' : 1, 'more color' : 2, 'blue' : 3, 'gray' : 4, 'red' : 5,
                'pink' : 6, 'green' : 7, 'yellow' : 8, 'brown' : 9, 'purple' : 10, 'orange' : 11,
                'cyan' : 12, 'dark blue' : 13}, # major_color
        {'necktie no': 0, 'necktie yes' : 1}, # wearing_necktie
        {'collar no': 0, 'collar yes' : 1}, # collar_presence
        {'scarf no': 0, 'scarf yes' : 1}, # wearing_scarf
        {'long sleeve' : 0, 'short sleeve' : 1, 'no sleeve' : 2}, # sleeve_length
        {'round' : 0, 'folded' : 1, 'v-shape' : 2}, # neckline_shape
        {'shirt' : 0, 'outerwear' : 1, 't-shirt' : 2, 'dress' : 3,
            'tank top' : 4, 'suit' : 5, 'sweater' : 6}, # clothing_category
        {'jacket no': 0, 'jacket yes' : 1}, # wearing_jacket
        {'hat no': 0, 'hat yes' : 1}, # wearing_hat
        {'glasses no': 0, 'glasses yes' : 1}, # wearing_glasses
        {'one layer': 0, 'more layer' : 1}, # multiple_layers
        {'black' : 0, 'white' : 1, 'more color' : 2, 'blue' : 3, 'gray' : 4, 'red' : 5,
                'pink' : 6, 'green' : 7, 'yellow' : 8, 'brown' : 9, 'purple' : 10, 'orange' : 11,
                'cyan' : 12, 'dark blue' : 13}, # necktie_color
        {'solid' : 0, 'striped' : 1, 'spotted' : 2}, # necktie_pattern
        {'black' : 0, 'white': 1, 'blond' : 2, 'brown' : 3, 'gray' : 4}, # hair_color
        {'long' : 0, 'medium' : 1, 'short' : 2, 'bald' : 3} # hair_longth
    ]
    def __init__(self, image_data_dir, manifest_data_path, batch_size=32, transform=None,
                train_split_size=80, test_split_size=10):
        '''
        - image_data_dir: directory holding the image data structure
        - manifest_data_path: file path holding the manifest file
        - batch_size: mini-batch size for sampling
        - transform: torchvision transforms to apply to each image
        '''
        self.image_data_dir = image_data_dir
        self.manifest_data_path = manifest_data_path
        self.batch_size = batch_size
        self.transform = transform
        self.train_split_frac = train_split_size / 100.0
        self.eval_split_frac = eval_split_size / 100.0
        self.test_split_frac = 1 - self.train_split_frac - self.eval_split_frac
        # state variables for mini-batch sampling
        self.cur_attrib = 0
        self.cur_eval_start_idx = 0
        self.cur_test_start_idx = 0
        # loaded necessary data
        self.preload()

    '''
    Performs necessary precomputation like loading manifest file information,
    and building structure for minibatch sampling.
    '''
    def preload(self):
        # load in file name for images of each anchor
        manifest_data = pickle.load(open(self.manifest_data_path, 'rb'))
        self.img_name_data = []
        self.img_meta_data = []
        train_max = 0
        eval_max = 0
        for gid, img_group in enumerate(manifest_data):
            # for training and evaluation, collect all the images
            if 1. * gid / len(manifest_data) < 1 - self.test_split_frac:
                if 1. * gid / len(manifest_data) > self.train_split_frac and train_max == 0:
                    train_max = len(self.img_name_data) + 1
                for pid, img in enumerate(img_group):
                    self.img_name_data.append(img[2])
                    self.attribute_data.append(img[3])
            # for test, only collect one image per person 
            else:
                if eval_max == 0:
                    eval_max = len(self.img_name_data) + 1
                self.img_name_data.append(img_group[0][2])
                self.attribute_data.append(img_group[0][3])

        print("Total images: ", len(self.img_name_data))
        self.num_images = len(self.img_name_data)
   
        # create splits
        self.train_inds = range(0, train_max)
        self.eval_inds = range(train_max, eval_max)
        self.test_inds = range(eval_max, self.num_images)

        # build sampling structure:
        # each attribute has a dictionary pointing to all image indices which contain that attribute value
        self.attrib_inds = []
        for attrib in StreetStyleDataset.attributes:
            self.attrib_inds.append({x : [] for x in range(0, len(attrib))})
        # fill with training image attributes
        for train_idx in self.train_inds:
            attribs = self.attrib_data[train_idx]
            for attrib_idx, attrib_value in enumerate(attribs):
                if attrib_value != -1:
                    self.attrib_inds[attrib_idx][attrib_value].append(train_idx)

    def next_train(self):
        '''
        Returns the next training mini-batch using stratified sampling.
        '''
        labels = np.zeros((self.batch_size, len(StreetStyleDataset.attributes)))
        images = None
        for i in range(0, self.batch_size):
            # sample a random value to use for current attribute
            val = random.randint(0, len(StreetStyleDataset.attributes[self.cur_attrib])-1)
            # get a random image index with that value
            img_idx = self.attrib_inds[self.cur_attrib][val][random.randint(0, len(self.attrib_inds[self.cur_attrib][val])-1)]
            # update
            self.cur_attrib = (self.cur_attrib + 1) % len(StreetStyleDataset.attributes)
            # load the image
            img = self.load_img(img_idx)
            label = np.array(self.attrib_data[img_idx])
            # add image and label to batch
            if images is None:
                images = torch.Tensor(self.batch_size, img.shape[0], img.shape[1], img.shape[2])
            images[i] = img
            labels[i] = label
        return images, torch.from_numpy(labels)

    def next_eval(self):
        '''
        Returns the next mini-batch from the evaluation split or None if no
        more eval data is available. If None, resets so next time called will get the
        first eval batch and so on.
        '''
        if self.cur_eval_start_idx == len(self.eval_inds):
            self.cur_eval_start_idx = 0
            return None, None

        images = None
        # start at cur eval start idx
        start_idx = self.cur_eval_start_idx
        end_idx = self.cur_eval_start_idx + self.batch_size
        # make sure doesn't go past end
        end_idx = end_idx if end_idx <= len(self.eval_inds) else len(self.eval_inds)
        self.cur_eval_start_idx = end_idx
        img_indices = self.eval_inds[start_idx:end_idx]
        labels = np.zeros((len(img_indices), len(StreetStyleDataset.attributes)))
        for i, img_idx in enumerate(img_indices):
            # load the image
            img = self.load_img(img_idx)
            label = np.array(self.attrib_data[img_idx])
            if images is None:
                images = torch.Tensor(len(img_indices), img.shape[0], img.shape[1], img.shape[2])
            images[i] = img
            labels[i] = label
        return images, torch.from_numpy(labels)

    def next_test(self):
        '''
        Returns the next mini-batch from the test split or None if no
        more test data is available. If None, resets so next time called will get the
        first test batch and so on.
        '''
        if self.cur_test_start_idx == len(self.test_inds):
            self.cur_test_start_idx = 0
            return None, None

        images = None
        # start at cur eval start idx
        start_idx = self.cur_test_start_idx
        end_idx = self.cur_test_start_idx + self.batch_size
        # make sure doesn't go past end
        end_idx = end_idx if end_idx <= len(self.test_inds) else len(self.test_inds)
        self.cur_test_start_idx = end_idx
        img_indices = self.test_inds[start_idx:end_idx]
        labels = np.zeros((len(img_indices), len(StreetStyleDataset.attributes)))
        for i, img_idx in enumerate(img_indices):
            # load the image
            img = self.load_img(img_idx)
            label = np.array(self.attrib_data[img_idx])
            if images is None:
                images = torch.Tensor(len(img_indices), img.shape[0], img.shape[1], img.shape[2])
            images[i] = img
            labels[i] = label
        return images, torch.from_numpy(labels)    
   

    def load_img(self, img_idx):
        '''
        Loads the given image with corresponding crop and transform applied.
        '''
        img_name = self.img_name_data[img_idx]
        # load the image (located in path a/b/c/abcimagename.jpg)
        img_path = os.path.join(self.image_data_dir, img_name)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img