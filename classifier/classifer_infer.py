from base_test import BaseTest
from streetstyle_dataset import StreetStyleDataset
from newsAnchor_dataset import NewsAnchorDataset
from classifier_model import StreetStyleClassifier

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from data_utils import ResizeTransform

class StreetStyleClassifierInfer(BaseTest):

    def __init__(self, use_gpu=True):
        super(self.__class__, self).__init__(use_gpu)

    def create_data_loaders(self):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = NewsAnchorDataset('cloth_test', 'cloth_dict.pkl', batch_size=32, 
                                        transform=transform)
        self.infer_loader = dataset

    def imshow(self, inp):
        """Imshow for Tensor."""
        plt.figure()
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)

    def visualize_single_batch(self):
        # get some random training images
        images, manifest = self.infer_loader.next_infer()
        img = torchvision.utils.make_grid(images[:16], nrow=4)
        self.imshow(img)

    def load_model(self, checkpoint_name):
        self.model = torch.load(checkpoint_name)['best_model']
        if self.use_gpu:
            self.model = self.model.cuda()
            
    def infer(self):
        '''
        Use the best model to infer entire dataset.
        Returns infer result for each attribute.
        '''
        self.model.eval()
        infer_result = {}
        print("Inference starting...")
        iter_count = 0
        # get first batch
        images, manifest = self.infer_loader.next_infer()
        while images is not None:
            print("Infer iteration %d" % iter_count)
            
            if self.use_gpu:
                images = Variable(images.float().cuda(), requires_grad=False)
            else:
                images = Variable(images.float(), requires_grad=False)

            # classify mini-batch
            output = self.model(images)
            # calculate loss/accuracy
            for j, attrib_output in enumerate(output):
                _, predicted = torch.max(attrib_output, 1)
                video_name = manifest[j][0]
                pid = manifest[j][1]
                if not video_name in infer_result:
                    infer_result[video_name] = []
                if len(infer_result[video_name]) < pid:
                    infer_result[video_name].append([])
                infer_result[video_name][pid].append(predicted)
            iter_count += 1

            # next batch
            images, manifest = self.infer_loader.next_infer()

        return infer_result
