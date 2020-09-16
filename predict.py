import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
import torch.nn.functional as F

from collections import OrderedDict
import PIL
import json

import utils
import sys

import argparse


args = argparse.ArgumentParser(description='Predict args')

args.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
args.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
args.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
args.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
args.add_argument('--category_names', dest="category_names", action="store", default = 'cat_to_name.json')
args.add_argument('--gpu', action="store_true", dest="gpu")

parser = args.parse_args()
input_image_path = parser.input
top_k = parser.top_k
is_gpu = parser.gpu
checkpoint_path = parser.checkpoint
category_mapping = parser.category_names

def predict():
    
    device = 'cuda' if torch.cuda.is_available() and is_gpu else 'cpu'
    print("Using Device -",device)
    
    print("start predicting...")
    model = utils.load_checkpoint(checkpoint_path)

    with open(category_mapping, 'r') as f:
        cat_to_name = json.load(f)

    top_probs, top_classes = utils.predict(input_image_path,model,top_k,device)
    
    for i in range(len(top_classes)):
        print("Class- {}, Probability- {:.3f}".format(top_classes[i], top_probs[i]))

    top_labels = [ cat_to_name[label] for label in top_classes ]
    
    if(category_mapping is not None):
        for i in range(len(top_labels)):
            print("Class- {}, Probability- {:.3f}".format(top_labels[i], top_probs[i]))
    print("Done predicting...")

if __name__ == "__main__":
    print(args.parse_args())
    predict()

