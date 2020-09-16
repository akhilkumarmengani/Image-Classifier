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

import argparse


args = argparse.ArgumentParser(description='Training args')


args.add_argument('data_dir', action="store", default="./flowers/")
args.add_argument('--gpu', action="store_true", dest="gpu")
args.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
args.add_argument('--learning_rate', dest="learning_rate", action="store", type = float ,default=0.001)
args.add_argument('--dropout', dest = "dropout", action = "store", type = float ,default = 0.5)
args.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
args.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
args.add_argument('--hidden_units', dest="hidden_units", action="store", type = int , default=1024)


parser = args.parse_args()
data_dir = parser.data_dir
is_gpu = parser.gpu
learning_rate = parser.learning_rate
network = parser.arch
hidden_layer_1 = parser.hidden_units
epochs = parser.epochs
dropout = parser.dropout
save_dir = parser.save_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def train():
    
    device = 'cuda' if torch.cuda.is_available() and is_gpu else 'cpu'
    print("Using Device -",device)

    print("Start Training Process...\n")
    
    train_dataset = utils.tranform_train_dataset(train_dir)
    trainloader = utils.DataLoader(train_dataset,64)
    
    valid_dataset = utils.transform_valid_test_dataset(valid_dir)
    validloader = utils.DataLoader(valid_dataset, 64)
    
    test_dataset = utils.transform_valid_test_dataset(test_dir)
    testloader = utils.DataLoader(test_dataset, 64)
    
    fc_model = OrderedDict([('fc1',nn.Linear(25088,hidden_layer_1)),
                               ('relu',nn.ReLU()),
                               ('dropout1',nn.Dropout(dropout)),
                               ('fc2',nn.Linear(hidden_layer_1,hidden_layer_1)),
                               ('relu',nn.ReLU()),
                               #('fc3',nn.Linear(256,256)),
                               #('relu',nn.ReLU()),
                               ('fc4',nn.Linear(hidden_layer_1,102)),
                               ('output',nn.LogSoftmax(dim=1))
                           ])
    
    model = utils.build_network(fc_model, network, dropout, device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate)
    
    model = utils.train(model, trainloader, epochs, learning_rate, criterion,optimizer, device , validloader)
    model.class_to_idx = train_dataset.class_to_idx
    utils.save_checkpoint(model, optimizer, epochs, save_dir, network, hidden_layer_1, dropout, learning_rate)
    
    print("End Training Process...\n")
    
    print("Start Test Process...\n")
    
    utils.test(model,testloader,criterion,device)
    
    print("End Test Process...\n")

if __name__ == "__main__":
    print(args.parse_args())
    train()
                         



