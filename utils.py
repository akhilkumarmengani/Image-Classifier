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



#Transforms for validation and testing sets
def valid_test_transforms():
    return transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
#Transforms for training
def train_transforms():
    return transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
def tranform_train_dataset(filepath):
    return  datasets.ImageFolder(filepath, transform=train_transforms())

def transform_valid_test_dataset(filepath):
    return datasets.ImageFolder(filepath, transform=valid_test_transforms())

# TODO: Using the image datasets and the trainforms, define the dataloaders
def DataLoader(dataset , size):
    return torch.utils.data.DataLoader(dataset, batch_size=size, shuffle=True) 

def category_to_name(filepath):
    cat_to_name = {}
    with open('filepath', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def build_network(fc_model ,trained_model, dropout=0.5,device= 'cuda'):
    
    model = None
    
    if( trained_model =='vgg16'):
        model = models.vgg16(pretrained=True)
    elif( trained_model == 'resnet18'):
        model = models.resnet18(pretrained=True)
    elif (trained_model == 'vgg13'):
        model = models.vgg13(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
     
    classifier = nn.Sequential(fc_model)
    model.classifier = classifier
    model.to(device)
    print("Done - {} Network with Classifier... ".format(trained_model))
    return model


def train(model,trainloader,epochs,lr,criterion,optimizer,device,validloader):
     
    running_loss = 0    
    steps = 0
    print('----------Start Training Model--------------')
    for e in range(epochs):
        running_loss = 0
        for images,labels in trainloader:
            steps+=1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        if ( (e+1) % 5 == 0):
            print('----------Start Validating Model--------------')
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for images,labels in validloader:
                    images,labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    loss = criterion(logps,labels)
                    test_loss += loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            print(f"Epoch {e+1}/{epochs}.."
                  f"Validation Loss: {test_loss/len(validloader):.3f}.."
                  f"Accuracy: {accuracy/len(validloader):.3f}..")
            model.train()
            print('----------End Validating Model--------------')

        print('Epoch - {},  Training loss: {:.7f}'.format(e+1, running_loss/len(trainloader)))

    print('----------End Training Model--------------')
    return model
    

def test(model,testloader,criterion,device):
    test_loss = 0
    accuracy = 0
    
    model.to(device)
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images ,labels  = images.to(device), labels.to(device)

            logps = model.forward(images)
            test_loss = criterion(logps,labels)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1,dim =1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        print("Test Loss - {:.3f}".format(test_loss/len(testloader)))
        print("Test Accuracy - {:.3f}".format(accuracy/len(testloader)))

        
def load_network(arch, dropout, hidden_layer_1,device= 'cuda'):
    
  
    model = None
    
    if( arch =='vgg16'):
        model = models.vgg16(pretrained=True)
    elif( arch == 'resnet18'):
        model = models.resnet18(pretrained=True)
    elif (arch == 'vgg13'):
        model = models.vgg13(pretrained=True)
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    fc_model = OrderedDict([('fc1',nn.Linear(25088,hidden_layer_1)),
                               ('relu',nn.ReLU()),
                               ('dropout1',nn.Dropout(dropout)),
                               ('fc2',nn.Linear(hidden_layer_1,hidden_layer_1)),
                               ('relu',nn.ReLU()),
                               ('fc4',nn.Linear(hidden_layer_1,102)),
                               ('output',nn.LogSoftmax(dim=1))
                           ])
        
    classifier = nn.Sequential(fc_model)
    model.classifier = classifier
    model.to(device)
    print("Done - Loaded Model from checkpoint... ")
    return model


def save_checkpoint(model, optimizer, epochs ,filepath, arch, hidden_layer_1, dropout, learning_rate):
    model.to('cpu')
    checkpoint = {
             'structure' : arch,
             'dropout' : dropout,
             'epochs': epochs,
             'hidden_layer_1' : hidden_layer_1,
             'learning_rate':learning_rate,
             'state_dict':model.state_dict(),
             'class_to_idx':model.class_to_idx,
             'optimizer_dict':optimizer.state_dict()}
    torch.save( checkpoint, filepath )
    print("Checkpoint Saved...")

def load_checkpoint(filepath):
    print("start loading arch..")
    checkpoint = torch.load(filepath)
    model = None
    model = load_network(checkpoint['structure'],checkpoint['dropout'], checkpoint['hidden_layer_1'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    print("Done- loading arch..")
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
        ])
    pil_image = PIL.Image.open(image)
    image = image_transforms(pil_image)
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to('cpu')
    model.eval()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()
    image.to('cpu')
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_probs ,top_labels = ps.topk(topk)
    
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    idx_to_class = { value : key for key, value in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    
    
    return top_probs, top_labels 


def sanity_check(image_path,model):
    flower_category = image_path.split('/')[2]
    
    
    plt.rcParams["figure.figsize"] = (6,10)
    plt.subplot(211)
    p_image = process_image(image_path)
    
    top_probs, top_labels = predict(p_image,model)
    cat_to_name =  category_to_name('cat_to_name.json')

    axs = imshow(p_image, ax = plt)
    axs.axis('off')
    axs.title(cat_to_name[str(flower_category)])
    axs.show()
    
    top_flowers  = [cat_to_name[label] for label in top_labels]
    
    plt.subplot(211)
    sns.barplot(x=top_probs, y=top_flowers, color=sns.color_palette()[0]);
    plt.show()
    
    




