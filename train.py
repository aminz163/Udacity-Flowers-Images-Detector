
#importing basic libraries


import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models,transforms, datasets

from torch import optim

import argparse
import json
from collections import OrderedDict

def get_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir' , type = str , default = 'flowers',help= 'path to folder of images' )
    parser.add_argument('--save_dir',type = str, default = 'checkpoint.pth', help = 'path to save trained model')
    parser.add_argument('--learning_rate' , type = float, default = 0.001 , help = 'learning rate value')
    parser.add_argument('--epochs', type = int, default = 10 , help = 'number of epochs')
    parser.add_argument('--batches' , type = int, default = 150, help = 'number of batches')
    parser.add_argument('--device' , type = str , default = 'gpu', help= 'using gpu for inference')
    parser.add_argument('--arch' , type = str , default = 'vgg19', help = 'model architecture')
    
    return parser.parse_args()



in_arg = get_input_arguments()

if in_arg.device == 'gpu':
    device = 'cuda'
    
else:
    device = 'cpu'


def gpu_check():
    print('pytorch version is {}'.format(torch.__version__))
    gpu_check = torch.cuda.is_available()
    
    if gpu_check:
        print('GPU Device is Available')
        
    else:
        warnings.warn('GPU NOT FOUND, PLEASE USE GPU TO TRAIN YOUR NETWORK')
        
    return gpu_check
    
   
checking_gpu = gpu_check()  
    

# loading data

data_dir = in_arg.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.RandomRotation(30)
                                       ,transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]) 

test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

train_datasets = datasets.ImageFolder(train_dir,transform = train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_transforms)


trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 40, shuffle= True)
validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
# setting up training model



training_models = {'vgg19':models.vgg19(pretrained = True),
                 'densenet121':models.densenet121(pretrained = True),
                 'resnet101': models.resnet101(pretrained = True)}

model = training_models.get(in_arg.arch , 'vgg19')
classifier = None
optimizer = None

if in_arg.arch == 'vgg19':
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,4096)),
                                       ('relu1',nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                       ('fc2',nn.Linear(4096, 1024)),
                                       ('relu2',nn.ReLU()),
                                        ('dropout2',nn.Dropout(0.2)),
                                       ('fc3',nn.Linear(1024, 102)),
                                       ('output',nn.LogSoftmax(dim= 1))]))
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)
        
elif in_arg.arch == 'densenet121':
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(1024,600)),
                                       ('relu1',nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                       ('fc2',nn.Linear(600, 200)),
                                       ('relu2',nn.ReLU()),
                                        ('dropout2',nn.Dropout(0.2)),
                                       ('fc3',nn.Linear(200, 102)),
                                       ('output',nn.LogSoftmax(dim= 1))]))
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)
        
elif in_arg.arch == 'resnet101':
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(2048,1024)),
                                       ('relu1',nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                       ('fc2',nn.Linear(1024, 600)),
                                       ('relu2',nn.ReLU()),
                                        ('dropout2',nn.Dropout(0.2)),
                                       ('fc3',nn.Linear(600, 102)),
                                       ('output',nn.LogSoftmax(dim= 1))]))
    model.fc = classifier
    optimizer = optim.Adam(model.fc.parameters(), lr = in_arg.learning_rate)

    
criterion = nn.NLLLoss()
model.to(device)


# function for validation
def validation(model, validationloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in validationloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output,labels).item()
        probs = torch.exp(output)
        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

# training process
from workspace_utils import active_session

with active_session():
    epochs = in_arg.epochs
    steps = 0
    
    printevery = in_arg.batches
    running_loss = 0
    for e in range(epochs):
        accuracy = 0
    
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps +=1
        
            optimizer.zero_grad()
            
            
            output = model.forward(images)
            
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item()
            probs = torch.exp(output)
            equality = (labels.data == probs.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
            if steps % printevery ==0:
                model.eval()
            
                with torch.no_grad():
                    test_loss, validation_accuracy = validation(model, validationloader, criterion)
                
                print('epoch:{}/{}...'.format(e+1,epochs),
                    "training Loss:{:.3f}..".format(running_loss/printevery),
                    "training accuracy:{:.3f}....".format(accuracy/len(trainloader)),
                     "validation accuracy:{:.3f}...".format(validation_accuracy/len(validationloader)),
                    "validation loss:{:.3f}".format(test_loss/len(validationloader)))
                running_loss = 0
                accuracy = 0
                model.train()


            
            
#saving the model

model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'state_dict':model.state_dict(),
             'class_to_idx': model.class_to_idx,
             "arch":in_arg.arch,
             'epochs': in_arg.epochs}

if in_arg.arch == 'resnet101':
    checkpoint['fc'] = model.fc
    
else:
    checkpoint['classifier'] = model.classifier
    
    
torch.save(checkpoint, 'checkpoint.pth')
    
print('Model(%s) has been saved to path(%s)'%(in_arg.arch,'checkpoint.pth'))   