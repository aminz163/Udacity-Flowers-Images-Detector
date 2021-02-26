#importing basic libraries


import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import models,transforms, datasets
import argparse
import json


# getting input arguments
def get_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir' , type = str , default = 'checkpoint.pth',help= 'path to model' )
    parser.add_argument('--img_dir' , type = str , default = 'flowers/train/1/image_06734.jpg', help = 'path to required image')
    parser.add_argument('--category_names' , type = str , default = 'cat_to_name.json' , help= 'category to names')
    parser.add_argument('--K', type = int, default = 5, help = 'top K Ranking')
    parser.add_argument('--device' , type = str , default = 'gpu', help= 'using gpu for inference')
    
    return parser.parse_args()



in_arg = get_input_arguments()

# setting device

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
    

#loading pretrained model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    training_models = {'vgg19':models.vgg19(pretrained = True),
                 'densenet121':models.densenet121(pretrained = True),
                 'resnet101': models.resnet101(pretrained = True)}
    
    model = training_models.get(checkpoint['arch'] , 'vgg19')
    
    if checkpoint['arch'] == 'vgg19' or checkpoint['arch'] == 'densenet121':
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
                              
    else:
        model.fc = checkpoint['fc']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
                    
    return model



model = load_checkpoint(in_arg.model_dir)
model.to(device)


#acquiring image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    from PIL import Image
    
    im = Image.open(image)
    im_transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.RandomRotation(30)
                                       ,transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]) 

    final_img = im_transform(im)
    
    return final_img




def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = img.to(device)
    img = img.unsqueeze_(0)
    model.eval()
    with torch.no_grad():
        img_output = model.forward(img)
        ps = torch.exp(img_output)
        class_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}
        top_p, top_class = ps.topk(topk, dim=1)
        prob = [p.item() for p in top_p[0].data]
        classes = [class_idx_dict[i.item()] for i in top_class[0].data]
        model.train()
        
        
    return prob,classes




#prediction process




with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

image_path = in_arg.img_dir


result = process_image(image_path)

predictions, classes = predict(image_path, model, topk= in_arg.K)

names = [cat_to_name[str(c)] for c in classes]
print('Predictions = ', predictions)
print('\n')
print('Classes = ', names)