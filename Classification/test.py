import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import pickle

parser = argparse.ArgumentParser(description='Model Prediction')
parser.add_argument('-ip', '--img_path', help='Image path for testing')


parser.add_argument('-mp', '--model_path', default='model_best.pth.tar', 
                    help='Trained Model Path (default:model_best.pth.tar)')
parser.add_argument('--Use_Gpu', action='store_true',
                    help='Use GPU or Run only on cpu')
parser.add_argument('-gn', '--gpu_no', default=0, 
                    help='Specific Gpu number for testing (default:0)')

args = parser.parse_args()
def load_model(model_path,num_classes,Use_Gpu=True,gpu_no=0):
    model=EfficientNet.from_name('efficientnet-b0',num_classes=num_classes)
    if os.path.isfile(model_path):
        if Use_Gpu is None:
            checkpoint = torch.load(model_path)
        else:
            # Map model to be loaded to specified single gpu.
            torch.cuda.set_device(gpu_no)
            model = model.cuda(gpu_no)
            loc = 'cuda:{}'.format(gpu_no)
            checkpoint = torch.load(model_path, map_location=loc)

        model.load_state_dict(checkpoint['state_dict'])
    return model

def model_prediction(model,test_img_path,Use_Gpu=True,gpu_no=0):
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open(test_img_path)).unsqueeze(0)
    # Load ImageNet class names
    if Use_Gpu is not None:
        img = img.cuda(gpu_no, non_blocking=True)
    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)
    return outputs


with open("categories_names.pickle", "rb") as input_file:
    classes = pickle.load(input_file)
    

model=load_model(args.model_path,len(classes),Use_Gpu=args.Use_Gpu,gpu_no=args.gpu_no)
outputs=model_prediction(model,args.img_path,Use_Gpu=args.Use_Gpu,gpu_no=args.gpu_no)

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=classes[idx], p=prob*100))