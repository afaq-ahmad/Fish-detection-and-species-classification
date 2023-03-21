# Author: afaq ahmad

"""
Simple Inference Script of EfficientDet-Pytorch
"""
Use_Gpu=True
use_cuda = True
use_float16 = False
from torch.backends import cudnn
cudnn.fastest = True
cudnn.benchmark = True

import time
import torch

from matplotlib import colors
import sys
from backbone import EfficientDetBackbone
import cv2
import numpy as np
import json
import os
import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pickle

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box,preprocess_video


force_input_size = None  # set None to use default size

detection_weights=sys.argv[1]
classification_model_path=sys.argv[2]
video_src = sys.argv[3]#'test/img.png'
compound_coef= int(sys.argv[4])# 3
#'model_best.pth.tar'
# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

obj_list = ['fish']

if not os.path.exists('test'):
    os.makedirs('test')
STANDARD_COLORS=STANDARD_COLORS+STANDARD_COLORS+STANDARD_COLORS+STANDARD_COLORS
color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(detection_weights, map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


def load_model(model_path,classes_no,Use_Gpu=True,gpu_no=0):
    model_classification=EfficientNet.from_name('efficientnet-b0',num_classes=classes_no)
    if os.path.isfile(model_path):
        if Use_Gpu is None:
            checkpoint = torch.load(model_path)
        else:
            # Map model to be loaded to specified single gpu.
            torch.cuda.set_device(gpu_no)
            model_classification = model_classification.cuda(gpu_no)
            loc = 'cuda:{}'.format(gpu_no)
            checkpoint = torch.load(model_path, map_location=loc)

        model_classification.load_state_dict(checkpoint['state_dict'])
    return model_classification

def model_prediction(model_classification,test_img,Use_Gpu=True,gpu_no=0):
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(test_img).unsqueeze(0)
    # Load ImageNet class names
    if Use_Gpu is not None:
        img = img.cuda(gpu_no, non_blocking=True)
    # Classify
    model_classification.eval()
    with torch.no_grad():
        outputs = model_classification(img)
    return outputs
    
with open("categories_names.pickle", "rb") as input_file:
    classes = pickle.load(input_file)


model_classification=load_model(classification_model_path,len(classes),Use_Gpu=Use_Gpu,gpu_no=0)


def display(preds, imgs,classes,img_path,frame_number,Specices_Counter, imshow=True, imwrite=False):
    Main_dict={}
    Main_dict['source_id']=img_path
    Main_dict['frame_id']=frame_number
    Main_dict['detection']=[]
    
    
    for i in range(len(imgs)):
        Specices_Counter_perframe={}
        Record_of_detection=[]
        if len(preds[i]['rois']) == 0:
            preds[i]={}
            continue
        
        # imgs[i] = imgs[i].copy()
        class_ids=[]
        for j in range(len(preds[i]['rois'])):
            detection_dict={}
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            test_img=imgs[i][y1:y2,x1:x2]
            
            # print(test_img.shape)
            obj=model_prediction(model_classification,Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)),Use_Gpu=Use_Gpu,gpu_no=0)
            obj=torch.topk(obj, k=1).indices.squeeze(0).tolist()[0]
            obj=classes[obj]
            class_ids.append(obj)
            # obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            
            detection_dict['x1']=int(x1)
            detection_dict['y1']=int(y1)
            detection_dict['x2']=int(x2)
            detection_dict['y2']=int(y2)
            detection_dict['score']=score
            detection_dict['species_id']=obj
            Record_of_detection.append(detection_dict)
            
            
            if obj in Specices_Counter.keys():
                Specices_Counter[obj]=Specices_Counter[obj]+1
            else:
                Specices_Counter[obj]=1
            
            if obj in Specices_Counter_perframe.keys():
                Specices_Counter_perframe[obj]=Specices_Counter_perframe[obj]+1
            else:
                Specices_Counter_perframe[obj]=1
            
            
            
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)
        cv2.putText(imgs[i], str(Specices_Counter_perframe), (30, imgs[i].shape[0]-30), 0, 1, [255, 255, 255],
                    thickness=2, lineType=cv2.FONT_HERSHEY_SIMPLEX)
        
        Main_dict['detection']=Record_of_detection
        
    return Main_dict,Specices_Counter,imgs

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

video_frame_folder=video_src.split('.')[0]
if not os.path.exists(video_frame_folder):
    os.makedirs(video_frame_folder)
    
# Video capture
cap = cv2.VideoCapture(video_src)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(video_src.split('.')[0]+'_output.avi',fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

fp=0
All_pred={}
All_pred['Results']=[]
Specices_Counter={}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame preprocessing
    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    # model predict
    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
                        
    out = invert_affine(framed_metas, out)
    # print(len(ori_imgs),ori_imgs[0].shape)
    Main_dict,Specices_Counter,img_show=display(out, ori_imgs,classes,video_src,fp,Specices_Counter, imshow=False, imwrite=True)
    # show frame by frame
    out_video.write(img_show[0])
    cv2.imwrite(video_frame_folder+'/'+str(fp)+'.png',img_show[0])
    fp=fp+1
    All_pred['Results'].append(Main_dict)
    All_pred['Specices_Counter']=Specices_Counter
    cv2.imshow('frame',img_show[0])
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
with open(video_src+'.json','w') as fp:
    json.dump(All_pred,fp, sort_keys=True,indent=4)
cap.release()

out_video.release()

cv2.destroyAllWindows()