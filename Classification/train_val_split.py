import shutil
import os
import random

split_r=0.1

Images_directry='Classif_Dataset/'
directory_train='Classif_Dataset/train/'
directory_test='Classif_Dataset/val/'

Folders=os.listdir(Images_directry)

if not os.path.exists(directory_train):
    os.makedirs(directory_train)
if not os.path.exists(directory_test):
    os.makedirs(directory_test)



for fold in range(len(Folders)):
    perclass_images=os.listdir(Images_directry+Folders[fold])

    random.seed(0)
    random.shuffle(perclass_images)
    
    
    if not os.path.exists(directory_train+Folders[fold]):
        os.makedirs(directory_train+Folders[fold])
    
    if not os.path.exists(directory_test+Folders[fold]):
        os.makedirs(directory_test+Folders[fold])
    
    for img in perclass_images[:int(len(perclass_images)*split_r)]:
        shutil.move(Images_directry+Folders[fold]+'/'+img,directory_test+Folders[fold]+'/'+img)
    
    for img in perclass_images[int(len(perclass_images)*split_r):]:
        shutil.move(Images_directry+Folders[fold]+'/'+img,directory_train+Folders[fold]+'/'+img)
        
    os.rmdir(Images_directry+Folders[fold])