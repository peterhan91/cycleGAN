import pandas as pd
import numpy as np
import os
import glob
import imageio
from skimage.transform import resize
from pathlib import Path


def normalize_to_plus_minus_one(img):
    img_zero = img - np.amin(img)
    img_one = img_zero / np.amax(img_zero)
    img_one_one = img_one * 2.0 - 1.0
    return img_one_one


chexpert_path = '/media/tianyu.han/mri-scratch/DeepLearning/Stanford_MIT_CHEST/CheXpert-v1.0'
train_csv_path = 'train.csv'
train_df = pd.read_csv(os.path.join(chexpert_path, train_csv_path))
lateral_df = train_df[train_df['Frontal/Lateral'] == 'Lateral']
counter = 0

for path in lateral_df['Path'].tolist():
    img_name = ['*_lateral.jpg', '*_frontal.jpg'] 
    basepath = '/media/tianyu.han/mri-scratch/DeepLearning/Stanford_MIT_CHEST/'
    
    for name in img_name:
        img_path = glob.glob(os.path.join(basepath+os.path.dirname(path), name))
        if img_path:
            img = imageio.imread(img_path[0])
            if img.ndim != 2:
                img = img[:,:,0]

            image_resize = resize(img, (1024, 1024), anti_aliasing=True)
            image_resize = normalize_to_plus_minus_one(image_resize)
            image_resize = np.clip(np.rint((image_resize + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
            image_name = str(counter) + '.png'
            dir_name = '../'+name[2:9]
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            imageio.imwrite(os.path.join(dir_name, image_name), image_resize)
        else:
            print('no image found under ', os.path.join(basepath+os.path.dirname(path), name))
        
    counter+=1

