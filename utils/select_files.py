import numpy as np
import os,shutil

def select_imgs():
    image_dir=r"H:\warship\images"
    according=r"H:\warship\masks"

    img_suffix=".png"
    new_image_dir=image_dir+"_new"
    os.makedirs(new_image_dir)

    for file in os.listdir(according):
        img_path=os.path.join(image_dir,file.replace('.json',img_suffix))
        new_path=os.path.join(new_image_dir,file.replace('.json',img_suffix))
        shutil.copy(img_path,new_path)

def main():
    select_imgs()

main()