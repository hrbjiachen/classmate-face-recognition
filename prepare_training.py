from preprocess import PreProcessor
import os
import facenet
import numpy as np


raw_img_folder = './data/raw_img'
train_img_folder = './data/train_img'
pp = PreProcessor()

# process all sub folders(people) under raw image folder


def processAllFolder():
    dataset = facenet.get_dataset(raw_img_folder)

    # looping the subfolders for all people
    for subfolder in dataset:
        output_class_dir = os.path.join(train_img_folder, subfolder.name)
        # create training image folder if not exists
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        # align all images for each person and save in train_img folder
        num_image = pp.align_dir(subfolder, output_class_dir)
        print("Aligned %s images from %s folder" % (num_image, subfolder.name))
    print("Image Preprocess All Done!")


# make sure folder is placed under raw_image_folder
# E.g processOneFolder('Jason Jia')
def processOneFolder(folder_name):
    input_class_dir = os.path.join(raw_img_folder, folder_name)
    output_class_dir = os.path.join(train_img_folder, folder_name)

    dataset = facenet.get_dataset(input_class_dir)

    # create training image folder if not exists
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    # align all images within this folder
    for img in dataset:
        img_path = os.path.join(input_class_dir, img.name)
        out_path = os.path.join(output_class_dir, img.name)
        pp.align(img_path, out_path)


processOneFolder('Ziming Mao')
