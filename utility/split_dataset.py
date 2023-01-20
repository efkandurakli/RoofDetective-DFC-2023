import os
import random
import shutil
import json   

RGB_ROOT = "data/track1/train-all/rgb"
SAR_ROOT = "data/track1/train-all/sar"
MERGE_ROOT = "data/track1/train-all/merge"
ANNOTATIONS_PATH = "data/track1/roof_fine_train.json"


def split_dataset(root):
    train_root = os.path.join(root, "train")
    if not os.path.exists(train_root):
        os.makedirs(train_root)

    val_root = os.path.join(root, "val")
    if not os.path.exists(val_root):
        os.makedirs(val_root)

    cities_dict = {}
    train_dict = {}
    val_dict = {}

    images = os.listdir(root)
    for image in images:
        if not os.path.isdir(os.path.join(root, image)):
            city_name = image.split("_")[1]
            if city_name in cities_dict:
                cities_dict[city_name].append(image)
            else:
               cities_dict[city_name] = [image]


    for city in cities_dict:
        random.shuffle(cities_dict[city])
        train_list = cities_dict[city][:round(len(cities_dict[city]) * 0.8)]
        val_list = cities_dict[city][round(len(cities_dict[city]) * 0.8):]

        train_dict[city] = train_list
        val_dict[city] = val_list
    
    for city in cities_dict:
        for train_image_file in train_dict[city]:
            shutil.move(os.path.join(root, train_image_file), os.path.join(root, "train"))
        for val_image_file in val_dict[city]:
            shutil.move(os.path.join(root, val_image_file), os.path.join(root, "val"))



  


def split_annotations(annotations_path, images_root):
    f = open(annotations_path)
    data = json.load(f)


    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    train_image_files = os.listdir(os.path.join(images_root, 'train'))
    val_image_files = os.listdir(os.path.join(images_root, 'val'))

    train_image_ids = [image['id'] for image in images if image['file_name'] in train_image_files]
    val_image_ids = [image['id'] for image in images if image['file_name'] in val_image_files] 


    train_image = [image for image in images if image['file_name'] in train_image_files]
    val_image = [image for image in images if image['file_name'] in val_image_files]  


    train_annotations = [annotation for annotation in annotations if annotation['image_id'] in train_image_ids]
    val_annotations = [annotation for annotation in annotations if annotation['image_id'] in val_image_ids]

    train_dict = {}
    val_dict = {}

    train_dict['images'] = train_image
    train_dict['annotations'] = train_annotations
    train_dict['categories'] = categories

    val_dict['images'] = val_image
    val_dict['annotations'] = val_annotations
    val_dict['categories'] = categories

    with open("data/track1/train-all/annotations/train.json", 'w') as json_file:
        json.dump(train_dict, json_file)

    with open("data/track1/train-all/annotations/val.json", 'w') as json_file:
        json.dump(val_dict, json_file)


    f.close()

split_annotations(ANNOTATIONS_PATH, RGB_ROOT)
#split_dataset(RGB_ROOT)
#split_dataset(SAR_ROOT)
#split_dataset(MERGE_ROOT)