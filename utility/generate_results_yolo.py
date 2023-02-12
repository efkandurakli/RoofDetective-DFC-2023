import os
import json
import cv2
from ultralytics import YOLO
import pycocotools.mask as mask_util
import numpy as np
import argparse


def create_results(dataset_root, model_path):

    model = YOLO(model_path)

    image_ids_file = open(os.path.join(dataset_root, "image_ids", "image_id_val.json"))

    data = json.load(image_ids_file)


    images = data["images"]
    results_dict = []

    for image_dict in images:
        image_file_path = os.path.join(dataset_root, "rgb", image_dict["file_name"])
        image = cv2.imread(image_file_path)
        results = model([image])
        masks = results[0].masks
        if masks is not None:
            masks = masks.data.cpu().numpy()
            boxes = results[0].boxes
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            bboxes = boxes.xywh.cpu().numpy().tolist()
            for i in range(masks.shape[0]):
                category_id = int(cls[i]) + 1
                bimask = masks[i]
                rlemask = mask_util.encode(np.asfortranarray(bimask,dtype=np.uint8))
                rlemask['counts'] = str(rlemask['counts'],'utf-8')
                results_dict.append({"image_id": image_dict["id"], "bbox": bboxes[i], "score": float(conf[i]), "category_id": category_id, "segmentation": rlemask})

    with open("results.json", "w") as outfile:
        json.dump(results_dict, outfile)

    image_ids_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Coco segmentation annotations to yolo txt format")

    parser.add_argument('--dataset-root', help='the root of the dataset', required=True)
    parser.add_argument('--model-path', help='the path of yolo model', required=True)

    args = parser.parse_args()

    dataset_root = args.dataset_root
    model_path = args.model_path

    create_results(dataset_root, model_path)