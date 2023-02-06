import os
import cv2
import numpy as np
import argparse
from pycocotools.coco import COCO


def min_index(arr1, arr2):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    for k in range(2):
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def polygonFromMask(maskedArr): 
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    return segmentation



def coco_to_yolo_txt(json_file_path, out_dir_path):

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    dataset = COCO(json_file_path)
    images = dataset.loadImgs(dataset.getImgIds())
    anns = dataset.loadAnns(dataset.getAnnIds())


    for img in images:
        file_name = img["file_name"]
        height = img["height"]
        width = img["width"]
        id = img["id"]

        annots = [ann for ann in anns if ann["image_id"] == id]

        f = open(os.path.join(out_dir_path, file_name[:-3] + "txt"), "w")
        print(os.path.join(out_dir_path, file_name[:-3] + "txt"))
        segments = []
        for ann in annots:
            segmentation = ann['segmentation']
            if type(segmentation) == dict:
                mask = dataset.annToMask(ann)
                segmentation = polygonFromMask(mask)
            

            cls = ann['category_id'] - 1
            if len(segmentation) > 1:
                s = merge_multi_segment(segmentation)
                s = (np.concatenate(s, axis=0) / np.array([width, height])).reshape(-1).tolist()
            else:
                s = [j for i in segmentation for j in i]
                s = (np.array(s).reshape(-1, 2) / np.array([width, height])).reshape(-1).tolist()
            s = [cls] + s
            if s not in segments:
                segments.append(s)

            last_iter=len(segments)-1
            line = *(segments[last_iter]),  
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    
        f.close()


  
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Coco segmentation annotations to yolo txt format")

    parser.add_argument('--out-dir-path', help='the path of output directory', required=True)
    parser.add_argument('--json-file-path', help='the path of json file', required=True)

    args = parser.parse_args()

    out_dir_path = args.out_dir_path
    json_file_path = args.json_file_path

    coco_to_yolo_txt(json_file_path, out_dir_path)