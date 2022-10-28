
from __future__ import annotations
import argparse
import cv2
import os
import glob
import numpy as np
import json
import random
import tqdm
import matplotlib.pyplot as plt

from pycocotools.coco import COCO  


class CocoSegDatasetReader():
    """
    @Class CocoSegDatasetReader can manipulate a coco dataset
    """

    def __init__(self, dataset_path,
                 image_type='png'):
        self.dataset_path = dataset_path
        self.dataset_result_path = dataset_path + 'result.json'
        self.coco = COCO(self.dataset_result_path)

        self.cat_ids = self.coco.getCatIds()
        self.cat_names = [self.coco.loadCats(cat_id)[0]['name'] for cat_id in self.cat_ids]

        self.img_ids = self.coco.getImgIds()

        # create masks from annotations if not exists
        self.create_mask()


    def create_mask(self):
        """
        Reads the dataset in the COCO format.
        """
        mask_dataset_path = os.path.join(self.dataset_path, 'masks')
        if os.path.exists(mask_dataset_path):
            print('Mask dataset already exists')
            return
        
        #create mask dataset path
        os.makedirs(mask_dataset_path)

        # read dataset
        for img_id in tqdm.tqdm(self.img_ids):
            img_info = self.coco.loadImgs([img_id])[0]
            img_name = img_info['file_name'].split('/')[-1]
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            mask = self.coco.annToMask(anns[0])
            for ann in anns:
                if ann['category_id'] in self.cat_ids:
                    mask += self.coco.annToMask(ann)

            mask_path = os.path.join(mask_dataset_path, str(img_name))
            cv2.imwrite(mask_path, mask)


    def load_coco_dataset(self):
        """
        Load coco dataset for training process
        Args:
            None
        Return: 
            - images: list of images masks
            - masks: list of annotations masks
        """
        images = []
        masks = []
        for img_id in tqdm.tqdm(self.img_ids):
            img_info = self.coco.loadImgs([img_id])[0]
            img_name = self.dataset_path + '/' + img_info['file_name'].replace('1/', '')
            print("img_name: {}".format(img_name))
            images.append(img_name)
            masks.append(img_name.replace('images', 'masks'))


        return images, masks



def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='COCO dataset reader')

    parser.add_argument('--dataset_path', type=str, default='../data/coco/annotations/instances_train2017.json',
                        help='Path to the dataset')

    return parser.parse_args()


def main():
    """
    Main function
    """
    print("pass here")
    args = parse_args()
    dataset_path = args.dataset_path
    
    reader = CocoSegDatasetReader(dataset_path)



if __name__ == '__main__':
    main()

