
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
import shutil

from pycocotools.coco import COCO
from merge_two_coco_json import combine


class Coco2VocConvert():
    """
    @Class Coco2VocConvert can manipulate a coco dataset
    """

    def __init__(self, dataset_path,
                 image_type='png'):

        self.dataset_path = dataset_path

        self.json_results = []
        self.abs_dirs = []

        self.images = []
        self.masks = []
        self.val_images = []
        self.masks_val = []

        # create masks from annotations if not exists
        self.create_mask()

        # parameters
        self.crop = True
        self.square_crop = [(180, 720), (0, 1280)]


    def create_mask(self):
        """
        Reads the dataset in the COCO format in each folder present in dataset path.
        """
        dirs = os.listdir(self.dataset_path)
        for dir in dirs:
            abs_dir_path = os.path.join(self.dataset_path, dir)
            self.abs_dirs.append(abs_dir_path)

            # create each masks inside each folders
            json_result_path = os.path.join(abs_dir_path, 'result.json')
            self.json_results.append(json_result_path)
            coco = COCO(json_result_path)
            cat_ids = coco.getCatIds()
            cat_names = [coco.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]
            img_ids = coco.getImgIds()
            print("cat_ids : {} // cat_names: {}".format(cat_ids, cat_names))
            
            print("Creating mask images")
            lbl_dir_path = os.path.join(abs_dir_path, 'labels')
            if os.path.exists(lbl_dir_path):
                print('Mask dataset already exists in {}'.format(abs_dir_path))
                continue
            
            #create mask dataset path
            os.makedirs(lbl_dir_path)

            # read dataset
            for img_id in tqdm.tqdm(img_ids):
                img_info = coco.loadImgs([img_id])[0]
                img_name = img_info['file_name'].split('/')[-1]

                ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
                anns = coco.loadAnns(ann_ids)
                mask = coco.annToMask(anns[0])
                for ann in anns:
                    if ann['category_id'] in cat_ids:
                        mask += coco.annToMask(ann)

                mask_path = os.path.join(lbl_dir_path, str(img_name))
                mask_chanels = cv2.merge((mask, mask, mask))
                cv2.imwrite(mask_path, mask_chanels)


    def create_voc_dataset(self, out_dataset):
        """
        Create VOC dataset
        Args:
            out_dataset: 
        Return: 
            - out_dataset: folder labels which contains labelled images
                           folder images which contains dataset images
                           train_x.txt: file contains images named to train dataset
                           val_x.txt: file contains images named to validation dataset
        """

        if os.path.exists(out_dataset):
                print('out_dataset already exists in {}'.format(out_dataset))
                return
        #create out dataset architecture
        os.makedirs(out_dataset)

        # create images folders 
        if self.crop: 
            img_cropped_data_dir = os.path.join(out_dataset, "cropped_images")
            lbl_cropped_data_dir = os.path.join(out_dataset, "cropped_labels")

        images_dataset_dir = os.path.join(out_dataset, "images")
        labels_dataset_dir = os.path.join(out_dataset, "labels")

        if os.path.exists(images_dataset_dir):
            print("it seems already images are created please verify at : {}".format(images_dataset_dir))
            return

        if self.crop:
            if os.path.exists(img_cropped_data_dir):
                print("it seems already cropped images are created please verify at : {}".format(img_cropped_data_dir))
                return
            os.makedirs(img_cropped_data_dir)
            os.makedirs(lbl_cropped_data_dir)
            
        os.makedirs(images_dataset_dir)
        os.makedirs(labels_dataset_dir)

        jsons_2_merge = []
        for dir in self.abs_dirs:
            images = sorted(glob.glob(os.path.join(dir, "images/*.png")))
            labels = sorted(glob.glob(os.path.join(dir, "labels/*.png")))
            jsons_2_merge.append(os.path.join(dir, 'result.json'))

            # check size consistency
            if len(images) != len(labels):
                print("nb of images should be equal to nb of labels")
                return

            for img, lbl in zip(images, labels):
                if self.crop:
                    # define parameters
                    img_name = img.split('/')[-1]
                    img_out_path = os.path.join(img_cropped_data_dir, img_name)
                    lbl_out_path = os.path.join(lbl_cropped_data_dir, img_name)

                    img_loaded = cv2.imread(img)
                    lbl_loaded = cv2.imread(lbl)
                    img_cropped = img_loaded[180:720, 0:1280]
                    lbl_cropped = lbl_loaded[180:720, 0:1280]

                    cv2.imwrite("{}".format(img_out_path), img_cropped)
                    cv2.imwrite("{}".format(lbl_out_path), lbl_cropped)

                shutil.copy(img, images_dataset_dir)
                shutil.copy(lbl, labels_dataset_dir)

        # merge and create final json
        json_id = 0
        id = 0
        json_to_store = os.path.join('/tmp', 'temp_json_{}.json'.format(json_id))
        first_json = jsons_2_merge[0]
        json_stored  = first_json
        for id in range(0, len(jsons_2_merge)):
            if id == 0:
                continue

            if id == 1:
                json_stored = combine(first_json, jsons_2_merge[id], json_to_store)
                json_id +=1
                json_to_store = os.path.join('/tmp', 'temp_json_{}.json'.format(json_id))
                continue

            json_stored = combine(json_stored, jsons_2_merge[id], json_to_store)
            json_id +=1
            json_to_store = os.path.join('/tmp', 'temp_json_{}.json'.format(json_id))

        # copy final json merged to out dataset
        shutil.copy(json_stored, out_dataset + '/result.json')

        # create train and val txt
        split_coef = 0.85
        train_file = os.path.join(out_dataset, 'train_{}.txt'.format(split_coef))
        val_file = os.path.join(out_dataset, 'val_{}.txt'.format(split_coef))
        
        list_images = [os.path.basename(x) for x in glob.glob(images_dataset_dir + '/*.png')]
        list_images = [s.replace('.png', '') for s in list_images]
        train_size = int(split_coef * len(list_images))
        train_images = list_images[:train_size]
        val_images = list_images[train_size:]

        with open(train_file, 'w') as fp:
            fp.write('\n'.join(train_images))

        with open(val_file, 'w') as fp:
            fp.write('\n'.join(val_images))


    def create_cropped_dataset(self, out_dataset):
        """
        Create a croppde dataset
        Args:
            out_dataset
        Return: 
            - images: list of images masks
            - masks: list of annotations masks
        """

        pass


    def load_train_val_coco_dataset(self):
        """
        Load coco dataset for training process
        Args:
            None
        Return: 
            - images: list of images masks
            - masks: list of annotations masks
        """
        print("Loading train val coco datasets")

        images = []
        masks = []
        for img_id in tqdm.tqdm(self.img_ids):
            img_info = self.coco.loadImgs([img_id])[0]
            info = img_info['file_name'].split('/')
            img_path = "{}/images/{}".format(self.dataset_path, info[-1])

            images.append(img_path)
            masks.append(img_path.replace('images', 'masks'))

        nb_images = len(images)
        train_size = int(0.6 * nb_images)
        
        self.images = images[:train_size]
        self.masks = masks[:train_size]
        self.val_images = images[train_size:]
        self.masks_val = masks[train_size:]


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='COCO dataset reader')

    parser.add_argument('-d', '--dataset_path', type=str, default='../data/coco/annotations/instances_train2017.json',
                        help='Path to the dataset')
    parser.add_argument('-o', '--out_dataset', type=str, default='/tmp/Dataset',
                        help='Path to the output dataset to create dataset')
    parser.add_argument('-c', '--create', action='store_true', 
                        help='Path to the dataset')
    return parser.parse_args()


def main():
    """
    Main function
    """
    args = parse_args()
    dataset_path = args.dataset_path

    reader = Coco2VocConvert(dataset_path)

    if args.create: 
        reader.create_voc_dataset(args.out_dataset)


if __name__ == '__main__':
    main()

