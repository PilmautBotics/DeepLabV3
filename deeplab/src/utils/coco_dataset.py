
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
        self.images_augmentation_path = os.path.join(dataset_path, 'images_augmented')
        self.mask_augmentation_path = os.path.join(dataset_path, 'masks_augmented')
        self.coco = COCO(self.dataset_result_path)

        self.cat_ids = self.coco.getCatIds()
        self.cat_names = [self.coco.loadCats(cat_id)[0]['name'] for cat_id in self.cat_ids]

        self.img_ids = self.coco.getImgIds()

        self.images = []
        self.masks = []
        self.val_images = []
        self.masks_val = []

        # create masks from annotations if not exists
        self.create_mask()

        # load coco dataset 
        self.load_train_val_coco_dataset()

        # data augmentation for training
        self.augment_data()
    

    def blur_images(self, image):
        """
        """ 
        img1 = image.copy()
        fsize = 9
        img1 = cv2.blur(img1,(fsize,fsize))
        
        img2 = image.copy()
        fsize = 5
        img2 = cv2.GaussianBlur(img2, (fsize, fsize), 0)
        
        img3 = image.copy()
        fsize = 3
        img3 = cv2.medianBlur(image, fsize)

        return img1, img2, img3


    def noisify_images(self, image):
        """
        """
        # gauss noisify
        image=image.copy() 
        mean=0
        st=0.7
        gauss = np.random.normal(mean, st, image.shape)
        gauss = gauss.astype('uint8')
        img1 = cv2.add(image, gauss)
        
        # noisify
        img2 = image.copy() 
        prob = 0.05
        if len(img2.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = img2.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
                
        probs = np.random.random(img2.shape[:2])
        img2[probs < (prob / 2)] = black
        img2[probs > 1 - (prob / 2)] = white


        return img1, img2

    def color_jitter(self, image):
        """
        """
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s2 = s 
        v3 = v

        if value >= 0:
            lim = 255 - value
            v3[v3 > lim] = 255
            v3[v3 <= lim] += value
            s2[s2 > lim] = 255
            s2[s2 <= lim] += value
        else:
            lim = np.absolute(value)
            v3[v3 < lim] = 0
            v3[v3 >= lim] -= np.absolute(value)
            s2[s2 < lim] = 0
            s2[s2 >= lim] -= np.absolute(value)

        final_hsv1 = cv2.merge((h, s, v3))
        final_hsv2 = cv2.merge((h, s2, v))

        img1= cv2.cvtColor(final_hsv1, cv2.COLOR_HSV2BGR)
        img2= cv2.cvtColor(final_hsv2, cv2.COLOR_HSV2BGR)

        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(image)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img3 = np.uint8(dummy)

        return img1, img2, img3


    def create_mask(self):
        """
        Reads the dataset in the COCO format.
        """
        print("Creating mask images")
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

            print("img_name: {}".format(img_name))
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            mask = self.coco.annToMask(anns[0])
            for ann in anns:
                if ann['category_id'] in self.cat_ids:
                    mask += self.coco.annToMask(ann)

            mask_path = os.path.join(mask_dataset_path, str(img_name))
            mask_chanels = cv2.merge((mask, mask, mask))
            cv2.imwrite(mask_path, mask_chanels)


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

        if os.path.exists(self.images_augmentation_path) == True or os.path.exists(self.mask_augmentation_path) == True:
            print('Augmented Image and Masks folder already exists')
            for img_path in glob.glob(self.images_augmentation_path + '/*'):
                images.append(img_path)
                masks.append(img_path.replace(self.images_augmentation_path, self.mask_augmentation_path))

        nb_images = len(images)
        train_size = int(0.6 * nb_images)
        
        self.images = images[:train_size]
        self.masks = masks[:train_size]
        self.val_images = images[train_size:]
        self.masks_val = masks[train_size:]


    def write_image_mask_pair(self, image, mask, type_name, img_name, mask_name, img_path, mask_path):
        """
        """
        cv2.imwrite("{}/{}-{}.png".format(img_path, type_name, img_name), image)
        cv2.imwrite("{}/{}-{}.png".format(mask_path, type_name, mask_name), mask)


    def augment_data(self):
        """
        Data augmentation
        """ 

        print("Augmenting data from coco dataset")

        # check if augmented data already processed
        if os.path.exists(self.images_augmentation_path) == True or os.path.exists(self.mask_augmentation_path) == True:
            print('Augmented Image and Masks folder already exists')
            return

        os.makedirs(self.images_augmentation_path)
        os.makedirs(self.mask_augmentation_path)

        for x, y in tqdm.tqdm(zip(self.images, self.masks), total=len(self.images)):
            name = x.split("/")[-1].split(".")
            """ Extracting the name and extension of the image and the mask. """
            image_name = name[0]
            image_extn = name[1]
            mask_name = name[0]
            mask_extn = name[1]

            """ Reading image and mask. """
            img_path = "{}/images/{}.{}".format(self.dataset_path, image_name, image_extn)
            mask_path = "{}/masks/{}.{}".format(self.dataset_path, mask_name, mask_extn)
            x = cv2.imread(img_path, cv2.IMREAD_COLOR)
            y = cv2.imread(mask_path, cv2.IMREAD_COLOR)

            """ Augmentation """
            # color jitter images
            img1, img2, img3 = self.color_jitter(x)
            self.write_image_mask_pair(img1, y, "jitter_one", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)
            self.write_image_mask_pair(img2, y, "jitter_two", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)
            self.write_image_mask_pair(img3, y, "jitter_three", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)

            # noisy images
            img1, img2 = self.noisify_images(x)
            self.write_image_mask_pair(img1, y, "noisy_gauss", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)
            self.write_image_mask_pair(img2, y, "noisy_rgb", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)

            # blur images 
            img1, img2, img3 = self.blur_images(x)
            self.write_image_mask_pair(img1, y, "blur", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)
            self.write_image_mask_pair(img2, y, "blur_gauss", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)
            self.write_image_mask_pair(img3, y, "blur_median", image_name, mask_name, self.images_augmentation_path, self.mask_augmentation_path)

        # reload dataset with other images
        self.load_train_val_coco_dataset()



            


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='COCO dataset reader')

    parser.add_argument('-d', '--dataset_path', type=str, default='../data/coco/annotations/instances_train2017.json',
                        help='Path to the dataset')

    return parser.parse_args()


def main():
    """
    Main function
    """
    args = parse_args()
    dataset_path = args.dataset_path
    
    reader = CocoSegDatasetReader(dataset_path)



if __name__ == '__main__':
    main()

