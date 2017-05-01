import os
import numpy as np
import skimage.io
import skimage.transform
import openface
import openface.helper
from openface.data import iterImgs
import dlib
import cv2
from PIPA_db import HumanDetection
from PIPA_db import Photo
from PIPA_db import Manager
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def head_detection_top(photos, dlibFacePredictor, f_head_annotate):
    align = openface.AlignDlib(dlibFacePredictor)
    head_total = 0
    for photo in photos:
        image = skimage.io.imread(photo.file_path)
        head_index = 0

        for head_id in photo.human_detections:
            front_head = check_front_head(photo, image, head_id, align, head_index)
            f_head_annotate.write(str(front_head) + '\n')
            head_total += 1
            head_index += 1

def check_front_head(photo, image, head_id, align, head_index):
    #crop head position
    xmin = int(head_id.head_bbox[0])
    ymin = int(head_id.head_bbox[1])
    width = int(head_id.head_bbox[2])
    height = int(head_id.head_bbox[3])    

    #crop to square/out of boundary
    if xmin < 0:
        x_left = 0
    else:
        x_left = xmin

    if ymin < 0:
        y_up = 0
    else:
        y_up = ymin


    if (xmin+width) > image.shape[1]:
        x_right = image.shape[1]
    else:
        x_right = xmin+width

    if (ymin+height) > image.shape[0]:
        y_down = image.shape[0]
    else:
        y_down = ymin+height

    new_width = x_right - x_left
    new_height = y_down - y_up

    if new_width > new_height:
        length = new_height
    else:
        length = new_width

    new_x_left = (x_left+x_right)/2  - length/2
    new_x_right = (x_left+x_right)/2  + length/2    
    new_y_up = (y_up+y_down)/2  - length/2
    new_y_down = (y_up+y_down)/2  + length/2    

    if (new_y_up >= new_y_down) or (new_x_left >= new_x_right):
        return 0

    head_image = image[new_y_up:new_y_down, new_x_left:new_x_right]
    bbs = align.getAllFaceBoundingBoxes(head_image)

    if not bbs:
        return 0
    else:
        return 1

if __name__ == '__main__':

    head_anno_train_file = os.path.join(config.head_path, 'annotations', 'train_head_annotate.txt')
    f_train_head_annotate = open(head_anno_train_file,'w')
    head_anno_valid_file = os.path.join(config.head_path, 'annotations', 'valid_head_annotate.txt')
    f_valid_head_annotate = open(head_anno_valid_file,'w')
    head_anno_test_file = os.path.join(config.head_path, 'annotations', 'test_head_annotate.txt')
    f_test_head_annotate = open(head_anno_test_file,'w')


    manager = Manager('PIPA')
    training_photos = manager.get_training_photos()
    validation_photos = manager.get_validation_photos()
    testing_photos = manager.get_testing_photos()

    #extract and align head
    head_detection_top(training_photos, config.dlibFacePredictor, f_train_head_annotate)
    head_detection_top(validation_photos, config.dlibFacePredictor, f_valid_head_annotate)
    head_detection_top(testing_photos, config.dlibFacePredictor, f_test_head_annotate)
   