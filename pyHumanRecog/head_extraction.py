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

def head_extraction_top(photos, dlibFacePredictor, imgDim, head_dir):
	align = openface.AlignDlib(dlibFacePredictor)


	for photo in photos:
		image = skimage.io.imread(photo.file_path)
		head_index = 0
		for head_id in photo.human_detections:
			align_head(photo, image, head_id, align, head_index, imgDim, head_dir)
			head_index += 1


def align_head(photo, image, head_id, align, head_index, imgDim, head_dir):
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
		return

	#save head image
	if config.save_head_image == True:
		head_image = image[new_y_up:new_y_down, new_x_left:new_x_right]
		head_file = os.path.join(head_dir,  photo.album_id + '_' + photo.photo_id + '_head_' + str(head_index) + '.jpg')
		skimage.io.imsave(head_file, head_image)

	#save aligned head
	if config.save_aligned_head_image == True:
		dlib_bbox = dlib.rectangle(left=new_x_left, top=new_y_up, right=new_x_right, bottom=new_y_down)
		alignedFace = align.align(imgDim, image, dlib_bbox, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
		if image.ndim == 3:
			alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2BGR)
		align_head_file = os.path.join(head_dir,  photo.album_id + '_' + photo.photo_id + '_aligned_head_' + str(head_index) + '.jpg')
		cv2.imwrite(align_head_file, alignedFace)

if __name__ == '__main__':
	head_dir = os.path.join(config.head_path, 'head')

	train_head_dir = os.path.join(head_dir, 'train')
	valid_head_dir = os.path.join(head_dir, 'valid')		
	test_head_dir = os.path.join(head_dir, 'test')

	if not os.path.exists(head_dir):
		os.mkdir(head_dir)
	if not os.path.exists(train_head_dir):	
		os.mkdir(train_head_dir)
	if not os.path.exists(valid_head_dir):
		os.mkdir( valid_head_dir )
	if not os.path.exists(test_head_dir):	
		os.mkdir( test_head_dir )

	manager = Manager('PIPA')
	training_photos = manager.get_training_photos()
	validation_photos = manager.get_validation_photo()
	testing_photos = manager.get_testing_photos()

	#extract and align head
	head_extraction_top(training_photos, config.dlibFacePredictor, config.imgDim, train_head_dir)
	head_extraction_top(validation_photos, config.dlibFacePredictor, config.imgDim, valid_head_dir)
	head_extraction_top(testing_photos, config.dlibFacePredictor, config.imgDim, test_head_dir)  
