import numpy as np
import skimage.io
import skimage.transform
import openface
import openface.helper
from openface.data import iterImgs
import dlib
import cv2

def head_extraction_top(photos, dlibFacePredictor, imgDim):
	align = openface.AlignDlib(dlibFacePredictor)
	
	for photo in photos:
		image = skimage.io.imread(photo.file_path)
		head_index = 0
		for head_id in photo.human_detections:
			align_head(photo, image, head_id, align, head_index, imgDim)
			head_index += 1


def align_head(photo, image, head_id, align, head_index, imgDim):
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

	#save head image
	#head_image = image[new_y_up:new_y_down, new_x_left:new_x_right]
	#head_file = head_path + photo.album_id + '_' + photo.photo_id + '_head_' + str(head_index) + '.jpg'
	#skimage.io.imsave(head_file, head_image)

	dlib_bbox = dlib.rectangle(left=new_x_left, top=new_y_up, right=new_x_right, bottom=new_y_down)
	alignedFace = align.align(imgDim, image, dlib_bbox, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
	outBgr = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2BGR)
	#save aligned head image
	#align_head_file = head_path + photo.album_id + '_' + photo.photo_id + '_aligned_head_' + str(head_index) + '.jpg'
	#cv2.imwrite(align_head_file, outBgr)


