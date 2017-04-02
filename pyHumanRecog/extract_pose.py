""" Extract pose from the PIPA dataset
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import cPickle as pickle

import skimage.io
import skimage.transform
import cv2
import tensorflow as tf
import sys
sys.path.append('.')
from CPM import cpm
import PIPA_db
import argparse


# picture size: all images will be resize to (PH, PW) when loaded
PH, PW = 376, 656
# CPM model weights
pose_pkl_file = 'models/CPM/_trained_MPI/params.pkl'
# size of the pose map (as the input to CPM posenet)
H, W = 376, 376


def project_detection_to_resized_img(detection, original_size, new_size):
    scale_h = float(new_size[0]) / original_size[0]
    scale_w = float(new_size[1]) / original_size[1]
    detection.scale(scale_h, scale_w)
    return


def gaussian_kernel(h, w, sigma_h, sigma_w):
    yx = np.mgrid[-h//2:h//2, -w//2:w//2]**2
    return np.exp(-yx[0, :, :] / sigma_h**2 - yx[1, :, :] / sigma_w**2)


def preprocess_img(image):
    return image[np.newaxis] / 255.0 - 0.5


def prepare_input_posenet(img, size, border=800, sigma=25):
    """
    prepare input for CPM pose-net.
    :param img: image
    :param size: expected input size (h,w) of CPM posenet
    :return: (human_sub_images, human_confidence_maps)
    From the photo we will be extracting a sub-region of the image containing the human head with
    fixed size (hopefully this will give us a good result)
    """
    img_size = (img.shape[1], img.shape[2])
    padded_img = np.zeros((1, img_size[0]+border, img_size[1]+border, 3))
    padded_img[:, border // 2:-border // 2, border // 2:-border // 2, :] = img
    num_human = len(photo.human_detections)
    human_sub_images = np.zeros((num_human, size[0], size[1], 3))
    human_confidence_maps = np.zeros((num_human, size[0], size[1], 1))
    for i in range(num_human):
        detection = photo.human_detections[i]
        head_y, head_x = detection.get_estimated_human_center()
        dh, dw = size[0]//2, size[1]//2
        y1 = int(border // 2 + head_y - dh); y2 = int(border // 2 + head_y + dh)
        x1 = int(border // 2 + head_x - dw); x2 = int(border // 2 + head_x + dw)
        human_sub_images[i, :, :, :] = padded_img[0, y1:y2, x1:x2, :]
        human_confidence_maps[i, :, :, 0] = gaussian_kernel(size[0], size[1], sigma, sigma)
    return human_sub_images, human_confidence_maps


def detect_parts_heatmaps(heatmaps, centers, size, num_parts=14):
  parts = np.zeros((len(centers), num_parts, 2), dtype=np.int32)
  for oid, (yc, xc) in enumerate(centers):
    part_hmap = skimage.transform.resize(np.clip(heatmaps[oid], -1, 1), size)
    for pid in xrange(num_parts):
      y, x = np.unravel_index(np.argmax(part_hmap[:, :, pid]), size)
      parts[oid,pid] = y+yc-size[0]//2, x+xc-size[1]//2
  return parts

LIMBS = np.array([1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14]).reshape((-1,2))-1
COLORS = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
          [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]]


def draw_limbs(image, parts):
  for oid in xrange(parts.shape[0]):
    for lid, (p0, p1) in enumerate(LIMBS):
      y0, x0 = parts[oid][p0]
      y1, x1 = parts[oid][p1]
      cv2.line(image, (x0,y0), (x1,y1), COLORS[lid], 2)


def draw_bbox(img, bbox):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)


def draw_marker(img, position):
    cv2.drawMarker(img, (position[1], position[0]), (0,255,0))


def tf_init_weights(root_scope, params_dict):
    names_to_values = {}
    for scope, weights in params_dict.iteritems():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '%s/%s' % (root_scope, scope))
        assert len(weights) == len(variables)
        for v, w in zip(variables, weights):
            names_to_values[v.name] = w
    return tf.contrib.framework.assign_from_values(names_to_values)


parser = argparse.ArgumentParser()
parser.add_argument('img_dump_folder', type=str)
parser.add_argument('pose_dump_folder', type=str)
args = parser.parse_args()
img_dump_folder = args.img_dump_folder
pose_dump_folder = args.pose_dump_folder

pose_params = pickle.load(open(pose_pkl_file))

tf.reset_default_graph()
with tf.variable_scope('CPM'):
    pose_image_in = tf.placeholder(tf.float32, [None, H, W, 3])
    pose_centermap_in = tf.placeholder(tf.float32, [None, H, W, 1])
    heatmap_pose = cpm.inference_pose(pose_image_in, pose_centermap_in)
    init_pose_op, init_pose_feed = tf_init_weights('CPM/PoseNet', pose_params)

db = PIPA_db.Manager('PIPA')
photos = db.get_photos()

for ii in range(len(photos)):
    print('processing {0}...'.format(photos[ii].file_path))
    photo = photos[ii]

    # load img
    img_path = photo.file_path
    img = skimage.io.imread(img_path)

    # preparing data
    img_size = (img.shape[0], img.shape[1])
    img = skimage.transform.resize(img, [PH, PW], preserve_range=True)
    for detection in photo.human_detections:
        project_detection_to_resized_img(detection, img_size, (PH, PW))
    img_processed = preprocess_img(img)
    pose_image, pose_cmap = prepare_input_posenet(img_processed, (H, W))

    # collect human body center
    centers = np.zeros((len(photo.human_detections), 2))
    for i in range(len(photo.human_detections)):
        cy, cx = photo.human_detections[i].get_estimated_human_center()
        centers[i, :] = [cy, cx]

    # running CPM evaluation
    with tf.Session() as sess:
        sess.run(init_pose_op, init_pose_feed)
        feed_dict = {pose_image_in: pose_image,
                     pose_centermap_in: pose_cmap}
        _hmap_pose = sess.run(heatmap_pose, feed_dict)

    # extract parts from heatmaps
    parts = detect_parts_heatmaps(_hmap_pose, centers, [H, W])

    # plot result
    for detection in photo.human_detections:
        draw_bbox(img, detection.head_bbox)
        draw_marker(img, detection.get_estimated_human_center())
    draw_limbs(img, parts)

    img = img.astype(np.uint8)
    skimage.io.imsave((img_dump_folder + '/pose_{0}.png').format(ii), img)

    # dump normalized parts
    for oid in xrange(parts.shape[0]):
        for lid, (p0, p1) in enumerate(LIMBS):
            y0, x0 = parts[oid][p0]
            y1, x1 = parts[oid][p1]
            y0 /= H; y1 /= H
            x0 /= W; x1 /= W
            parts[oid][p0] = [y0, x0]
            parts[oid][p1] = [y1, x1]
    pickle.dump(parts, open((pose_dump_folder + '/pose_{0}.parts.pkl').format(ii), 'wb'))
