""" Extract FaceNet(https://arxiv.org/pdf/1503.03832.pdf) feature
"""
import os
import sys
import argparse
import cPickle as pickle
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf
import PIPA_db
from facenet.src import facenet


def whiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def batch_iter(photos, batch_size, image_size, pre_whiten=True):
    images = []
    count = 0
    for i in range(len(photos)):
        photo = photos[i]
        img = skimage.io.imread(photo.file_path)
        for detection in photo.human_detections:
            bbox = detection.get_clipped_bbox()
            face_img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
            face_img = skimage.transform.resize(face_img, (image_size, image_size))
            if pre_whiten:
                face_img = whiten(face_img)
            images.append(face_img)
            count += 1
            if count == batch_size:
                yield np.array(images), count
                images = []
                count = 0
    if count != 0:
        yield np.array(images), count
    return


def load_facenet(model_dir):
    meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)
    facenet.load_model(model_dir, meta_file, ckpt_file)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    image_size = int(images_placeholder.get_shape()[1])
    embedding_size = int(embeddings.get_shape()[1])
    return (images_placeholder, embeddings, phase_train_placeholder), (image_size, embedding_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_load_dir', type=str, default='models/face_model')
    parser.add_argument('--feature_dump_path', type=str, default='feat/face.feat')
    args = parser.parse_args()
    batch_size = args.batch_size
    model_load_dir = args.model_load_dir
    feature_dump_path = args.feature_dump_path

    # initialize data manager
    print('initializing data manager...')
    manager = PIPA_db.Manager('PIPA')
    photos = manager.get_testing_photos()

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # build and load facenet
            tf_handlers, info = load_facenet(model_dir=model_load_dir)
            tf_images_placeholder, tf_embeddings, tf_phase_train_placeholder = tf_handlers
            image_size, feature_length = info

            # run inference
            completed = 0
            total = len(manager.get_testing_detections())
            face_features = []
            for face_images, count in batch_iter(photos, batch_size, image_size, pre_whiten=True):
                feed_dict = {tf_images_placeholder: face_images, tf_phase_train_placeholder: False}
                features = sess.run(tf_embeddings, feed_dict=feed_dict)
                features = features[:count, :]
                face_features.extend(features)
                completed += count
                print('{0}/{1} finished'.format(completed, total))

    # dump result
    face_features = np.array(face_features)
    print('dumping features...')
    fd = open(feature_dump_path, 'wb')
    pickle.dump(face_features, fd)
    fd.close()
    print('feature extraction finished.')
