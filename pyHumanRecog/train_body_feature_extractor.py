""" Fine-tuning Inception-V3 with COCO loss
"""
import os
import sys
import argparse
import random

import tensorflow as tf
sys.path.append('./TFext/models/slim')
from datasets import dataset_utils
from nets import inception
from preprocessing import inception_preprocessing
slim = tf.contrib.slim

import PIPA_db
from coco_loss import coco_loss_layer


url = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
checkpoints_dir = '/home/mscvproject/users/yumao/humanRecog/yumao/pretrained_model'
checkpoint_name = 'inception_v3.ckpt'
original_variable_namescope = 'InceptionV3'
feature_length = 1024


image_size = inception.inception_v3.default_image_size


def densify_label(labels):
    result = []
    top_idx = 0
    labels_to_idx = {}
    for label in labels:
        if label not in labels_to_idx:
            labels_to_idx[label] = top_idx
            top_idx += 1
        result.append(labels_to_idx[label])
    return result


def get_minibatch(photos, batch_size):
    raw_img_data = []
    body_bbox = []
    labels = []
    while len(raw_img_data) < batch_size:
        photo = photos[random.randrange(0, len(photos))]
        detection = photo.human_detections[random.randrange(0, len(photo.human_detections))]
        raw_img_data.append(open(photo.file_path, 'rb').read())
        body_bbox.append(detection.get_estimated_body_bbox())
        labels.append(detection.identity_id)
    labels = densify_label(labels)
    return raw_img_data, body_bbox, labels


def download_inception_v3():
    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)
    if not tf.gfile.Exists(os.path.join(checkpoints_dir, checkpoint_name)):
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iteration', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss_print_freq', type=int, default=1)
    parser.add_argument('--summary_dir', type=str, default='./body_log')
    parser.add_argument('--model_save_dir', type=str, default='./body_model')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_save_freq', type=int, default=1000)
    args = parser.parse_args()
    max_iterations = args.max_iteration
    batch_size = args.batch_size
    loss_print_freq = args.loss_print_freq
    summary_dir = args.summary_dir
    model_save_dir = args.model_save_dir
    model_save_freq = args.model_save_freq
    model_load_dir = args.model_load_dir

    if not tf.gfile.Exists(summary_dir):
        tf.gfile.MakeDirs(summary_dir)
    if not tf.gfile.Exists(model_save_dir):
        tf.gfile.MkDir(model_save_dir)

    # data manager initialization
    print('initializing data manager...')
    download_inception_v3()
    manager = PIPA_db.Manager('PIPA')
    training_photos = manager.get_training_photos()
    total_detections = 0
    for photo in training_photos:
        total_detections += len(photo.human_detections)

    # building graph
    print('building graph...')
    graph = tf.Graph()
    with graph.as_default():

        # input
        tf_raw_image_data = tf.placeholder(tf.string, shape=(batch_size,))
        tf_body_bbox = tf.placeholder(tf.int32, shape=(batch_size, 4))
        tf_labels = tf.placeholder(tf.int32, shape=(batch_size,))

        # pre-processing pipeline
        crops = []
        for i in range(batch_size):
            image = tf.image.decode_jpeg(tf_raw_image_data[i], channels=3)
            body_crop = tf.image.crop_to_bounding_box(image, tf_body_bbox[i, 1], tf_body_bbox[i, 0], tf_body_bbox[i, 3],
                                                      tf_body_bbox[i, 2])
            processed_crop = inception_preprocessing.preprocess_image(body_crop, image_size, image_size,
                                                                      is_training=True)
            crops.append(processed_crop)
        processed_images = tf.stack(crops)

        # training pipeline
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, endpoints = inception.inception_v3(processed_images, num_classes=1001, is_training=True)

        # load model parameters
        varaibles = slim.get_model_variables(original_variable_namescope)
        init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, checkpoint_name),
                                                 slim.get_model_variables(original_variable_namescope))

        net_before_pool = tf.reshape(endpoints['Mixed_7c'], shape=(batch_size, -1))
        tf_features = slim.fully_connected(net_before_pool, feature_length, activation_fn=None)
        tf_loss = coco_loss_layer(tf_features, tf_labels, batch_size)

        # optimizer
        tf_lr = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(tf_loss)

        # summary
        tf.summary.scalar('coco_loss', tf_loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summary_dir)

        # global init
        init = tf.global_variables_initializer()

        # saver
        saver = tf.train.Saver()

    # model initialization
    print('initializing model...')
    sess = tf.Session(graph=graph)

    if model_load_dir is None:
        sess.run(init)
        init_fn(sess)
    else:
        model_full_path = os.path.join(model_load_dir, 'model.ckpt')
        print('restoring model from ' + model_full_path)
        saver.restore(sess, model_full_path)
        print('model restored.')

    # start training
    print('start training...')
    lr = 0.005
    epoch = 0
    for iter in range(max_iterations):
        raw_img_data, body_bbox, labels = get_minibatch(training_photos, batch_size=batch_size)
        _, loss, summary = sess.run([train, tf_loss, summary_op], feed_dict={tf_raw_image_data: raw_img_data,
                                                                             tf_body_bbox: body_bbox,
                                                                             tf_labels: labels,
                                                                             tf_lr: lr})
        summary_writer.add_summary(summary, global_step=iter)

        # count epoch
        if iter * batch_size > (epoch+1) * total_detections:
            epoch += 1

        # decrease the learning rate by 0.2 after 10 epochs
        if epoch == 10:
            lr *= 0.8

        # report loss
        if iter % loss_print_freq == 0:
            print('[iter: {0}, epoch: {1}] loss: {2}'.format(iter, epoch, loss))

        # save model
        if iter % model_save_freq == 0:
            print('saving model to ' + model_save_dir + '...')
            saver.save(sess, os.path.join(model_save_dir, 'model.ckpt'))
            print('model saved.')

    print('training finished.')
