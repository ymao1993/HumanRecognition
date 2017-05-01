import os
import sys
import tensorflow as tf
sys.path.append('./TFext/models/slim')
from datasets import dataset_utils
from nets import inception
from preprocessing import inception_preprocessing
from coco_loss import coco_loss_layer
slim = tf.contrib.slim

url = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
checkpoints_dir = '/home/mscvproject/users/yumao/humanRecog/yumao/pretrained_model'
checkpoint_name = 'inception_v3.ckpt'
original_variable_namescope = 'InceptionV3'
feature_length = 1024
image_size = inception.inception_v3.default_image_size


def build_network(batch_size, is_training):
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
                                                                  is_training=is_training)
        crops.append(processed_crop)
    processed_images = tf.stack(crops)

    # training pipeline
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, endpoints = inception.inception_v3(processed_images, num_classes=1001, is_training=is_training)

    # load model parameters
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, checkpoint_name),
                                             slim.get_model_variables(original_variable_namescope))

    net_before_pool = tf.reshape(endpoints['Mixed_7c'], shape=(batch_size, -1))
    tf_features = slim.fully_connected(net_before_pool, feature_length, activation_fn=None)
    tf_features_normalized = tf.nn.l2_normalize(tf_features, dim=1)
    tf_loss = coco_loss_layer(tf_features_normalized, tf_labels, batch_size)

    # optimizer
    tf_lr = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(tf_loss)

    # summary
    tf.summary.scalar('coco_loss', tf_loss)
    summary_op = tf.summary.merge_all()

    return (tf_raw_image_data, tf_body_bbox, tf_labels), (init_fn, tf_loss, tf_lr, train, summary_op), tf_features


def download_pretrained_model():
    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)
    if not tf.gfile.Exists(os.path.join(checkpoints_dir, checkpoint_name)):
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)