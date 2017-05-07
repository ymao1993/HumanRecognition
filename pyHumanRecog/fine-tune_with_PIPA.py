""" Fine-tuning Inception-V3 with PIPA, recognition as classification
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


url = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
checkpoints_dir = '/home/mscvproject/users/yumao/humanRecog/yumao/pretrained_model'
# checkpoints_dir = '/home/ilim/tiffany/11775_project/HumanRecognition/pretrained_model'
checkpoint_name = 'inception_v3.ckpt'
original_variable_namescope = 'InceptionV3'
image_size = inception.inception_v3.default_image_size
num_classes = 2356

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

    # 
    for itm in endpoints:
        print itm,':  ',endpoints[itm]
    # 

    # load model parameters
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, checkpoint_name),
                                             slim.get_model_variables(original_variable_namescope))


    # Add cls prediction FC layer
    net_before_pool = tf.reshape(endpoints['Mixed_7c'], shape=(batch_size, -1))
    cls_pred = slim.fully_connected(net_before_pool, num_classes, activation_fn=None)
    one_hot_labels = slim.one_hot_encoding(tf_labels, num_classes)
    slim.losses.softmax_cross_entropy(cls_pred, one_hot_labels)
    tf_loss = slim.losses.get_total_loss()


    # optimizer
    tf_lr = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(tf_loss)

    # Create some summaries to visualize the training process:
    tf.summary.scalar('cls_loss', tf_loss)
    summary_op = tf.summary.merge_all()

    return (tf_raw_image_data, tf_body_bbox, tf_labels), (init_fn, tf_loss, tf_lr, train, summary_op), cls_pred

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iteration', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss_print_freq', type=int, default=1)
    parser.add_argument('--summary_dir', type=str, default='./finetune_log')
    parser.add_argument('--model_save_dir', type=str, default='./finetune_model')
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

    # # download pre-trained model
    # print('downloading pre-trained model')
    # download_pretrained_model()

    # data manager initialization
    print('initializing data manager...')
    manager = PIPA_db.Manager('PIPA')
    training_photos = manager.get_training_photos()
    total_detections = 0
    for photo in training_photos:
        total_detections += len(photo.human_detections)

    # building graph
    print('building graph...')
    graph = tf.Graph()
    with graph.as_default():
        input_pack, train_pack, _ = build_network(batch_size=batch_size, is_training=True)
        tf_raw_image_data, tf_body_bbox, tf_labels = input_pack
        init_fn, tf_loss, tf_lr, train, summary_op = train_pack
        summary_writer = tf.summary.FileWriter(summary_dir)
        init = tf.global_variables_initializer()
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