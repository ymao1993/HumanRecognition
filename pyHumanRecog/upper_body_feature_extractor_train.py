""" Fine-tuning Inception-V3 with COCO loss
"""
import os
import argparse
import random
import numpy as np
import tensorflow as tf
import PIPA_db
from upper_body_feature_extractor_common import build_network
from upper_body_feature_extractor_common import download_pretrained_model


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
    upperbody_bbox = []
    labels = []
    while len(raw_img_data) < batch_size:
        photo = photos[random.randrange(0, len(photos))]
        detection = photo.human_detections[random.randrange(0, len(photo.human_detections))]
        raw_img_data.append(open(photo.file_path, 'rb').read())
        upperbody_bbox.append(detection.get_estimated_upper_body_bbox())
        labels.append(detection.identity_id)
    labels = densify_label(labels)
    return raw_img_data, upperbody_bbox, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iteration', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss_print_freq', type=int, default=1)
    parser.add_argument('--summary_dir', type=str, default='./upperbody_log')
    parser.add_argument('--model_save_dir', type=str, default='models/upperbody_model')
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

    # download pre-trained model
    print('downloading pre-trained model')
    download_pretrained_model()

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
        input_pack, train_pack, tf_features = build_network(batch_size=batch_size, is_training=True)
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
        _, loss, summary, features = sess.run([train, tf_loss, summary_op, tf_features], feed_dict={tf_raw_image_data: raw_img_data,
                                                                                          tf_body_bbox: body_bbox,
                                                                                          tf_labels: labels,
                                                                                          tf_lr: lr})
        summary_writer.add_summary(summary, global_step=iter)

        if np.isnan(features).any():
            assert False, 'feature validation check not passed!'

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
