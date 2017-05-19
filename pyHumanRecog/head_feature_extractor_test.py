""" Test body feature extractor
"""
import os
import sys
import argparse
import cPickle as pickle
import numpy as np
import tensorflow as tf
sys.path.append('./TFext/models/slim')
slim = tf.contrib.slim
import PIPA_db
from head_feature_extractor_common import build_network


def batch_iter(photos, batch_size):
    raw_img_data = []
    head_bbox = []
    count = 0
    for i in range(len(photos)):
        photo = photos[i]
        raw = open(photo.file_path, 'rb').read()
        for j in range(len(photos[i].human_detections)):
            detection = photo.human_detections[j]
            count += 1
            raw_img_data.append(raw)
            head_bbox.append(detection.get_clipped_bbox())
            if count == batch_size:
                raw_img_data = np.array(raw_img_data)
                yield raw_img_data, head_bbox, count
                raw_img_data= []
                head_bbox = []
                count = 0
    if count != 0:
        # just repeat the last sample to ensure we are providing a batch with batch_size
        for _ in range(batch_size - count):
            raw_img_data.append(raw_img_data[count-1])
            head_bbox.append(head_bbox[count-1])
        raw_img_data = np.array(raw_img_data)
        yield raw_img_data, head_bbox, count
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_load_dir', type=str, default='models/head_model')
    parser.add_argument('--feature_dump_path', type=str, default='feat/head.feat')
    args = parser.parse_args()
    batch_size = args.batch_size
    model_load_dir = args.model_load_dir
    feature_dump_path = args.feature_dump_path

    # data manager initialization
    print('initializing data manager...')
    manager = PIPA_db.Manager('PIPA')
    photos = manager.get_testing_photos()

    # building graph
    print('building graph...')
    graph = tf.Graph()
    with graph.as_default():
        input_pack, _, tf_features = build_network(batch_size=batch_size, is_training=False)
        tf_raw_image_data, tf_body_bbox, tf_labels = input_pack
        saver = tf.train.Saver()

    sess = tf.Session(graph=graph)

    # load weights
    model_full_path = os.path.join(model_load_dir, 'model.ckpt')
    print('restoring model from ' + model_full_path)
    saver.restore(sess, model_full_path)
    print('model restored.')

    # start training
    fd = open(feature_dump_path, 'w')
    print('start inference...')
    all_features = []
    completed_count = 0
    total_count = 0
    for photo in photos:
        total_count += len(photo.human_detections)
    for raw_img_data, body_bbox, batch_count in batch_iter(photos, batch_size):
        features = sess.run(tf_features, feed_dict={tf_raw_image_data: raw_img_data,
                                                    tf_body_bbox: body_bbox})
        tmp = features[0:batch_count]
        print ('nan of tmp: ' + str(np.sum(np.isnan(tmp).astype(int))))

        all_features.extend(features[0:batch_count])
        completed_count += batch_count

        print('{0}/{1} completed.'.format(completed_count, total_count))

    # dump
    all_features = np.array(all_features)
    print('dumping features...')
    pickle.dump(all_features, fd)
    fd.close()
    print('inference finished.')
