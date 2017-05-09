""" Fine-tuning Inception-V3 with PIPA, recognition as classification
"""
import os
import sys
import argparse
import random
import tensorflow as tf
import numpy as np

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
#num_classes = 2356
num_classes = 1409

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

def get_allbatch(photos, finetune_model):
    raw_img_data = []
    bbox = []
    labels = []

    for i in range(0, len(photos)):
        photo = photos[i]
        for j in range(0,len(photo.human_detections)):
            detection = photo.human_detections[j]
            
            if finetune_model == 'upper_body':
                raw_img_data.append(open(photo.file_path, 'rb').read())
                bbox.append(detection.get_estimated_upper_body_bbox())
                labels.append(detection.identity_id)
            elif finetune_model == 'face' and detection.is_face == 0:
                raw_img_data.append(open(photo.file_path, 'rb').read())
                x = int(detection.head_bbox[0])
                y = int(detection.head_bbox[1])
                w = int(detection.head_bbox[2])
                h = int(detection.head_bbox[3])
                x = np.clip(x, 0, photo.width)
                y = np.clip(y, 0, photo.height)
                w = np.clip(w, 0, photo.width - x)
                h = np.clip(h, 0, photo.height - y)

                bbox.append((x,y,w,h))
                labels.append(detection.identity_id) 
            elif finetune_model == 'body':
                raw_img_data.append(open(photo.file_path, 'rb').read())
                bbox.append(detection.get_estimated_body_bbox())
                labels.append(detection.identity_id)  
    
    return raw_img_data, bbox, labels


def get_minibatch(photos, batch_size, finetune_model):
    raw_img_data = []
    bbox = []
    labels = []
    while len(raw_img_data) < batch_size:
        photo = photos[random.randrange(0, len(photos))]
        detection = photo.human_detections[random.randrange(0, len(photo.human_detections))]
        
        if finetune_model == 'upper_body':
            raw_img_data.append(open(photo.file_path, 'rb').read())
            bbox.append(detection.get_estimated_upper_body_bbox())
            labels.append(detection.identity_id)
        elif finetune_model == 'face' and detection.is_face == 0:
            raw_img_data.append(open(photo.file_path, 'rb').read())
            x = int(detection.head_bbox[0])
            y = int(detection.head_bbox[1])
            w = int(detection.head_bbox[2])
            h = int(detection.head_bbox[3])
            x = np.clip(x, 0, photo.width)
            y = np.clip(y, 0, photo.height)
            w = np.clip(w, 0, photo.width - x)
            h = np.clip(h, 0, photo.height - y)

            bbox.append((x,y,w,h))
            labels.append(detection.identity_id) 
        elif finetune_model == 'body':
            raw_img_data.append(open(photo.file_path, 'rb').read())
            bbox.append(detection.get_estimated_body_bbox())
            labels.append(detection.identity_id)
    labels = densify_label(labels)
    return raw_img_data, bbox, labels

def split_data(photos):
    random.shuffle(photos)
    valid_set = photos[0:len(photos)/3]
    train_set = photos[len(photos)/3:len(photos)]
    return valid_set, train_set


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

    print ("")

    # load model parameters
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, checkpoint_name),
                                             slim.get_model_variables(original_variable_namescope))


    # Add cls prediction FC layer
    #net_before_pool = tf.reshape(endpoints['Mixed_7c'], shape=(batch_size, -1))
    #cls_pred = slim.fully_connected(net_before_pool, num_classes, activation_fn=None)
    #net_before_pool = tf.reshape(endpoints['PreLogits'], shape=(batch_size, -1))
    net_before_pool = endpoints['PreLogits']
    print  ("tf.shape(net_before_pool)",tf.shape(net_before_pool))
    cls_pred = slim.conv2d(net_before_pool, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')    
    cls_pred = tf.squeeze(cls_pred, [1, 2], name='SpatialSqueeze')
    #cls_pred = slim.fully_connected(net_before_pool, num_classes, activation_fn=None)
    
    one_hot_labels = slim.one_hot_encoding(tf_labels, num_classes)
    #slim.losses.softmax_cross_entropy(cls_pred, one_hot_labels)
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
    parser.add_argument('--summary_dir', type=str, default='./body_finetune_log')
    parser.add_argument('--model_save_dir', type=str, default='./body_finetune_model')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_save_freq', type=int, default=1000)
    parser.add_argument('--finetune_model', type=str, default='body')   

    args = parser.parse_args()
    max_iterations = args.max_iteration
    batch_size = args.batch_size
    loss_print_freq = args.loss_print_freq
    summary_dir = args.summary_dir
    model_save_dir = args.model_save_dir
    model_save_freq = args.model_save_freq
    model_load_dir = args.model_load_dir
    finetune_model = args.finetune_model

    


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

    manager.load_head_annotation('head_annotation')
    total_detections = 0
    for photo in training_photos:
        total_detections += len(photo.human_detections)

    #split data
    valid_set, train_set = split_data(training_photos)

    #extract valid data
    valid_img_data, valid_bbox, valid_labels = get_allbatch(valid_set, finetune_model)
    print ("len(valid_img_data)", len(valid_img_data))

    # building graph
    
    print('building graph...')
    graph = tf.Graph()
    with graph.as_default():
        input_pack, train_pack, cls_pred = build_network(batch_size=batch_size, is_training=True)
        tf_raw_image_data, tf_body_bbox, tf_labels = input_pack
        init_fn, tf_loss, tf_lr, train, summary_op = train_pack
        summary_writer = tf.summary.FileWriter(summary_dir)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # model initialization
    print('initializing model...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    sess = tf.Session(config=config, graph=graph)

    if model_load_dir is None:
        sess.run(init)
        init_fn(sess)
    else:
        model_full_path = os.path.join(model_load_dir, 'model.ckpt')
        print('restoring model from ' + model_full_path)
        saver.restore(sess, model_full_path)
        print('model restored.')

    raw_img_data, bbox, labels = get_minibatch(train_set, batch_size=batch_size, finetune_model=finetune_model)
    # start training
    print('start training...')
    lr = 0.0001
    epoch = 0   
    for iter in range(max_iterations):
        #raw_img_data, bbox, labels = get_minibatch(train_set, batch_size=batch_size, finetune_model=finetune_model)
        #print ("bbox", bbox)
        _, loss, summary = sess.run([train, tf_loss, summary_op], feed_dict={tf_raw_image_data: raw_img_data,
                                                                             tf_body_bbox: bbox,
                                                                             tf_labels: labels,
                                                                             tf_lr: lr})

        summary_writer.add_summary(summary, global_step=iter)

        # count epoch
        if iter * batch_size > (epoch+1) * total_detections:
            epoch += 1

        # decrease the learning rate by 0.2 after 10 epochs
        #if epoch == 10:
        #    lr *= 0.8

        # report loss
        if iter % loss_print_freq == 0:
            print('[iter: {0}, epoch: {1}] loss: {2}'.format(iter, epoch, loss))

        # save model
        if iter % model_save_freq == 0 and iter != 0:
            print('saving model to ' + model_save_dir + '...')
            saver.save(sess, os.path.join(model_save_dir, finetune_model +'.model.ckpt'))
            print('model saved.')
        
        if iter % (model_save_freq/10) == 0 and iter != 0: 
            pred_result = []
            true_result = []
            
            valid_cls_pred, valid_summary = sess.run([cls_pred, summary_op], feed_dict={tf_raw_image_data: raw_img_data,
                                                                             tf_body_bbox: bbox,
                                                                             tf_labels: labels})

            '''
            for i in range(0,len(valid_img_data)/batch_size):
                mini_valid_img_data = valid_img_data[i*batch_size:(i+1)*batch_size]
                mini_valid_bbox = valid_bbox[i*batch_size:(i+1)*batch_size]             
                mini_valid_labels = valid_labels[i*batch_size:(i+1)*batch_size]  

                valid_cls_pred, valid_summary = sess.run([cls_pred, summary_op], feed_dict={tf_raw_image_data: mini_valid_img_data,
                                                                                        tf_body_bbox: mini_valid_bbox,
                                                                                        tf_labels: mini_valid_labels})

                pred_result.extend(np.argmax(valid_cls_pred, axis=1))
                true_result.extend(mini_valid_labels)
            
            pred_result = np.array(pred_result)
            true_result = np.array(true_result)
            
            correct_num = np.count_nonzero(pred_result==true_result)
            print ("accuracy:", float(correct_num)/len(true_result))
            '''
            correct_num = np.count_nonzero(valid_cls_pred==labels)

            print ("accuracy:", float(correct_num)/len(labels))
            print ("pred:", valid_cls_pred)
            print ("labels:", labels)
            
            summary_writer.add_summary(valid_summary, global_step=iter)
    print('training finished.')
    
