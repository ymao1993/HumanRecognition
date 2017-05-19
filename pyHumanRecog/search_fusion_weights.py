""" PIPA data I/O
"""

import numpy as np
from numpy import linalg as LA
import random
import sklearn
import sklearn.metrics
import sklearn.preprocessing
from PIPA_db import Manager
import search_fusion_weights_config as config


def search_best_weights(manager):
    # create test splits
    photos = manager.get_testing_photos()
    print('randomly spliting test photos...')
    test_photos_0, test_photos_1 = split_test_photo(photos)
    test_detections_0 = Manager.get_detections_from_photos(test_photos_0)
    test_detections_1 = Manager.get_detections_from_photos(test_photos_1)

    # compute feature similarity
    feature_similarity = {}
    for feature_name in config.features:
        similarity = cal_feature_similarity(test_detections_0, test_detections_1, feature_name)
        feature_similarity[feature_name] = similarity

    weight_search_range = {}
    for feature_name in config.features:
        weight_search_range[feature_name] = config.features[feature_name]['weight_search_range']

    best_weights = None
    best_accuracy = 0
    for face_weight in weight_search_range['face']:
        for head_weight in weight_search_range['head']:
            for body_weight in weight_search_range['body']:
                for upperbody_weight in weight_search_range['upper-body']:
                    weights  = {'face': face_weight, 'head': head_weight,
                                'body': body_weight, 'upper-body': upperbody_weight}
                    print('experimenting weights config {0}...'.format(weights))
                    similarity = fuse_feature_similarity(feature_similarity, weights)
                    accuracy = evaluate_accuracy_with_inst_similarity(test_detections_0, test_detections_1, similarity)
                    print('--accuracy: {0} (highest: {1})'.format(accuracy, best_accuracy))
                    if best_accuracy < accuracy:
                        best_weights = weights
                        best_accuracy = accuracy
    return best_weights, best_accuracy


def evaluate_accuracy_with_inst_similarity(test_0, test_1, similarity):
    identity_set_0 = get_identity_set(test_0)
    identity_set_1 = get_identity_set(test_1)

    total_count = len(test_0) + len(test_1)
    correct = 0
    test_result_0to1 = np.argmax(similarity, axis=1)
    for i in range(len(test_0)):
        detection = test_0[i]
        gt_label = detection.identity_id
        predict = test_1[test_result_0to1[i]].identity_id
        if gt_label not in identity_set_1:
            total_count -= 1
        if predict == gt_label:
            correct += 1

    test_result_1to0 = np.argmax(similarity, axis=0)
    for i in range(len(test_1)):
        detection = test_1[i]
        gt_label = detection.identity_id
        predict = test_0[test_result_1to0[i]].identity_id
        if gt_label not in identity_set_0:
            total_count -= 1
        if predict == gt_label:
            correct += 1

    accuracy = float(correct)/total_count
    return accuracy


def get_identity_set(detections):
    identities = set()
    for detection in detections:
        identities.add(detection.identity_id)
    return identities


def split_test_photo(photos):
    random.shuffle(photos)
    test_0 = photos[0:len(photos)/2]
    test_1 = photos[len(photos)/2:len(photos)]
    return test_0, test_1


def cal_feature_similarity(detections_0, detections_1, feature_name):
    features_0 = get_features(detections_0, feature_name)    
    features_1 = get_features(detections_1, feature_name)
    features_0 = sklearn.preprocessing.normalize(features_0)
    features_1 = sklearn.preprocessing.normalize(features_1)
    similarity = sklearn.metrics.pairwise.cosine_similarity(features_0, features_1)
    similarity = 1.0/(1+np.exp(-(config.beta0 + config.beta1 * similarity)))
    return similarity


def get_features(detections, feature_name):
    features = np.empty((len(detections), config.features[feature_name]['length']), dtype=np.float64)
    for i in range(len(detections)):
        features[i] = detections[i].features[feature_name]
    return features


def fuse_feature_similarity(feature_similarity, weights):  
    mean_similarity = None
    for feature_name in feature_similarity:
        weight = weights[feature_name]
        similarity = feature_similarity[feature_name] * weight
        if mean_similarity is None:
            mean_similarity = similarity
        else:
            mean_similarity += similarity
    return mean_similarity


if __name__ == '__main__':   

    # initialize data manager
    print('initializing PIPA databse...')
    manager = Manager('PIPA')

    # load feature
    for feature_name in config.features:
        print('loading feature [{0}]...'.format(feature_name))
        manager.load_features(feature_name=feature_name,
                              feature_file=config.features[feature_name]['path'],
                              subset='test')

    # testing
    print('searching for weights...')
    (weights, accuracy) = search_best_weights(manager)

    # report result
    print('best weights found:')
    for feature_name in weights:
        print('{0}: {1}'.format(feature_name, weights[feature_name]))
    print('accuracy on test set: {0}'.format(accuracy))
