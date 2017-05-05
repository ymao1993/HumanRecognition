""" PIPA data I/O
"""

import os
import numpy as np
import random
import sklearn
import sklearn.metrics
import config
from PIPA_db import Manager
import performance_test_config as config


def test_pipeline(manager):
    test_detections = manager.get_testing_detections()

    # compute feature similarity
    (test_0, test_1) = split_test_data(test_detections)
    feature_similarity = {}
    for feature_name in config.features:
        similarity = cal_feature_similarity(test_0, test_1, feature_name)
        feature_similarity[feature_name] = similarity
    similarity = fuse_feature_similarity(feature_similarity)

    # evaluate accuracy
    module_accuracy = {}
    accuracy = evaluate_accuracy(test_0, test_1, similarity)
    for feature_name in config.features:
        module_accuracy[feature_name] = evaluate_accuracy(test_0, test_1, feature_similarity[feature_name])
    return accuracy, module_accuracy


def evaluate_accuracy(test_0, test_1, similarity):
    identity_set_0 = get_identity_set(test_0)
    identity_set_1 = get_identity_set(test_1)

    total_count = len(test_0) + len(test_1)
    correct = 0
    test_result_0to1 = knn_vote(similarity)
    for i in range(len(test_0)):
        detection = test_0[i]
        gt_label = detection.identity_id
        predict = test_1[test_result_0to1[i]].identity_id
        if gt_label not in identity_set_1:
            total_count -= 1
        if predict == gt_label:
            correct += 1

    test_result_1to0 = knn_vote(np.transpose(similarity))
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


def knn_vote(similarity):
    predicts = []
    sorted_nns = np.argsort(similarity, axis=1)[:, ::-1]
    total_points_count = len(similarity[0])
    for i in range(len(similarity)):
        stats = np.zeros(total_points_count)
        for j in range(config.K):
            stats[sorted_nns[i][j]] += 1
        predict = np.argmax(stats)
        predicts.append(predict)
    return np.array(predicts)


def get_identity_set(detections):
    identities = set()
    for detection in detections:
        identities.add(detection.identity_id)
    return identities


def split_test_data(detections):
    random.shuffle(detections)
    test_0 = detections[0:len(detections)/2]
    test_1 = detections[len(detections)/2:len(detections)]
    return test_0, test_1


def cal_feature_similarity(detections_0, detections_1, feature_name):
    features_0 = get_features(detections_0, feature_name)    
    features_1 = get_features(detections_1, feature_name)
    similarity = sklearn.metrics.pairwise.cosine_similarity(features_0, features_1)
    similarity = 1.0/(1+np.exp(-(config.beta0 + config.beta1 * similarity)))
    return similarity


def get_features(detections, feature_name):
    features = []
    for detection in detections:    
        features.extend([detection.features[feature_name]])
    features = np.array(features)  
    return features    


def fuse_feature_similarity(feature_similarity):  
    mean_similarity = None
    for feature_name in feature_similarity:
        weight = config.features[feature_name]['weight']
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
    testing_photos = manager.get_testing_photos()
    test_detections = manager.get_testing_detections()

    # compute accuracy by chance
    print('computing accuracy by chance...')
    identities = set()
    for detection in test_detections:
        identities.add(detection.identity_id)
    accuracy_by_chance = 1./len(identities)

    # load feature
    for feature_name in config.features:
        print('loading feature [{0}]...'.format(feature_name))
        manager.load_features(feature_name=feature_name,
                              feature_file=config.features[feature_name]['path'],
                              subset='test')

    # testing
    print('testing...')
    (accuracy, module_accuracy) = test_pipeline(manager)

    # report
    print('--Evaluation Result--')
    print('fused model accuracy: %.2f'%(accuracy * 100) + '%')
    for feature_name in config.features:
        print('accuracy with only %s features: %.2f'%(feature_name, module_accuracy[feature_name] * 100) + '%')
    print('accuracy by chance: %.2f'% (accuracy_by_chance * 100) + '%')
    print('---------------------')