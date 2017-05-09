""" PIPA data I/O
"""

import numpy as np
import random
import sklearn
import sklearn.metrics
import sklearn.preprocessing
from PIPA_db import Manager
import performance_test_config as config
import CRF_opt


def test_pipeline(manager):
    # create test splits
    photos = manager.get_testing_photos()
    test_photos_0, test_photos_1 = split_test_photo(photos)
    test_detections_0 = Manager.get_detections_from_photos(test_photos_0)
    test_detections_1 = Manager.get_detections_from_photos(test_photos_1)

    # compute feature similarity
    feature_similarity = {}
    for feature_name in config.features:
        similarity = cal_feature_similarity(test_detections_0, test_detections_1, feature_name)
        feature_similarity[feature_name] = similarity
    similarity = fuse_feature_similarity(feature_similarity)

    # compute accuracy with each modality
    module_accuracy = {}
    for feature_name in config.features:
        module_accuracy[feature_name] = evaluate_accuracy_with_inst_similarity(test_detections_0, test_detections_1, feature_similarity[feature_name])

    # use CRF to refine the predicted results
    if config.refine_with_photo_level_context:
        print('refining prediction with CRF...')
        num_identity = manager.get_num_labels_testing()
        lbl_map_global_to_test = manager.get_label_mapping_global_to_test()
        lbl_map_test_to_global = manager.get_label_mapping_test_to_global()
        cls_scores0, cls_scores1 = convert_inst_scores_to_cls_scores(similarity, test_detections_0, test_detections_1, num_identity, lbl_map_global_to_test)

        # create CRF optimizer
        crf = CRF_opt.CRFOptimizer()

        # refine the predictions photo-by-photo for test split 0
        print('--running CRF on test split 0')
        crf.build_compat_func(test_photos_1, lbl_map_global_to_test)
        cur = 0
        predicts_0 = []
        for photo in photos:
            cls_scores_photo = cls_scores0[cur: cur + len(photo.human_detections)]
            predicts_photo = crf.run_LBP(cls_scores_photo)
            predicts_0.extend(predicts_photo)
            cur += len(photo.human_detections)
        predicts_0 = [lbl_map_test_to_global[predict] for predict in predicts_0]
        predicts_0 = np.array(predicts_0)

        # refine the predictions photo-by-photo for test split 1
        print('--running CRF on test split 1')
        crf.build_compat_func(test_photos_0, lbl_map_global_to_test)
        cur = 0
        predicts_1 = []
        for photo in photos:
            cls_scores_photo = cls_scores1[cur: cur + len(photo.human_detections)]
            predicts_photo = crf.run_LBP(cls_scores_photo)
            predicts_1.extend(predicts_photo)
            cur += len(photo.human_detections)
        predicts_1 = [lbl_map_test_to_global[predict] for predict in predicts_1]
        predicts_1 = np.array(predicts_1)

        # compute accuracy
        accuracy = evaluate_accuracy_with_predicts(test_detections_0, test_detections_1, predicts_0, predicts_1)

    else:
        accuracy = evaluate_accuracy_with_inst_similarity(test_detections_0, test_detections_1, similarity)

    return accuracy, module_accuracy


def evaluate_accuracy_with_predicts(test_0, test_1, predicts_0, predicts_1):
    identity_set_0 = get_identity_set(test_0)
    identity_set_1 = get_identity_set(test_1)

    total_count = len(test_0) + len(test_1)
    correct = 0

    for i in range(len(test_0)):
        detection = test_0[i]
        gt_label = detection.identity_id
        predict = predicts_0[i]
        if gt_label not in identity_set_1:
            total_count -= 1
        if predict == gt_label:
            correct += 1

    for i in range(len(test_1)):
        detection = test_1[i]
        gt_label = detection.identity_id
        predict = predicts_1[i]
        if gt_label not in identity_set_0:
            total_count -= 1
        if predict == gt_label:
            correct += 1

    accuracy = float(correct)/total_count
    return accuracy


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


def convert_inst_scores_to_cls_scores(similarity, testset0, testset1, num_identity, lbl_map):
    cls_scores1 = []
    for i in range(len(similarity)):
        cls_score = np.zeros(num_identity)
        for j in range(len(similarity[i])):
            id = lbl_map[testset1[j].identity_id]
            cls_score[id] = max(cls_score[id], similarity[i][j])
        cls_scores1.append(cls_score)
    cls_scores1 = np.array(cls_scores1)
    sklearn.preprocessing.normalize(cls_scores1)

    similarity = np.transpose(similarity)
    cls_scores2 = []
    for i in range(len(similarity)):
        cls_score = np.zeros(num_identity)
        for j in range(len(similarity[i])):
            id = lbl_map[testset0[j].identity_id]
            cls_score[id] = max(cls_score[id], similarity[i][j])
        cls_scores2.append(cls_score)
    cls_scores2 = np.array(cls_scores2)
    sklearn.preprocessing.normalize(cls_scores2)

    return cls_scores1, cls_scores2


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