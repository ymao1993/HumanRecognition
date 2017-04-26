""" PIPA data I/O
"""

import os
import numpy as np
import random
import sklearn
import sklearn.metrics
import config
from PIPA_db import HumanDetection
from PIPA_db import Photo
from PIPA_db import Manager

def test_pipeline(manager):
    test_detections = manager.get_testing_detections()
    
    #split test data
    (test_detections_0, test_detections_1) = split_test_data(test_detections)
    norm_similarity_face = calculate_feature_similarity(test_detections_0, test_detections_1, 'face_feature')
    norm_similarity_body = calculate_feature_similarity(test_detections_0, test_detections_1, 'body_feature')
    norm_similarity_cpm = calculate_feature_similarity(test_detections_0, test_detections_1, 'cpm_feature')

    (test_result_0to1, test_result_1to0) = simlilarity_fusion(norm_similarity_face, norm_similarity_body, norm_similarity_cpm)
    return (test_result_0to1, test_result_1to0)

def split_test_data(detections):
    random.shuffle(detections)
    test_detections_0 = detections[0:len(detections)/2]
    test_detections_1 = detections[len(detections)/2:len(detections)]
    return (test_detections_0, test_detections_1)

def calculate_feature_similarity(detections_0, detections_1, feature_name):
#def calculate_fature_similarity(features_0, features_1): 
    features_0 = get_features(detections_0, feature_name)    
    features_1 = get_features(detections_1, feature_name)
    coco_similarity = sklearn.metrics.pairwise.cosine_similarity(features_0, features_1)
    norm_similarity = 1.0/(1+np.exp(-(config.beta0 + config.beta1*coco_similarity)))

    return norm_similarity  

def get_features(detections, feature_name):    
    features = []
    for detection in detections:
        features.extend[detection.features[feature_name]]

    return features    
 
def simlilarity_fusion(norm_similarity_0, norm_similarity_1, norm_similarity_2):
    mean_similarity = config.w0*norm_similarity_0 + config.w1*norm_similarity_1 + config.w2*norm_similarity_2
    test_result_0to1 = np.argmax(mean_similarity,  axis=1)
    test_result_1to0 = np.argmax(mean_similarity,  axis=0)
    #print ("mean_similarity", mean_similarity)
    #print ("test_result_0to1", test_result_0to1)
    #print ("test_result_1to0", test_result_1to0)    

    return (test_result_0to1, test_result_1to0)

if __name__ == '__main__':
    manager = Manager('PIPA')
    (test_result_0to1, test_result_1to0) = test_pipeline(manager)

    #test coco pipeline
    #test_feature_0 = np.random.rand(5,10)
    #target_feature_0 = np.random.rand(3,10)

    #test_feature_1 = np.random.rand(5,20)
    #target_feature_1 = np.random.rand(3,20)

    #test_feature_2 = np.random.rand(5,30)
    #target_feature_2 = np.random.rand(3,30)

    #norm_similarity_0  = calculate_fature_similarity(test_feature_0, target_feature_0)
    #norm_similarity_1  = calculate_fature_similarity(test_feature_1, target_feature_1)
    #norm_similarity_2  = calculate_fature_similarity(test_feature_2, target_feature_2)     

    #(test_result_0to1, test_result_1to0) = simlilarity_fusion(norm_similarity_0, norm_similarity_1, norm_similarity_2)