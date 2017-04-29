""" PIPA data I/O
"""

import os
import numpy as np
import random
import sklearn
import sklearn.metrics
import config
import cPickle as pickle
from PIPA_db import HumanDetection
from PIPA_db import Photo
from PIPA_db import Manager

def test_pipeline(manager):
    test_detections = manager.get_testing_detections()
    
    #split test data
    (test_detections_0, test_detections_1) = split_test_data(test_detections)
    (norm_similarity_face, result_0to1_face, result_1to0_face) = cal_feature_similarity(test_detections_0, test_detections_1, 'face_feature')
    (norm_similarity_body, result_0to1_body, result_1to0_body) = cal_feature_similarity(test_detections_0, test_detections_1, 'body_feature')
    (norm_similarity_cpm, result_0to1_cpm, result_1to0_cpm) = cal_feature_similarity(test_detections_0, test_detections_1, 'cpm_feature')

    (test_result_0to1, test_result_1to0) = simlilarity_fusion(norm_similarity_face, norm_similarity_body, norm_similarity_cpm)
    return (result_0to1_face, result_1to0_face, \
            result_0to1_body, result_1to0_body, \
            result_0to1_cpm, result_1to0_cpm, \
            test_result_0to1, test_result_1to0)

def split_test_data(detections):
    random.shuffle(detections)
    test_detections_0 = detections[0:len(detections)/2]
    test_detections_1 = detections[len(detections)/2:len(detections)]
    return (test_detections_0, test_detections_1)

def cal_feature_similarity(detections_0, detections_1, feature_name):
    features_0 = get_features(detections_0, feature_name)    
    features_1 = get_features(detections_1, feature_name)
    coco_similarity = sklearn.metrics.pairwise.cosine_similarity(features_0, features_1)
    norm_similarity = 1.0/(1+np.exp(-(config.beta0 + config.beta1*coco_similarity)))
    test_result_0to1 = np.argmax(norm_similarity,  axis=1)
    test_result_1to0 = np.argmax(norm_similarity,  axis=0)

    return (norm_similarity, test_result_0to1, test_result_1to0)

def get_features(detections, feature_name):    
    features = []
    for detection in detections:    
        features.extend([detection.features[feature_name]])

    features = np.array(features)  
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
    
    if not os.path.exists('temp'):
        os.mkdir('temp')    

    #test coco pipeline
    manager = Manager('PIPA')
    testing_photos = manager.get_testing_photos()
    test_detections = manager.get_testing_detections()

    #generate random feature
    face_feature = np.random.rand(len(test_detections),10)
    body_feature = np.random.rand(len(test_detections),20)
    cpm_feature = np.random.rand(len(test_detections),30)
    
    #write to .feat file
    pickle.dump( face_feature, open( 'temp/face.feat', 'wb' ))
    pickle.dump( body_feature, open( 'temp/body.feat', 'wb' ))
    pickle.dump( cpm_feature, open( 'temp/cpm.feat', 'wb' ))

    manager.load_features(feature_name = 'face_feature', 
                          feature_file = 'temp/face.feat', 
                          subset = 'test')

    manager.load_features(feature_name = 'body_feature', 
                          feature_file = 'temp/body.feat', 
                          subset = 'test')

    manager.load_features(feature_name = 'cpm_feature', 
                          feature_file = 'temp/cpm.feat', 
                          subset = 'test')
    

    (result_0to1_face, result_1to0_face, \
     result_0to1_body, result_1to0_body, \
     result_0to1_cpm, result_1to0_cpm, \
     test_result_0to1, test_result_1to0) = test_pipeline(manager)

    print ("test_result_0to1", test_result_0to1)
    print ("test_result_1to0", test_result_1to0)    