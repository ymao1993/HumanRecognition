""" PIPA data I/O
"""

import os
import numpy as np
import sklearn
import sklearn.metrics
import config

def test_pipeline(test_feature, target_feature):
    coco_similarity = sklearn.metrics.pairwise.cosine_similarity(test_feature, target_feature)
    test_result = np.argmax(coco_similarity,  axis=1)

    return (coco_similarity, test_result)

if __name__ == '__main__':
    #test coco pipeline
    test_feature = np.random.rand(50,100)
    target_feature = np.random.rand(200,100)

    (coco_similarity, test_result) = test_pipeline(test_feature, target_feature)
    #print ('test_result', test_result)