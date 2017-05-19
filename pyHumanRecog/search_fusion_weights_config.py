import numpy as np
import random

random.seed(0)

# feature configuration
features = \
{
    'face': {'path': 'feat/face.feat', 'length': 128, 'weight_search_range': [0.5]},
    'body': {'path': 'feat/body2.feat', 'length': 1024, 'weight_search_range': np.arange(0.21, 0.4, 0.02)},
    'upper-body': {'path': 'feat/upperbody.feat', 'length': 1024, 'weight_search_range': np.arange(0.51, 0.7, 0.02)},
    'head': {'path': 'feat/head.feat', 'length': 1024, 'weight_search_range': np.arange(0.51, 0.7, 0.02)}
}


# normalization parameter for calculating similarity
beta0 = 0.0
beta1 = 1.0
