import random

random.seed(0)

# feature configuration
features = \
{
    'face': {'path': 'feat/face.feat', 'weight': 0.5, 'similarity_metric': 'euclidean'},
    'body': {'path': 'feat/body.feat', 'weight': 1.0, 'similarity_metric': 'cosine'}
}

# K in KNN in prediction
K = 1

# normalization parameter for calculating similarity
beta0 = 0.0
beta1 = 1.0
