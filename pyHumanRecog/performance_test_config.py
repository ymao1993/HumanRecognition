import random

random.seed(0)

# feature configuration
features = \
{
    'face': {'path': 'feat/face.feat', 'weight': 0.5, 'length': 128},
    'body_new': {'path': 'feat/body2.feat', 'weight': 0.29, 'length': 1024},
    'upper-body': {'path': 'feat/upperbody.feat', 'weight': 0.63, 'length': 1024},
    'head': {'path': 'feat/head.feat', 'weight': 0.65, 'length': 1024}
}

# normalization parameter for calculating similarity
beta0 = 0.0
beta1 = 1.0

# whether to refine the result with photo-level co-occurrence pattern via Loopy Belief Propagation(LBP)
refine_with_photo_level_context = False
