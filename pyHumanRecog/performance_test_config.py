import random

random.seed(0)

# feature configuration
features = \
{
    'face': {'path': 'feat/face.feat', 'weight': 0.5},
    'body': {'path': 'feat/body.feat', 'weight': 1.0}
}

# normalization parameter for calculating similarity
beta0 = 0.0
beta1 = 1.0

# whether to refine the result with photo-level co-occurrence pattern via Loopy Belief Propagation(LBP)
refine_with_photo_level_context = True

