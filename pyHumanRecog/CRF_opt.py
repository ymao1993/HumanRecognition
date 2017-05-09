""" CRF_opt
This module contains code for post-processing the
prediction result by incorporating photo-level
context using CRF via loopy belief propagation(LBP).
@Yu
"""
import numpy as np
import CRF_opt_config as config


class CRFOptimizer:

    def __init__(self):
        self._compat_mat = None

    def build_compat_func(self, photos, lbl_map):
        """ construct compatibility function for all identities within photos
        :return:
        """
        mat = {}
        for photo in photos:
            for i in range(len(photo.human_detections)):
                for j in range(i + 1, len(photo.human_detections)):
                    id1 = lbl_map[photo.human_detections[i].identity_id]
                    id2 = lbl_map[photo.human_detections[j].identity_id]
                    if id2 < id1:
                        id1, id2 = id2, id1
                    if id1 not in mat:
                        mat[id1] = {}
                    mat[id1][id2] = config.compat_label_co_occur_val
        self._compat_mat = mat
        return

    def get_cooc_prob(self, id1, id2):
        if id1 == id2:
            return config.compat_label_equal_val
        if id2 < id1:
            id1, id2 = id2, id1
        return self._compat_mat[id1].get(id2, config.compat_label_not_co_occur_val)

    def run_LBP(self, scores):
        """
        :param scores: class prediction scores of shape (N,M) where N is the number of instances and M is the number of identities
        :return: refined labels after LBP (which has the same shape as predicted_lbls)
        """
        predicts = np.argmax(scores, axis=1)
        return predicts
