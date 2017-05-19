""" CRF_opt
This module contains code for post-processing the
prediction result by incorporating photo-level
context using CRF via loopy belief propagation(LBP).
@Yu
"""
import sklearn.preprocessing
import numpy as np
import CRF_opt_config as config


class CRFOptimizer:

    def __init__(self):
        self._compat_mat = None

    def build_compat_func(self, photos, lbl_map):
        """
        construct compatibility function for all identities within photos
        :return:
        """
        num_identity = len(lbl_map)
        # print(num_identity)
        compat_mat = np.empty((num_identity, num_identity))
        compat_mat.fill(config.compat_label_not_co_occur_val)
        for i in range(num_identity):
            compat_mat[i][i] = config.compat_label_equal_val

        for photo in photos:
            for i in range(len(photo.human_detections)):
                for j in range(i + 1, len(photo.human_detections)):
                    id1 = lbl_map[photo.human_detections[i].identity_id]
                    id2 = lbl_map[photo.human_detections[j].identity_id]
                    assert(id1 != id2)
                    compat_mat[id1][id2] = config.compat_label_co_occur_val
                    compat_mat[id2][id1] = config.compat_label_co_occur_val
        self._compat_mat = compat_mat
        return

    def run_LBP(self, scores):
        """
        Running Loopy Belief Propagation(LBP)
        :param scores: class prediction scores of shape (N,M) where N is the number of instances and M is the number of identities
        :return: refined labels after LBP, shape is (N,)
        """

        num_hidden_nodes = len(scores)
        num_labels = len(scores[0])

        # print('num_instances: {0}'.format(num_hidden_nodes))

        # message initialization
        messages_on_fly = np.empty((num_hidden_nodes, num_hidden_nodes, num_labels))
        messages_on_fly.fill(0.00001)
        for i in range(num_hidden_nodes):
            messages_on_fly[i, i, :] = 0

        new_message_on_fly = np.zeros((num_hidden_nodes, num_hidden_nodes, num_labels))

        # message passing
        for iter in range(config.num_iteration):
            # print('({1})iteration {0}...'.format(iter, num_hidden_nodes))
            for i in range(num_hidden_nodes):
                # at each iteration, every node will send messages to all of its neighboring nodes
                for j in range(num_hidden_nodes):
                    if i == j:
                        continue
                    neighbor_contribs = messages_on_fly[:, i].sum(axis=0)
                    msg = np.empty(num_labels)
                    for kk in range(num_labels):    # over recipient's label
                        for k in range(num_labels):  # over sender's label
                           msg[kk] = max(msg[kk], scores[i, k] + self._compat_mat[kk, k] + neighbor_contribs[k])
                    msg = sklearn.preprocessing.normalize(msg.reshape((1, -1)))
                    new_message_on_fly[i, j] = msg
            messages_on_fly[:, :, :] = new_message_on_fly

        # get final belief
        belief = np.zeros((num_hidden_nodes, num_labels))
        for i in range(num_hidden_nodes):
            for j in range(num_labels):
                belief[i, j] = scores[i, j] + messages_on_fly[:, i, j].sum()
        predicts = np.argmax(belief, axis=1)
        predicts_before_refine = np.argmax(scores, axis=1)

        return predicts, predicts_before_refine
