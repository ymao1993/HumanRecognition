"""
This module implements Congenerous Cosine(COCO) Loss function
@Yu
"""

import tensorflow as tf


def coco_loss_layer(features, labels, batch_size):
    """
    compute coco loss
    :param features: Tensor of shape (batch_size, feature_length)
    :param labels: String Tensor of shape (batch_size,)
    :param batch_size: batch size
    :return:
    """
    # first compute the centroid of each class
    centroids = []
    for i in range(batch_size):
        feat_indices = tf.reshape(tf.where(tf.equal(labels, i)), shape=(-1, 1))
        centroid = tf.squeeze(tf.reduce_mean(tf.gather(features, feat_indices), axis=0))
        centroids.append(centroid)
    centroids = tf.stack(centroids)
    centroids = tf.where(tf.is_nan(centroids), tf.zeros_like(centroids), centroids)
    centroids = tf.nn.l2_normalize(centroids, dim=1)

    # compute a binary mask to indicate the existence of labels in this mini-batch
    label_mask = tf.sparse_to_dense(sparse_indices=tf.expand_dims(tf.unique(labels)[0], axis=1),
                                    output_shape=(batch_size,),
                                    sparse_values=1.)

    # compute loss
    features = tf.nn.l2_normalize(features, dim=1)
    features_lst = tf.unstack(features)
    log_probs = []
    for i in range(len(features_lst)):
        centroid = centroids[labels[i]]
        tmp1 = tf.reduce_sum(tf.multiply(centroid, features_lst[i]))
        tmp2 = tf.log(tf.reduce_sum(tf.multiply(tf.exp(tf.reduce_sum(tf.multiply(centroids, features_lst[i]), axis=1)), label_mask)))
        log_probs.append(tmp2 - tmp1)
    log_probs = tf.stack(log_probs)
    loss = tf.reduce_sum(log_probs)
    return loss


def _coco_loss_ref(features, labels, batch_size):
    import numpy as np
    from sklearn.preprocessing import normalize
    """
    This function is implemented to test coco_loss
    """
    # label mask
    label_mask = np.zeros(batch_size)
    for i in range(len(labels)):
        label_mask[labels[i]] = 1.

    # compute centroid for each classes
    centroids = []
    for i in range(batch_size):
        centroid = np.zeros_like(features[0])
        count = 0
        for j in range(len(features)):
            if labels[j] == i:
                count += 1
                centroid += features[j]
        if count !=0:
            centroid /= count
        centroids.append(centroid)
    centroids = np.array(centroids)
    centroids = normalize(centroids, axis=1)

    features = normalize(features, axis=1)
    # compute loss for each sample
    loss = []
    for i in range(len(features)):
        tmp1 = np.sum(centroids[labels[i]] * features[i])
        tmp2 = np.log(np.sum(np.exp(np.sum(centroids * features[i], axis=1)) * label_mask))
        loss.append(tmp2 - tmp1)
    loss = np.sum(np.array(loss))
    return loss


if __name__ == '__main__':
    # ---- NOTE ----
    # this function is only used for demonstrating
    # how to use this module. Do not append your
    # code here. Thanks!
    # @ Yu
    import numpy as np
    feature_length = 3
    features = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.], [13., 14., 15.], [16., 17., 18.]])
    labels = np.array([0, 0, 0, 0, 0, 1])
    batch_size = 6

    tf_features = tf.placeholder(dtype=tf.float32, shape=(batch_size, feature_length), name='features')
    tf_labels = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='labels')
    tf_output = coco_loss_layer(tf_features, tf_labels, batch_size)

    sess = tf.Session()
    print('start testing')
    print('running reference implementation....')
    ref_output = _coco_loss_ref(features, labels, batch_size)
    print('running Yu\'s implementation...')
    output = sess.run(tf_output, feed_dict={tf_features: features, tf_labels: labels})
    print('ref output: ' + str(ref_output))
    print('output: ' + str(output))

