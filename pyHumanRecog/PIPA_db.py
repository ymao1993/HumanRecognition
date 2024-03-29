""" PIPA data I/O
"""

import os

import numpy as np
from PIL import Image
import cPickle as pickle
SUBSET_LEFT = 0
SUBSET_TRAIN = 1
SUBSET_VAL = 2
SUBSET_TEST = 3


class HumanDetection:
    def __init__(self, head_bbox, identity_id, photo):
        self.head_bbox = [int(item) for item in head_bbox]
        self.identity_id = identity_id
        self.features = {}
        self.is_face = None
        self.photo = photo

    def scale(self, h_scale, w_scale):
        self.head_bbox[0] = int(self.head_bbox[0] * w_scale)
        self.head_bbox[1] = int(self.head_bbox[1] * h_scale)
        self.head_bbox[2] = int(self.head_bbox[2] * w_scale)
        self.head_bbox[3] = int(self.head_bbox[3] * h_scale)

    def get_head_center(self):
        x = int(self.head_bbox[0] + self.head_bbox[2]//2)
        y = int(self.head_bbox[1] + self.head_bbox[3]//2)
        return y, x

    def get_clipped_bbox(self):
        bbox = self.head_bbox
        bbox[0] = np.clip(bbox[0], 0, self.photo.width)
        bbox[1] = np.clip(bbox[1], 0, self.photo.height)
        bbox[2] = np.clip(bbox[2], 0, self.photo.width - bbox[0])
        bbox[3] = np.clip(bbox[3], 0, self.photo.height - bbox[1])
        return bbox

    def get_estimated_human_center(self):
        """ estimating human center (used by CPM)
        """
        w = self.head_bbox[2]
        h = self.head_bbox[3]
        l = min(w, h)
        hy, hx = self.get_head_center()
        x = hx
        y = hy + 2 * l
        return y, x

    def get_estimated_body_bbox(self):
        """ estimating body bounding box
        :return: (x, y, w, h)
        """
        w = self.head_bbox[2]
        hx = int(self.head_bbox[0] + self.head_bbox[2] // 2)
        x = hx - w
        y = self.head_bbox[1]
        w *= 2
        h = self.head_bbox[3] * 6

        # clip the body bbox within the image
        x = np.clip(x, 0, self.photo.width)
        y = np.clip(y, 0, self.photo.height)
        w = np.clip(w, 0, self.photo.width - x)
        h = np.clip(h, 0, self.photo.height - y)

        return x, y, w, h

    def get_estimated_upper_body_bbox(self):
        """ estimating body bounding box
        :return: (x, y, w, h)
        """
        w = self.head_bbox[2]
        hx = int(self.head_bbox[0] + self.head_bbox[2] // 2)
        x = hx - w
        y = self.head_bbox[1]
        w *= 2
        h = self.head_bbox[3] * 3

        # clip the body bbox within the image
        x = np.clip(x, 0, self.photo.width)
        y = np.clip(y, 0, self.photo.height)
        w = np.clip(w, 0, self.photo.width - x)
        h = np.clip(h, 0, self.photo.height - y)

        return x, y, w, h


class Photo:
    def __init__(self, album_id, photo_id, subset_id, file_path):
        self.album_id = album_id
        self.photo_id = photo_id
        self.subset_id = subset_id
        self.human_detections = []
        self.file_path = file_path
        img = Image.open(file_path)
        self.width, self.height = img.size

    def add_human_detection(self, bbox, identity_id):
        self.human_detections.append(HumanDetection(bbox, identity_id, self))


class Manager:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.photos = None
        self.photo_id_to_idx = {}
        self.split_indices = [[], [], []]
        self.label_map_subset_to_global = [[], [], []]
        self.label_map_global_to_subset = [{}, {}, {}]
        self.num_labels = 0
        self.num_labels_train = None
        self.num_labels_val = None
        self.num_labels_test = None
        annotation_file = os.path.join(data_folder, 'annotations/index.txt')
        self._parse_annoatations(annotation_file)

    def get_photos(self):
        return self.photos

    def get_training_photos(self):
        return self.photos[self.split_indices[0]]

    def get_validation_photos(self):
        return self.photos[self.split_indices[1]]

    def get_testing_photos(self):
        return self.photos[self.split_indices[2]]

    def get_training_detections(self):
        return Manager.get_detections_from_photos(self.get_training_photos())

    def get_validation_detections(self):
        return Manager.get_detections_from_photos(self.get_validation_photos())

    def get_testing_detections(self):
        return Manager.get_detections_from_photos(self.get_testing_photos())

    @staticmethod
    def get_detections_from_photos(photos):
        detections = []
        for photo in photos:
            detections.extend(photo.human_detections)
        return detections

    def get_photo_path(self, subset_id, album_id, photo_id):
        if subset_id == SUBSET_LEFT:
            subset_folder = 'leftover'
        elif subset_id == SUBSET_TRAIN:
            subset_folder = 'train'
        elif subset_id == SUBSET_VAL:
            subset_folder = 'val'
        elif subset_id == SUBSET_TEST:
            subset_folder = 'test'
        else:
            raise Exception('invalid subset id')
        return os.path.join(self.data_folder, subset_folder, album_id + '_' + photo_id + '.jpg')

    def get_num_labels(self):
        return self.num_labels

    def get_num_labels_training(self):
        return len(self.label_map_subset_to_global[0])

    def get_num_labels_validation(self):
        return len(self.label_map_subset_to_global[1])

    def get_num_labels_testing(self):
        return len(self.label_map_subset_to_global[2])

    def get_label_mapping_global_to_train(self):
        return self.label_map_global_to_subset[0]

    def get_label_mapping_train_to_global(self):
        return self.label_map_subset_to_global[0]

    def get_label_mapping_global_to_val(self):
        return self.label_map_global_to_subset[1]

    def get_label_mapping_val_to_global(self):
        return self.label_map_subset_to_global[1]

    def get_label_mapping_global_to_test(self):
        return self.label_map_global_to_subset[2]

    def get_label_mapping_test_to_global(self):
        return self.label_map_subset_to_global[2]

    def load_features(self, feature_name, feature_file, subset='test'):
        assert subset == 'train' or subset == 'val' or subset == 'test', 'invalid subset name'
        fd = open(feature_file, 'rb')
        features = pickle.load(fd)

        if subset == 'train':
            detections = self.get_training_detections()
        elif subset == 'val':
            detections = self.get_validation_detections()
        else:
            detections = self.get_testing_detections()
        assert(len(features) == len(detections))

        for detection, feature in zip(detections, features):
            detection.features[feature_name] = feature

    def load_head_annotation(self, head_annotation_folder):
        detections = self.get_training_detections()
        head_annotation_file = os.path.join(head_annotation_folder, 'train_head_annotate.txt')
        assert(os.path.exists(head_annotation_file))
        head_annotations = open(head_annotation_file).read().strip().split()
        head_annotations = [bool(int(annotation)) for annotation in head_annotations]
        assert(len(detections) == len(head_annotations))
        for detection, annotation in zip(detections, head_annotations):
            detection.is_face = annotation

        detections = self.get_validation_detections()
        head_annotation_file = os.path.join(head_annotation_folder, 'valid_head_annotate.txt')
        assert(os.path.exists(head_annotation_file))
        head_annotations = open(head_annotation_file).read().strip().split()
        head_annotations = [bool(int(annotation)) for annotation in head_annotations]
        assert(len(detections) == len(head_annotations))
        for detection, annotation in zip(detections, head_annotations):
            detection.is_face = annotation

        detections = self.get_testing_detections()
        head_annotation_file = os.path.join(head_annotation_folder, 'test_head_annotate.txt')
        assert(os.path.exists(head_annotation_file))
        head_annotations = open(head_annotation_file).read().strip().split()
        head_annotations = [bool(int(annotation)) for annotation in head_annotations]
        assert(len(detections) == len(head_annotations))
        for detection, annotation in zip(detections, head_annotations):
            detection.is_face = annotation

    def _parse_annoatations(self, annotation_file):
        if not os.path.exists(annotation_file):
            raise Exception('annotation file {0} does not exist'.format(annotation_file))
        photos = []
        file = open(annotation_file)
        identity_str_to_idx = {}
        identity_idx_to_str = []
        for line in file:
            fields = line.strip().split()
            assert(len(fields) == 8)
            album_id = fields[0]
            photo_id = fields[1]
            subset_id = int(fields[7])
            if subset_id == SUBSET_LEFT:    # ignore leftover data
                continue
            if photo_id in self.photo_id_to_idx:
                photo = photos[self.photo_id_to_idx[photo_id]]
            else:
                file_path = self.get_photo_path(subset_id, album_id, photo_id)
                assert os.path.exists(file_path)
                photo = Photo(album_id, photo_id, subset_id, file_path)
                photos.append(photo)
                idx = len(photos) - 1
                self.photo_id_to_idx[photo_id] = idx
                self.split_indices[subset_id - 1].append(idx)
            xmin = float(fields[2])
            ymin = float(fields[3])
            width = float(fields[4])
            height = float(fields[5])
            bbox = [xmin, ymin, width, height]
            identity_str = fields[6]
            if identity_str not in identity_str_to_idx:
                global_idx = len(identity_idx_to_str)
                identity_str_to_idx[identity_str] = global_idx
                identity_idx_to_str.append(identity_str)

                subset_idx = len(self.label_map_subset_to_global[subset_id - 1])
                self.label_map_global_to_subset[subset_id - 1][global_idx] = subset_idx
                self.label_map_subset_to_global[subset_id - 1].append(global_idx)
            photo.add_human_detection(bbox, identity_str_to_idx[identity_str])

        self.photos = np.array(photos)
        self.num_labels = len(identity_idx_to_str)


if __name__ == '__main__':
    # ---- NOTE ----
    # this function is only used for demonstrating
    # how to use this module. Do not append your
    # code here. Thanks!
    # @ Yu

    # basic usage
    manager = Manager('PIPA')
    training_photos = manager.get_training_photos()
    validation_photos = manager.get_validation_photos()
    testing_photos = manager.get_testing_photos()
    print(len(training_photos))
    print(len(validation_photos))
    print(len(testing_photos))

    # feature loading
    manager.load_features(feature_name='body_feature',
                          feature_file='feat/body.feat',
                          subset='test')

    # head annotation loading
    manager.load_head_annotation('head_annotation')

    # statistics on head annotation
    training_detections = manager.get_training_detections()
    validation_detections = manager.get_validation_detections()
    testing_detections = manager.get_testing_detections()
    
    count = 0
    for i in range(len(training_detections)):
        if training_detections[i].is_face:
            count += 1
    print('training set: {0}/{1} faces'.format(count, len(training_detections)))

    count = 0
    for i in range(len(validation_detections)):
        if validation_detections[i].is_face:
            count += 1
    print('validation set: {0}/{1} faces'.format(count, len(validation_detections)))

    count = 0
    for i in range(len(testing_detections)):
        if testing_detections[i].is_face:
            count += 1
    print('testing set: {0}/{1} faces'.format(count, len(testing_detections)))

    for photo in manager.get_testing_photos():
        if len(photo.human_detections) == 0:
            print('no human detection in {0}'.format(photo.file_path))

