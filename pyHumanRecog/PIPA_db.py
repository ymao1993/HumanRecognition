""" PIPA data I/O
"""

import os

import numpy as np
from PIL import Image
SUBSET_LEFT = 0
SUBSET_TRAIN = 1
SUBSET_VAL = 2
SUBSET_TEST = 3


class HumanDetection:
    def __init__(self, head_bbox, identity_id, photo):
        self.head_bbox = [int(item) for item in head_bbox]
        self.identity_id = identity_id
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


class Photo:
    def __init__(self, album_id, photo_id, subset_id, file_path):
        self.album_id = album_id
        self.photo_id = photo_id
        self.subset_id = subset_id
        self.human_detections = []
        self.file_path = file_path
        with Image.open(file_path) as img:
            self.width, self.height = img.size

    def add_human_detection(self, bbox, identity_id):
        self.human_detections.append(HumanDetection(bbox, identity_id, self))


class Manager:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.photos = None
        self.photo_id_to_idx = {}
        self.split_indices = [[], [], []]
        self.num_labels = 0
        annotation_file = os.path.join(data_folder, 'annotations/index.txt')
        self._parse_annoatations(annotation_file)

    def get_photos(self):
        return self.photos

    def get_training_photos(self):
        return self.photos[self.split_indices[0]]

    def get_validation_photo(self):
        return self.photos[self.split_indices[1]]

    def get_testing_photos(self):
        return self.photos[self.split_indices[2]]

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
                identity_str_to_idx[identity_str] = len(identity_idx_to_str)
                identity_idx_to_str.append(identity_str)
            photo.add_human_detection(bbox, identity_str_to_idx[identity_str])
        self.photos = np.array(photos)
        self.num_labels = len(identity_idx_to_str)


if __name__ == '__main__':
    # ---- NOTE ----
    # this function is only used for demonstrating
    # how to use this module. Do not append your
    # code here. Thanks!
    # @ Yu
    manager = Manager('PIPA')
    training_photos = manager.get_training_photos()
    validation_photos = manager.get_validation_photo()
    testing_photos = manager.get_testing_photos()
    print(len(training_photos))
    print(len(validation_photos))
    print(len(testing_photos))

