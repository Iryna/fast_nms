# this is a re-implementation of the http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

import numpy as np


class FNMSuppression(object):
    def __init__(self, bbs, overlapThreshold = 0.45):
        self.bbs = bbs
        self.overlapThreshold = overlapThreshold

    def check_empty(self):
        # return an empty list, if there are no boxes
        if len(self.bbs) == 0:
            return []
        else:
            return self.bbs

    def check_dtype(self):
        # if the bounding boxes integers, convert them to floats (divisions)
        if self.bbs.dtype.kind == "i":
            self.bbs = self.bbs.astype("float")
        return self.bbs

    def bb_coordinates(self):
        # get the coordinates of the bounding boxes
        x1 = self.bbs[:, 0]
        y1 = self.bbs[:, 1]
        x2 = self.bbs[:, 2]
        y2 = self.bbs[:, 3]
        return x1, y1, x2, y2

    def bb_area(self):
        # compute the area of the bounding boxes
        x1, y1, x2, y2 = self.bb_coordinates()
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        return area

    def calc_ovarlap(self, x1, y1, x2, y2, idxs, last, i, area):
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        return overlap

    def suppress(self):
        self.check_empty()
        self.check_dtype()

        # initialize the list of picked indexes
        picked = []

        x1, y1, x2, y2 = self.bb_coordinates()

        # compute the area of the bounding boxes
        area = self.bb_area()

        # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # take the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            picked.append(i)

            overlap = self.calc_ovarlap(x1, y1, x2, y2, idxs, last, i, area)

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.overlapThreshold)[0])))

            # return only the bounding boxes that were picked using the
            # integer data type

        return bbs[picked].astype("int")
