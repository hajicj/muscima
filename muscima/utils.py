"""This module implements a class that..."""
from __future__ import print_function, unicode_literals
from skimage.measure import label

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


def connected_components2bboxes(labels):
    """Returns a dictionary of bounding boxes (upper left c., lower right c.)
    for each label.

    >>> labels = [[0, 0, 1, 1], [2, 0, 0, 1], [2, 0, 0, 0], [0, 0, 3, 3]]
    >>> bboxes = connected_components2bboxes(labels)
    >>> bboxes[0]
    [0, 0, 4, 4]
    >>> bboxes[1]
    [0, 2, 2, 4]
    >>> bboxes[2]
    [1, 0, 3, 1]
    >>> bboxes[3]
    [3, 2, 4, 4]


    :param labels: The output of cv2.connectedComponents().

    :returns: A dict indexed by labels. The values are quadruplets
        (xmin, ymin, xmax, ymax) so that the component with the given label
        lies exactly within labels[xmin:xmax, ymin:ymax].
    """
    bboxes = {}
    for x, row in enumerate(labels):
        for y, l in enumerate(row):
            if l not in bboxes:
                bboxes[l] = [x, y, x+1, y+1]
            else:
                box = bboxes[l]
                if x < box[0]:
                    box[0] = x
                elif x + 1 > box[2]:
                    box[2] = x + 1
                if y < box[1]:
                    box[1] = y
                elif y + 1 > box[3]:
                    box[3] = y + 1
    return bboxes


def compute_connected_components(image):
    labels = label(image, background=0)
    cc = int(labels.max())
    bboxes = connected_components2bboxes(labels)
    return cc, labels, bboxes

