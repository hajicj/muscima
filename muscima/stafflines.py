"""This module implements functions for manipulating staffline symbols."""
from __future__ import print_function, unicode_literals

import logging

import numpy

from skimage.measure import label
from skimage.morphology import watershed
from skimage.filters import gaussian

from muscima.cropobject import cropobjects_merge_bbox, CropObject, cropobjects_on_canvas
from muscima.inference_engine_constants import InferenceEngineConstants as _CONST

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


##############################################################################


def merge_staffline_segments(cropobjects, margin=10):
    """Given a list of CropObjects that contain some staffline
    objects, generates a new list where the stafflines
    are merged based on their horizontal projections.
    Basic step for going from the staffline detection masks to
    the actual staffline objects.

    Assumes that stafflines are straight: their bounding boxes
    do not touch or overlap.

    :param cropobjects:
    :param margin:

    :return: A modified CropObject list: the original staffline-class
        symbols are replaced by the merged ones. If the original stafflines
        had any inlinks, they are preserved (mapped to the new staffline).
    """
    # margin is used to avoid the stafflines touching the edges,
    # which could perhaps break some assumptions down the line.
    old_staffline_cropobjects = [c for c in cropobjects
                             if c.clsname == _CONST.STAFFLINE_CLSNAME]
    canvas, (_t, _l) = cropobjects_on_canvas(old_staffline_cropobjects)

    _staffline_bboxes, staffline_masks = staffline_bboxes_and_masks_from_horizontal_merge(canvas)
    # Bounding boxes need to be adjusted back with respect to the original image!
    staffline_bboxes = [(t + _t, l + _l, b + _t, r + _l) for t, l, b, r in _staffline_bboxes]

    # Create the CropObjects.
    next_objid = max([c.objid for c in cropobjects]) + 1
    dataset_namespace = cropobjects[0].dataset
    docname = cropobjects[0].doc

    #  - Create the staffline CropObjects
    staffline_cropobjects = []
    for sl_bb, sl_m in zip(staffline_bboxes, staffline_masks):
        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        t, l, b, r = sl_bb
        c = CropObject(objid=next_objid,
                       clsname=_CONST.STAFFLINE_CLSNAME,
                       top=t, left=l, height=b - t, width=r - l,
                       mask=sl_m,
                       uid=uid)
        staffline_cropobjects.append(c)
        next_objid += 1

    non_staffline_cropobjects = [c for c in cropobjects
                                 if c.clsname != _CONST.STAFFLINE_CLSNAME]
    old_staffline_objids = set([c.objid for c in old_staffline_cropobjects])
    old2new_staffline_objid_map = {}
    for os in old_staffline_cropobjects:
        for ns in staffline_cropobjects:
            if os.overlaps(ns):
                old2new_staffline_objid_map[os.objid] = ns

    logging.info('Re-linking from the old staffline objects to new ones.')
    for c in non_staffline_cropobjects:
        new_outlinks = []
        for o in c.outlinks:
            if o in old_staffline_objids:
                new_staffline = old2new_staffline_objid_map[o]
                new_outlinks.append(new_staffline.objid)
                new_staffline.inlinks.append(c.objid)
            else:
                new_outlinks.append(o)

    output = non_staffline_cropobjects + staffline_cropobjects
    return output


def staffline_bboxes_and_masks_from_horizontal_merge(mask):
    """Returns a list of staff_line bboxes and masks
     computed from the input mask, with
    each set of connected components in the mask that has at least
    one pixel in a neighboring or overlapping row is assigned to
    the same label. Intended for finding staffline masks from individual
    components of the stafflines (for this purpose, you have to assume
    that the stafflines are straight)."""
    logging.info('Getting staffline connected components.')

    cc, labels, bboxes = compute_connected_components(mask)

    logging.info('Getting staffline component vertical projections')
    #  - Use vertical dimension of CCs to determine which ones belong together
    #    to form stafflines. (Criterion: row overlap.)
    n_rows, n_cols = mask.shape
    # For each row of the image: which CCs have pxs on that row?
    intervals = [[] for _ in range(n_rows)]
    for label, (t, l, b, r) in bboxes.items():
        if label == 0:
            continue
        # Ignore very short staffline segments that can easily be artifacts
        # and should not affect the vertical range of the staffline anyway.
        # The "8" is a magic number, no good reason for specifically 8.
        # (It should be some proportion of the image width, like 0.01.)
        if (r - l) < 8:
            continue
        for row in range(t, b):
            intervals[row].append(label)

    logging.warning('Grouping staffline connected components into stafflines.')
    # For each staffline, we collect the CCs that it is made of. We assume stafflines
    # are separated from each other by at least one empty row.
    staffline_components = []
    _in_staffline = False
    _current_staffline_components = []
    for r_labels in intervals:
        if not _in_staffline:
            # Last row did not contain staffline components.
            if len(r_labels) == 0:
                # No staffline component on current row
                continue
            else:
                _in_staffline = True
                _current_staffline_components += r_labels
        else:
            # Last row contained staffline components.
            if len(r_labels) == 0:
                # Current staffline has no more rows.
                staffline_components.append(set(_current_staffline_components))
                _current_staffline_components = []
                _in_staffline = False
                continue
            else:
                # Current row contains staffline components: the current
                # staffline continues.
                _current_staffline_components += r_labels

    logging.info('No. of stafflines, with component groups: {0}'
                 ''.format(len(staffline_components)))

    # Now: merge the staffline components into one bbox/mask.
    logging.info('Merging staffline components into staffline bboxes and masks.')
    staffline_bboxes = []
    staffline_masks = []
    for sc in sorted(staffline_components,
                     key=lambda c: min([bboxes[cc][0]
                                        for cc in c])):  # Sorted top-down
        st, sl, sb, sr = n_rows, n_cols, 0, 0
        for component in sc:
            t, l, b, r = bboxes[component]
            st, sl, sb, sr = min(t, st), min(l, sl), max(b, sb), max(r, sr)
        # Here, again, we assume that no two "real" stafflines have overlapping
        # bounding boxes.
        _sm = mask[st:sb, sl:sr]
        staffline_bboxes.append((st, sl, sb, sr))
        staffline_masks.append(_sm)

    # Check if n. of stafflines is divisible by 5
    n_stafflines = len(staffline_bboxes)
    logging.warning('\tTotal stafflines: {0}'.format(n_stafflines))
    if n_stafflines % 5 != 0:
        try:
            import matplotlib.pyplot as plt
            stafllines_mask_image = numpy.zeros(mask.shape)
            for i, (_sb, _sm) in enumerate(zip(staffline_bboxes, staffline_masks)):
                t, l, b, r = _sb
                stafllines_mask_image[t:b, l:r] = min(255, (i * 333) % 255 + 40)
            plt.imshow(stafllines_mask_image, cmap='jet', interpolation='nearest')
            plt.show()
        except ImportError:
            pass
        raise ValueError('No. of stafflines is not divisible by 5!')

    return staffline_bboxes, staffline_masks


def staff_bboxes_and_masks_from_staffline_bboxes_and_image(staffline_bboxes, mask):
    logging.warning('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    n_stafflines = len(staffline_bboxes)
    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5*i:5*(i+1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(mask[_st:_sb, _sl:_sr])

    logging.warning('Total staffs: {0}'.format(len(staff_bboxes)))

    return staff_bboxes, staff_masks


##############################################################################


def build_staff_cropobjects(cropobjects):
    """Derives staff objects from staffline objcets.

    Assumes each staff has 5 stafflines.

    Assumes the stafflines have already been merged."""
    stafflines = [c for c in cropobjects if c.clsname == _CONST.STAFFLINE_CLSNAME]
    staffline_bboxes = [c.bounding_box for c in stafflines]
    canvas, (_t, _l) = cropobjects_on_canvas(stafflines)

    logging.warning('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    n_stafflines = len(stafflines)
    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5*i:5*(i+1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(canvas[_st-_t:_sb-_t, _sl-_l:_sr-_l])

    logging.info('Creating staff CropObjects')
    next_objid = max([c.objid for c in cropobjects]) + 1
    dataset_namespace = cropobjects[0].dataset
    docname = cropobjects[0].doc

    staff_cropobjects = []
    for s_bb, s_m in zip(staff_bboxes, staff_masks):
        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        t, l, b, r = s_bb
        c = CropObject(objid=next_objid,
                       clsname=_CONST.STAFF_CLSNAME,
                       top=t, left=l, height=b - t, width=r - l,
                       mask=s_m,
                       uid=uid)
        staff_cropobjects.append(c)
        next_objid += 1

    for i, sc in enumerate(staff_cropobjects):
        sl_from = 5 * i
        sl_to = 5 * (i + 1)
        for sl in stafflines[sl_from:sl_to]:
            sl.inlinks.append(sc.objid)
            sc.outlinks.append(sl.objid)

    return staff_cropobjects