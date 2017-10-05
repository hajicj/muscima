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


def staffline_surroundings_mask(staffline_cropobject):
    """Find the parts of the staffline's bounding box which lie
    above or below the actual staffline.

    These areas will be very small for straight stafflines,
    but might be considerable when staffline curvature grows.
    """
    # We segment both masks into "above staffline" and "below staffline"
    # areas.
    elevation = staffline_cropobject.mask * 255
    # Blur, to plug small holes somewhat:
    elevation = gaussian(elevation, sigma=1.0)
    # Prepare the segmentation markers: 1 is ABOVE, 2 is BELOW
    markers = numpy.zeros(staffline_cropobject.mask.shape)
    markers[0, :] = 1
    markers[-1, :] = 2
    markers[staffline_cropobject.mask != 0] = 0
    seg = watershed(elevation, markers)

    bmask = numpy.ones(seg.shape)
    bmask[seg != 2] = 0
    tmask = numpy.ones(seg.shape)
    tmask[seg != 1] = 0

    return bmask, tmask

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


def build_staffspace_cropobjects(cropobjects):
    """Creates the staffspace objects based on stafflines
    and staffs. There is a staffspace between each two stafflines,
    one on the top side of each staff, and one on the bottom
    side for each staff (corresponding e.g. to positions of g5 and d4
    with the standard G-clef).

    Note that staffspaces do not assume anything about the number
    of stafflines per staff.

    :param cropobjects: A list of CropObjects that must contain
        all the relevant stafflines and staffs.

    :return: A list of staffspace CropObjects.
    """
    next_objid = max([c.objid for c in cropobjects]) + 1
    dataset_namespace = cropobjects[0].dataset
    docname = cropobjects[0].doc

    staff_cropobjects = [c for c in cropobjects
                         if c.clsname == _CONST.STAFF_CLSNAME]
    staffline_cropobjects = [c for c in cropobjects
                             if c.clsname == _CONST.STAFFLINE_CLSNAME]

    staffspace_cropobjects = []

    for i, staff in enumerate(staff_cropobjects):
        current_stafflines = [sc for sc in staffline_cropobjects
                              if sc.objid in staff.outlinks]
        sorted_stafflines = sorted(current_stafflines, key=lambda x: x.top)

        current_staffspace_cropobjects = []

        # Percussion single-line staves do not have staffspaces.
        if len(sorted_stafflines) == 1:
            continue

        #################
        # Internal staffspace
        for s1, s2 in zip(sorted_stafflines[:-1], sorted_stafflines[1:]):
            # s1 is the UPPER staffline, s2 is the LOWER staffline
            # Left and right limits: to simplify things, we take the column
            # *intersection* of (s1, s2). This gives the invariant that
            # the staffspace is limited from top and bottom in each of its columns.
            l = max(s1.left, s2.left)
            r = min(s1.right, s2.right)

            # Shift s1, s2 to the right by this much to have the cols. align
            # All of these are non-negative.
            dl1, dl2 = l - s1.left, l - s2.left
            dr1, dr2 = s1.right - r, s2.right - r

            # The stafflines are not necessarily straight,
            # so top is given for the *topmost bottom edge* of the top staffline + 1

            # First create mask
            canvas = numpy.zeros((s2.bottom - s1.top, r - l), dtype='uint8')

            # Paste masks into canvas.
            # This assumes that the top of the bottom staffline is below
            # the top of the top staffline... and that the bottom
            # of the top staffline is above the bottom of the bottom
            # staffline. This may not hold in very weird situations,
            # but it's good for now.
            logging.debug(s1.bounding_box, s1.mask.shape)
            logging.debug(s2.bounding_box, s2.mask.shape)
            logging.debug(canvas.shape)
            logging.debug('l={0}, dl1={1}, dl2={2}, r={3}, dr1={4}, dr2={5}'
                          ''.format(l, dl1, dl2, r, dr1, dr2))
            #canvas[:s1.height, :] += s1.mask[:, dl1:s1.width-dr1]
            #canvas[-s2.height:, :] += s2.mask[:, dl2:s2.width-dr2]

            # We have to deal with staffline interruptions.
            # One way to do this
            # is watershed fill: put markers along the bottom and top
            # edge, use mask * 10000 as elevation

            s1_above, s1_below = staffline_surroundings_mask(s1)
            s2_above, s2_below = staffline_surroundings_mask(s2)

            # Get bounding boxes of the individual stafflines' masks
            # that intersect with the staffspace bounding box, in terms
            # of the staffline bounding box.
            s1_t, s1_l, s1_b, s1_r = 0, dl1, \
                                     s1.height, s1.width - dr1
            s1_h, s1_w = s1_b - s1_t, s1_r - s1_l
            s2_t, s2_l, s2_b, s2_r = canvas.shape[0] - s2.height, dl2, \
                                     canvas.shape[0], s2.width - dr2
            s2_h, s2_w = s2_b - s2_t, s2_r - s2_l

            logging.debug(s1_t, s1_l, s1_b, s1_r, (s1_h, s1_w))

            # We now take the intersection of s1_below and s2_above.
            # If there is empty space in the middle, we fill it in.
            staffspace_mask = numpy.ones(canvas.shape)
            staffspace_mask[s1_t:s1_b, :] -= (1 - s1_below[:, dl1:s1.width-dr1])
            staffspace_mask[s2_t:s2_b, :] -= (1 - s2_above[:, dl2:s2.width-dr2])

            ss_top = s1.top
            ss_bottom = s2.bottom
            ss_left = l
            ss_right = r

            uid = CropObject.build_uid(dataset_namespace, docname, next_objid)

            staffspace = CropObject(next_objid, _CONST.STAFFSPACE_CLSNAME,
                                    top=ss_top, left=ss_left,
                                    height=ss_bottom - ss_top,
                                    width=ss_right - ss_left,
                                    mask=staffspace_mask,
                                    uid=uid)

            staffspace.inlinks.append(staff.objid)
            staff.outlinks.append(staffspace.objid)

            current_staffspace_cropobjects.append(staffspace)

            next_objid += 1

        ##########
        # Add top and bottom staffspace.
        # These outer staffspaces will have the width
        # of their bottom neighbor, and height derived
        # from its mask columns.
        # This is quite approximate, but it should do.

        # Upper staffspace
        tsl = sorted_stafflines[0]
        tsl_heights = tsl.mask.sum(axis=0)
        tss = current_staffspace_cropobjects[0]
        tss_heights = tss.mask.sum(axis=0)

        uss_top = max(0, tss.top - max(tss_heights))
        uss_left = tss.left
        uss_width = tss.width
        # We use 1.5, so that large noteheads
        # do not "hang out" of the staffspace.
        uss_height = int(tss.height / 1.2)
        # Shift because of height downscaling:
        uss_top += tss.height - uss_height
        uss_mask = tss.mask[:uss_height, :] * 1

        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        staffspace = CropObject(next_objid, _CONST.STAFFSPACE_CLSNAME,
                                top=uss_top, left=uss_left,
                                height=uss_height,
                                width=uss_width,
                                mask=uss_mask,
                                uid=uid)
        current_staffspace_cropobjects.append(staffspace)
        staff.outlinks.append(staffspace.objid)
        staffspace.inlinks.append(staff.objid)
        next_objid += 1

        # Lower staffspace
        bss = current_staffspace_cropobjects[-2]
        bss_heights = bss.mask.sum(axis=0)
        bsl = sorted_stafflines[-1]
        bsl_heights = bsl.mask.sum(axis=0)

        lss_top = bss.bottom # + max(bsl_heights)
        lss_left = bss.left
        lss_width = bss.width
        lss_height = int(bss.height / 1.2)
        lss_mask = bss.mask[:lss_height, :] * 1

        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        staffspace = CropObject(next_objid, _CONST.STAFFSPACE_CLSNAME,
                                top=lss_top, left=lss_left,
                                height=lss_height,
                                width=lss_width,
                                mask=lss_mask,
                                uid=uid)
        current_staffspace_cropobjects.append(staffspace)
        staff.outlinks.append(staffspace.objid)
        staffspace.inlinks.append(staff.objid)
        next_objid += 1

        staffspace_cropobjects += current_staffspace_cropobjects

    return staffspace_cropobjects