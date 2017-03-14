#!/usr/bin/env python
"""The script ``add_staffline_symbols.py`` takes as input a CVC-MUSCIMA
(page, writer) index and a corresponding CropObjectList file
and adds to the CropObjectList staffline and staff objects."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import os
import time

import numpy

from skimage.io import imread
from skimage.measure import label
import matplotlib.pyplot as plt

from muscima.dataset import CVC_MUSCIMA
from muscima.io import parse_cropobject_list, export_cropobject_list
from muscima.cropobject import CropObject


__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


STAFFLINE_CLSNAME = 'staff_line'
STAFF_CLSNAME = 'staff'


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


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-n', '--number', action='store', type=int,
                        required=True,
                        help='Number of the CVC-MUSCIMA page (1 - 20)')
    parser.add_argument('-w', '--writer', action='store', type=int,
                        required=True,
                        help='Writer of the CVC-MUSCIMA page (1 - 50)')

    parser.add_argument('-r', '--root', action='store',
                        default=os.getenv('CVC_MUSCIMA_ROOT', None),
                        help='Path to CVC-MUSCIMA dataset root. By default, will attempt'
                             ' to read the CVC_MUSCIMA_ROOT env var. If that does not'
                             ' work, the script will fail.')

    parser.add_argument('-a', '--annot', action='store', required=True,
                        help='The annotation file for which the staffline and staff'
                             ' CropObjects should be added.')
    parser.add_argument('-e', '--export', action='store',
                        help='A filename to which the output CropObjectList'
                             ' should be saved. If not given, will print to'
                             ' stdout.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.warning('Starting main...')
    _start_time = time.clock()

    ########################################################
    # Load gt image.
    logging.warning('Loading staffline image.')
    #  - Initialize Dataset. This checks for the root.
    cvc_dataset = CVC_MUSCIMA(root=args.root)

    # - Load the image.
    imfile = cvc_dataset.imfile(page=args.number, writer=args.writer, distortion='ideal',
                                mode='staff_only')
    gt = (imread(imfile, as_grey=True) * 255).astype('uint8')
    # - Cast as binary mask.
    gt[gt > 0] = 1

    ########################################################
    # Locate stafflines in gt image.
    logging.warning('Getting staffline connected components.')

    #  - Get connected components in gt image.
    cc, labels, bboxes = compute_connected_components(gt)

    #  - Use vertical dimension of CCs to determine which ones belong together
    #    to form stafflines. (Criterion: row overlap.)
    n_rows, n_cols = gt.shape
    intervals = [[] for _ in range(n_rows)] # For each row: which CCs have pxs on that row?
    for label, (t, l, b, r) in bboxes.items():
        if label == 0:
            continue
        for r in range(t, b):
            intervals[r].append(label)

    logging.warning('Grouping staffline connected components into stafflines.')
    staffline_components = []   # For each staffline, we collect the CCs that it is made of
    _in_staffline = False
    _current_staffline_components = []
    for r in intervals:
        if not _in_staffline:
            if len(r) == 0:
                continue
            else:
                _in_staffline = True
                _current_staffline_components += r
        else:
            if len(r) == 0:
                staffline_components.append(set(_current_staffline_components))
                _current_staffline_components = []
                _in_staffline = False
                continue
            else:
                _current_staffline_components += r

    logging.warning('No. of stafflines, with component groups: {0}'
                    ''.format(len(staffline_components)))

    # Now: merge the staffline components into one bbox/mask.
    logging.warning('Merging staffline components into staffline bboxes and masks.')
    staffline_bboxes = []
    staffline_masks = []
    for sc in sorted(staffline_components,
                     key=lambda c: min([bboxes[cc][0]
                                        for cc in c])):  # Sorted top-down
        st, sl, sb, sr = n_rows, n_cols, 0, 0
        for component in sc:
            t, l, b, r = bboxes[component]
            st, sl, sb, sr = min(t, st), min(l, sl), max(b, sb), max(r, sr)
        _sm = gt[st:sb, sl:sr]
        staffline_bboxes.append((st, sl, sb, sr))
        staffline_masks.append(_sm)


    # Check if n. of stafflines is divisible by 5
    n_stafflines = len(staffline_bboxes)
    logging.warning('\tTotal stafflines: {0}'.format(n_stafflines))
    if n_stafflines % 5 != 0:
        raise ValueError('No. of stafflines is not divisible by 5!')

    logging.warning('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5*i:5*(i+1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(gt[_st:_sb, _sl:_sr])

    logging.warning('Total staffs: {0}'.format(len(staff_bboxes)))

    ##################################################################
    # (Optionally fill in missing pixels, based on full image.)
    logging.warning('SKIP: fill in missing pixels based on full image.')
    #  - Load full image
    #  - Find gap regions
    #  - Obtain gap region masks from full image
    #  - Add gap region mask to staffline mask.

    # Create the CropObjects for stafflines and staffs:
    #  - Load corresponding annotation, to which the stafflines and
    #    staves should be added. (This is needed to correctly set docname
    #    and objids.)
    if not os.path.isfile(args.annot):
        raise ValueError('Annotation file {0} does not exist!'.foramt(args.annot))

    logging.warning('Creating cropobjects...')
    cropobjects = parse_cropobject_list(args.annot)
    logging.warning('Non-staffline cropobjects: {0}'.format(len(cropobjects)))

    next_objid = max([c.objid for c in cropobjects]) + 1
    dataset_namespace = cropobjects[0].dataset
    docname = cropobjects[0].doc

    #  - Create the staffline CropObjects
    staffline_cropobjects = []
    for sl_bb, sl_m in zip(staffline_bboxes, staffline_masks):
        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        t, l, b, r = sl_bb
        c = CropObject(objid=next_objid,
                       clsname=STAFFLINE_CLSNAME,
                       top=t, left=l, height=b - t, width=r - l,
                       mask=sl_m,
                       uid=uid)
        staffline_cropobjects.append(c)
        next_objid += 1

    #  - Create the staff CropObjects
    staff_cropobjects = []
    for s_bb, s_m in zip(staff_bboxes, staff_masks):
        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        t, l, b, r = s_bb
        c = CropObject(objid=next_objid,
                       clsname=STAFF_CLSNAME,
                       top=t, left=l, height=b - t, width=r - l,
                       mask=s_m,
                       uid=uid)
        staff_cropobjects.append(c)
        next_objid += 1

    #  - Add the inlinks/outlinks
    for i, sc in enumerate(staff_cropobjects):
        sl_from = 5 * i
        sl_to = 5 * (i + 1)
        for sl in staffline_cropobjects[sl_from:sl_to]:
            sl.inlinks.append(sc.objid)
            sc.outlinks.append(sl.objid)

    # - Join the lists together
    cropobjects_with_staffs = cropobjects \
                              + staffline_cropobjects \
                              + staff_cropobjects

    logging.warning('Exporting the new cropobject list: {0} objects'
                    ''.format(len(cropobjects_with_staffs)))
    # - Export the combined list.
    cropobject_string = export_cropobject_list(cropobjects_with_staffs)
    if args.export is not None:
        with open(args.export, 'w') as hdl:
            hdl.write(cropobject_string)
    else:
        print(cropobject_string)

    _end_time = time.clock()
    logging.warning('add_staffline_symbols.py done in {0:.3f} s'
                    ''.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
