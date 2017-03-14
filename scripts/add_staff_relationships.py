#!/usr/bin/env python
"""The ``add_staff_relationships.py`` script automates adding
the relationships of some staff-related symbols to staffs.
The symbols in question are:

* ``staff_grouping``
* ``measure_separator``
* ``repeat``
* ``key_signature``
* ``time_signature``
* ``g-clef``, ``c-clef``, ``f-clef``

"""
from __future__ import print_function, unicode_literals
import argparse
import logging
import os
import time

import collections

from muscima.io import parse_cropobject_list, export_cropobject_list
from muscima.cropobject import link_cropobjects

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


STAFFLINE_CLSNAME = 'staff_line'
STAFF_CLSNAME = 'staff'

STAFF_RELATED_CLSNAMES = {
    'staff_grouping',
    'measure_separator',
    'key_signature',
    'time_signature',
    'g-clef', 'c-clef', 'f-clef', 'other-clef',
}

# Notes will get added separately.

##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-a', '--annot', action='store', required=True,
                        help='The annotation file for which the staffline and staff'
                             ' CropObject relationships should be added.')
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
    logging.info('Starting main...')
    _start_time = time.clock()

    ##########################################################################
    logging.info('Import the CropObject list')
    if not os.path.isfile(args.annot):
        raise ValueError('Annotation file {0} not found!'
                         ''.format(args.annot))
    cropobjects = parse_cropobject_list(args.annot)

    ##########################################################################
    logging.info('Find the staff-related symbols')
    staffs = [c for c in cropobjects if c.clsname == STAFF_CLSNAME]

    staff_related_symbols = collections.defaultdict(list)
    for c in cropobjects:
        if c.clsname in STAFF_RELATED_CLSNAMES:
            staff_related_symbols[c.clsname].append(c)

    ##########################################################################
    logging.info('Adding relationships')
    #  - Which direction do the relationships lead in?
    #    Need to define this.
    #
    # Staff -> symbol?
    # Symbol -> staff?
    # It does not really matter, but it's more intuitive to attach symbols
    # onto a pre-existing staff. So, symbol -> staff.
    for clsname, cs in staff_related_symbols.items():
        for c in cs:
            # Find the related staff. Relatedness is measured by row overlap.
            # That means we have to modify the staff bounding box to lead
            # from the leftmost to the rightmost column. This holds
            # especially for the staff_grouping symbols.
            for s in staffs:
                st, sl, sb, sr = s.bounding_box
                sl = 0
                sr = max(sr, c.right)
                if c.overlaps((st, sl, sb, sr)):
                    link_cropobjects(c, s)

    ##########################################################################
    logging.info('Export the combined list.')
    cropobject_string = export_cropobject_list(cropobjects)
    if args.export is not None:
        with open(args.export, 'w') as hdl:
            hdl.write(cropobject_string)
    else:
        print(cropobject_string)

    _end_time = time.clock()
    logging.info('add_staff_reationships.py done in {0:.3f} s'
                 ''.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
