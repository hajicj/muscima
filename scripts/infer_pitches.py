#!/usr/bin/env python
"""This is a script that takes the full-grown notation graph
and recovers for each notehead the pitch to which it corresponds.

Assumptions
-----------

* Clefs are used in a standard way: G-clef on 4th, C-clef on 3rd, F-clef
  on 2nd staffline.
* Key signatures are used in a standard way, so that we can rely on counting
  the accidentals.
* Accidentals are valid up until the end of the bar.

Representation
--------------

Notes are not noteheads. Pitch is associated with a note, and it is derived
from the notehead's subgraph. The current goal of this exercise is obtaining
MIDI, so we discard in effect information about what is e.g. a G-sharp
and A-flat.

"""
from __future__ import print_function, unicode_literals, division
import argparse
import logging
import os
import pprint
import time

import collections

import itertools
import numpy

from muscima.io import parse_cropobject_list, export_cropobject_list
from muscima.cropobject import link_cropobjects


__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


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


##############################################################################


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # Your code goes here
    raise NotImplementedError()

    _end_time = time.clock()
    logging.info('[XXXX] done in {0:.3f} s'.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
