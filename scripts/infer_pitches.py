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

We are currently NOT processing any transpositions.

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


ON_STAFFLINE_RATIO_TRHESHOLD = 0.2
'''Magic number for determining whether a notehead is *on* a ledger
line, or *next* to a ledger line: if the ratio between the smaller
and larger vertical difference of (top, bottom) vs. l.l. (top, bottom)
is smaller than this, it means the notehead is most probably *NOT*
on the l.l. and is next to it.'''

STAFF_CROPOBJECT_CLSNAMES = ['staff_line', 'staff_space', 'staff']

NOTEHEAD_CLSNAMES = {
    'notehead-full',
    'notehead-empty',
    'grace-notehead-full',
    'grace-notehead-empty',
}

CLEF_CLSNAMES = {
    'g-clef',
    'c-clef',
    'f-clef',
}

ACCIDENTAL_CLSNAMES = {
    'sharp': 1,
    'flat': -1,
    'natural': 0,
    'double_sharp': 2,
    'double_flat': -2,
}


# The individual MIDI codes for for the unmodified steps.
_fs = range(5, 114, 12)
_cs = range(0, 121, 12)
_gs = range(7, 116, 12)
_ds = range(2, 110, 12)
_as = range(9, 118, 12)
_es = range(4, 112, 12)
_bs = range(11, 120, 12)

KEY_TABLE_SHARPS = {
    0: {},
    1: {i: 1 for i in _fs},
    2: {i: 1 for i in _fs + _cs},
    3: {i: 1 for i in _fs + _cs + _gs},
    4: {i: 1 for i in _fs + _cs + _gs + _ds},
    5: {i: 1 for i in _fs + _cs + _gs + _ds + _as},
    6: {i: 1 for i in _fs + _cs + _gs + _ds + _as + _es},
    7: {i: 1 for i in _fs + _cs + _gs + _ds + _as + _es + _bs},
}

KEY_TABLE_FLATS = {
    0: {},
    1: {i: -1 for i in _bs},
    2: {i: -1 for i in _bs + _es},
    3: {i: -1 for i in _bs + _es + _as},
    4: {i: -1 for i in _bs + _es + _as + _ds},
    5: {i: -1 for i in _bs + _es + _as + _ds + _gs},
    6: {i: -1 for i in _bs + _es + _as + _ds + _gs + _cs},
    7: {i: -1 for i in _bs + _es + _as + _ds + _gs + _cs + _fs},
}


##############################################################################


def pitch_delta_from_staffline_delta(base_pitch, delta):
    """Given a base staffline MIDI pitch code and a difference (in
    stafflines + staffspaces), computes the MIDI pitch code
    for the staffline shifted by X.

    >>> pitch_delta_from_staffline_delta(62, 4)
    69

    :param base_pitch: The MIDI pitch code relative to which the ``delta``
        is given.

    :param delta: How many stafflines away is the staffline we want
        to recover pitch for? Use positive numbers for going above
        the base pitch and negative numbers for going below.

        Both stafflines and staffspaces count: the adjacent staffspace
        is +/-1, the next staffline is +/-2, etc.

    :return: The MIDI pitch code of the staffline with the given delta.
    """
    MI_CODES = (4, 11, 16, 17)
    FA_CODES = (0, 5, 12, 17)

    base_mod = base_pitch % 12
    base_octave = base_pitch // 12

    # Each octave is 7 stafflines/spaces
    delta_octaves = delta // 7
    delta_within_octave = delta % 7

    # Go UP (if delta < 0, then delta_octaves corrects for this)
    current_mod = base_mod
    for i in range(delta_within_octave):
        if current_mod in MI_CODES:
            current_mod += 1
        else:
            current_mod += 2

    current_pitch = current_mod + (12 * delta_octaves)
    current_pitch += (12 * base_octave)

    return current_pitch

##############################################################################


def get_outlink_staff(c, cropobjects_dict, allow_multi=False):
    # Assuming one staff per clef
    s_objids = [o for o in c.outlinks
                if cropobjects_dict[o].clsname == 'staff']

    if len(s_objids) == 0:
        raise ValueError('All noteheads should be connected to staff!'
                         ' Notehead: {0}'.format(c.uid))
        #return None

    if allow_multi:
        return [cropobjects_dict[objid] for objid in s_objids]

    s = cropobjects_dict[s_objids[0]]
    return s


def get_clef_state(stafflines, staffspaces, clef=None, clef_type=None):
    """Gives a MIDI pitch number to each staffline and staffspace.

    Either ``clef`` or ``clef_type`` must be supplied. If both
    are supplied, ``clef_type`` has priority, allowing overrides
    on weird clefs.

    For now, we assume all staves are 5-line, 6-space, and
    that all clefs are used in their standard positions.

    :returns: The clef state is a map from staffline/staffspace
        objid to the MIDI code to which it would correspond
        without any modifications from key signatures and/or
        accidentals.
    """
    if clef_type is None:
        if clef is not None:
            clef_type = clef.clsname
        else:
            raise ValueError('Must supply at least one of clef, clef_type!')

    if clef_type == 'g-clef':
        midi_codes = [62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79]
    elif clef_type == 'c-clef':
        midi_codes = [52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69]
    elif clef_type == 'f-clef':
        midi_codes = [41, 43, 45, 47, 48, 50, 52, 54, 55, 57, 59]

    # Sort them bottom-up.
    sorted_categories = sorted(stafflines+staffspaces, key=lambda s: s.top, reverse=True)

    state = {}
    for midi_code, s in zip(midi_codes,
                            sorted_categories):
        state[s.objid] = midi_code

    return state


def get_key_state(stafflines, staffspaces, clef_state,
                  key_signature=None, n_sharps=None, n_flats=None,
                  cropobjects_dict=None):
    """Gives a +1, 0, or -1 to each staffline/staffspace
    in the current clef state, according to the key signature
    (or, alternately, according to the given number of sharps/flats).

    If no key signature, n_sharps and n_flats is provided,
    it is assumed the key signature is empty and the state is all 0's.
    """
    # Sort them bottom-up.
    sorted_categories = sorted(stafflines+staffspaces, key=lambda s: s.top, reverse=True)

    if (key_signature is None) and (n_sharps is None) and (n_flats is None):
        state = {s.objid: 0 for s in sorted_categories}
        return state

    if n_sharps is None:
        if key_signature is None:
            raise ValueError('Cannot derive number of sharps without key signature CropObject!')
        if cropobjects_dict is None:
            raise ValueError('Cannot derive number of sharps without CropObject dict!')
        n_sharps = len([o for o in key_signature.outlinks if cropobjects_dict[o].clsname == 'sharp'])

    if n_flats is None:
        if key_signature is None:
            raise ValueError('Cannot derive number of flats without key signature CropObject!')
        if cropobjects_dict is None:
            raise ValueError('Cannot derive number of flats without CropObject dict!')
        n_flats = len([o for o in key_signature.outlinks if cropobjects_dict[o].clsname == 'flat'])

    # Key signature combining both is quite good e.g. for contemporary harp pieces.
    if n_sharps + n_flats > 7:
        raise ValueError('Too many sharps and flats together in one key signature!')

    key_map = {}
    if n_sharps > 0:
        key_map.update(KEY_TABLE_SHARPS[n_sharps])
    if n_flats > 0:
        key_map.update(KEY_TABLE_FLATS[n_flats])

    key_state = {s.objid: 0 for s in sorted_categories}
    for s in sorted_categories:
        if clef_state[s.objid] in key_map:
            key_state[s.objid] = key_map[clef_state[s.objid]]

    return key_state


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
    if not os.path.isfile(args.annot):
        raise ValueError('Annotation file {0} not found!'
                         ''.format(args.annot))
    cropobjects = parse_cropobject_list(args.annot)

    _cropobjects_dict = {c.objid: c for c in cropobjects}

    ##########################################################################

    # Collect staves.
    staves = [c for c in cropobjects if c.clsname == 'staff']
    print('We have {0} staves.'.format(len(staves)))

    # Collect clefs and key signatures per staff.
    clefs = [c for c in cropobjects if c.clsname in CLEF_CLSNAMES]
    key_signatures = [c for c in cropobjects if c.clsname == 'key_signature']

    clef_to_staff_map = {}
    # There may be more than one clef per staff.
    staff_to_clef_map = collections.defaultdict(list)
    for c in clefs:
        # Assuming one staff per clef
        s = get_outlink_staff(c, _cropobjects_dict)
        clef_to_staff_map[c.objid] = s
        staff_to_clef_map[s.objid].append(c)

    key_to_staff_map = {}
    # There may be more than one key signature per staff.
    staff_to_key_map = collections.defaultdict(list)
    for k in key_signatures:
        s = get_outlink_staff(k, _cropobjects_dict)
        key_to_staff_map[k.objid] = s
        staff_to_key_map[s.objid].append(k)

    # Collect measure separators.
    measure_separators = [c for c in cropobjects if c.clsname == 'measure_separator']

    staff_to_msep_map = collections.defaultdict(list)
    for m in measure_separators:
        _m_staves = get_outlink_staff(m, _cropobjects_dict,
                                      allow_multi=True)
        # (Measure separators might belong to multiple staves.)
        for s in _m_staves:
            staff_to_msep_map[s.objid].append(m)
            # Collect accidentals per notehead.

    # Collect noteheads.
    noteheads = [c for c in cropobjects if c.clsname in NOTEHEAD_CLSNAMES]
    staff_to_noteheads_map = collections.defaultdict(list)
    for n in noteheads:
        s = get_outlink_staff(n, _cropobjects_dict)
        staff_to_noteheads_map[s.objid].append(n)

    # Inference
    # ---------

    # For each staff:
    #  - read all relevant attached objects left-to-right
    #  - notehead events: get pitch
    #  - other events: specify interpretation of stafflines

    relevant_objects_per_staff = {}
    for s in staves:

        # Retrieve the stafflines and staffspaces and prepare indexes
        stafflines = [_cropobjects_dict[o] for o in s.outlinks
                      if _cropobjects_dict[o].clsname == 'staff_line']
        stafflines = sorted(stafflines, key=lambda x: x.top)
        staffline2idx = {sl.objid: i for i, sl in enumerate(stafflines)}
        idx2staffline = {i: sl for i, sl in enumerate(stafflines)}

        staffspaces = [_cropobjects_dict[o] for o in s.outlinks
                       if _cropobjects_dict[o].clsname == 'staff_space']
        staffspaces= sorted(staffspaces, key=lambda x: x.top)
        staffspace2idx = {ss.objid: i for i, ss in enumerate(staffspaces)}
        idx2staffspace = {i: ss for i, ss in enumerate(staffspaces)}

        # Prepare the inference engine state.
        # -----------------------------------
        #
        # The state has three components:
        #  - clef state,
        #  - key state,
        #  - accidentals state.
        #
        # Each of these three is a map that goes from each
        # staffline and staffspace to a pitch (expressed in MIDI).
        #
        # Clef state gives the actual MIDI number for the given
        # staffline, as though there was no key or accidental.
        #
        # Key state gives modifications in +1, 0, or -1 for each
        # staffline.
        #
        # In the same way, the Accidental state gives these
        # modifications.
        #
        # To obtain the pitch for a staffline/staffspace,
        # you need to add up the Clef, Key, and Accidental
        # state numbers for the given staffline/staffspace.
        #
        # Note that for ledger lines, you may need to look
        # an octave (or two) up/down.

        # Default: g-clef
        clef_state = get_clef_state(stafflines, staffspaces, clef_type='g-clef')

        # Default: empty key signature
        key_state = get_key_state(stafflines, staffspaces, clef_state, None)

        # Default: no accidentals
        accidental_state = {s.objid: 0
                            for s in stafflines + staffspaces}

        ll_accidental_state = {}
        '''Records the accidentals at ledger lines for the given
        measure. Each space AND line is +1 (or -1). First ledger line
        above is +1, space above it is +2, second l.l. is +3, etc.
        (It starts at 1 because of the outer staffspace.)'''

        queue = sorted(
                    staff_to_clef_map[s.objid]
                    + staff_to_key_map[s.objid]
                    + staff_to_msep_map[s.objid]
                    + staff_to_noteheads_map[s.objid],
                    key=lambda x: x.left)

        pitches = {}
        '''MIDI pitch code for each notehead objid.'''

        for q in queue:

            if q.clsname in CLEF_CLSNAMES:
                clef_state = get_clef_state(stafflines, staffspaces, clef=q)

            elif q.clsname == 'key_signature':
                key_state = get_key_state(stafflines, staffspaces, clef_state, key_signature=q,
                                          cropobjects_dict=_cropobjects_dict)

            elif q.clsname == 'measure_separator':
                accidental_state = {_s.objid: 0
                                    for _s in stafflines + staffspaces}
                ll_accidental_state = {}

            elif q.clsname in NOTEHEAD_CLSNAMES:

                staff_objects = [_cropobjects_dict[o] for o in q.outlinks
                                 if _cropobjects_dict[o].clsname in ('staff_line', 'staff_space')]
                if len(staff_objects) > 1:
                    raise ValueError('{0}: Noteheads should not be connected to more than one'
                                     ' staffline or staffspace!'.format(q.uid))

                ###############################
                # First find accidentals
                accidentals = [_cropobjects_dict[o] for o in q.outlinks
                               if _cropobjects_dict[o].clsname in ACCIDENTAL_CLSNAMES]

                total_modification = 0
                if len(accidentals) > 1:

                    # Check for consistency
                    if len(accidentals) > 2:
                        raise ValueError('{0}: A notehead should not have more than 2 accidentals!'
                                         ''.format(q.uid))
                    elif len([a for a in accidentals if a.clsname != 'natural']) > 1:
                        raise ValueError('{0}: A notehead should not have more than 1 non-natural accidental!'
                                         ''.format(q.uid))
                    elif len([a for a in accidentals if a.clsname == 'natural']) > 1:
                        raise ValueError('{0}: A notehead should not have more than 1 natural!'
                                         ''.format(q.uid))

                    total_modification = sum([ACCIDENTAL_CLSNAMES[a.clsname] for a in accidentals])

                if len(staff_objects) == 0:
                    # TODO: LEDGER LINES!!!
                    #logging.warning('{0}: Currently not doing ledger lines!'.format(q.objid))

                    # Processing ledger lines:
                    #  - count ledger lines
                    lls = [_cropobjects_dict[o] for o in q.outlinks
                           if _cropobjects_dict[o].clsname == 'ledger_line']
                    n_lls = len(lls)
                    if n_lls == 0:
                        raise ValueError('Notehead with no staffline or staffspace,'
                                         ' but also no ledger lines: {0}'.format(q.uid))

                    #  Determine: is notehead above or below staff?
                    is_above_staff = (q.top < s.top)

                    #  Determine: is notehead on/next to (closest) ledger line?
                    #    This needs to be done *after* we know whether the notehead
                    #    is above/below staff: if the notehead is e.g. above,
                    #    then it would be weird to find out it is in the
                    #    mini-staffspace *below* the closest ledger line,
                    #    signalling a mistake in the data.
                    closest_ll = min(lls, key=lambda x: x.top - q.top)
                    dtop = q.top - closest_ll.top
                    dbottom = closest_ll.bottom - q.bottom

                    _on_ledger_line = True
                    if min(dtop, dbottom) / max(dtop, dbottom) < ON_STAFFLINE_RATIO_TRHESHOLD:
                        _on_ledger_line = False

                        # Check orientation congruent with rel. to staff
                        if (dtop**2 > dbottom**2) and not is_above_staff:
                            raise ValueError('Notehead in LL space with wrong orientation w.r.t. staff:'
                                             ' {0}'.format(q.uid))
                        if (dbottom**2 > dtop**2) and is_above_staff:
                            raise ValueError('Notehead in LL space with wrong orientation w.r.t. staff:'
                                             ' {0}'.format(q.uid))

                    #  Determine base pitch.
                    #
                    #   - Get base pitch of outermost ledger line.
                    if is_above_staff:
                        outer_staffline = min(stafflines, key=lambda x: x.bottom)
                    else:
                        outer_staffline = max(stafflines, key=lambda x: x.top)
                    _delta = 2 * n_lls - 1
                    if not _on_ledger_line:
                        _delta += 1
                    if not is_above_staff:
                        _delta *= -1
                    ll_base_pitch = pitch_delta_from_staffline_delta(
                        clef_state[outer_staffline.objid],
                        _delta
                    )

                    #  Check for key signature.
                    #   - mod 12
                    for _s in staff_objects:
                        if (clef_state[_s.objid] % 12) == (ll_base_pitch % 12):
                            ll_base_pitch += key_state[_s.objid]
                            break
                    ll_pitch = ll_base_pitch

                    #  Check for accidentals.
                    if (_delta not in ll_accidental_state) \
                            or (len(accidentals) > 0):
                        ll_accidental_state[_delta] = total_modification

                    ll_pitch += ll_accidental_state[_delta]

                    midi_code = ll_pitch

                else:

                    # Update *accidentals* state, not *ll_accidentals*:
                    if len(accidentals) > 0:
                        accidental_state[s.objid] = total_modification

                    cs = staff_objects[0]
                    # Now read the pitch:
                    midi_code = clef_state[cs.objid] + key_state[cs.objid] + accidental_state[cs.objid]

                # Set pitch to the given MIDI code
                pitches[q.objid] = midi_code

        pprint.pprint('Staff: {0}'.format(s.uid))
        # pprint.pprint(pitches)

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
