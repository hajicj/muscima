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
import copy
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


class PitchInferenceEngineConstants(object):
    """This class stores the constants used for pitch inference."""

    ON_STAFFLINE_RATIO_TRHESHOLD = 0.2
    '''Magic number for determining whether a notehead is *on* a ledger
    line, or *next* to a ledger line: if the ratio between the smaller
    and larger vertical difference of (top, bottom) vs. l.l. (top, bottom)
    is smaller than this, it means the notehead is most probably *NOT*
    on the l.l. and is next to it.'''

    STAFF_CROPOBJECT_CLSNAMES = ['staff_line', 'staff_space', 'staff']

    STAFFLINE_CROPOBJECT_CLSNAMES = ['staff_line', 'staff_space']

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

    KEY_SIGNATURE_CLSNAMES = {
        'key_signature',
    }

    MEASURE_SEPARATOR_CLSNAMES = {
        'measure_separator',
    }

    FLAGS_CLSNAMES = {
        '8th_flag',
        '16th_flag',
        '32th_flag',
        '64th_and_higher_flag',
    }

    BEAM_CLSNAMES = {
        'beam',
    }

    FLAGS_AND_BEAMS ={
        '8th_flag',
        '16th_flag',
        '32th_flag',
        '64th_and_higher_flag',
        'beam',
    }

    ACCIDENTAL_CLSNAMES = {
        'sharp': 1,
        'flat': -1,
        'natural': 0,
        'double_sharp': 2,
        'double_flat': -2,
    }

    MIDI_CODE_RESIDUES_FOR_PITCH_STEPS = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'Eb',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'Ab',
        9: 'A',
        10: 'Bb',
        11: 'B',
    }
    '''Simplified pitch naming.'''

    # The individual MIDI codes for for the unmodified steps.
    _fs = list(range(5, 114, 12))
    _cs = list(range(0, 121, 12))
    _gs = list(range(7, 116, 12))
    _ds = list(range(2, 110, 12))
    _as = list(range(9, 118, 12))
    _es = list(range(4, 112, 12))
    _bs = list(range(11, 120, 12))

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

    PITCH_STEPS = ['C', 'D', 'E', 'F', 'G', 'A', 'B',
                   'C', 'D', 'E', 'F', 'G', 'A', 'B']
    # Wrap around twice for easier indexing.

    ACCIDENTAL_CODES = {'sharp': '#', 'flat': 'b',
                        'double_sharp': 'x', 'double_flat': 'bb'}

    REST_CLSNAMES = {
        'whole_rest',
        'half_rest',
        'quarter_rest',
        '8th_rest',
        '16th_rest',
        '32th_rest',
        '64th_and_higher_rest',
        'multi-measure_rest',
    }

    TIME_SIGNATURES = {
        'time_signature',
    }

    @property
    def clsnames_affecting_onsets(self):
        """Returns a list of CropObject class names for objects
        that affect onsets. Assumes notehead and rest durations
        have already been given."""
        output = set()
        output.update(self.NOTEHEAD_CLSNAMES)
        output.update(self.REST_CLSNAMES)
        output.update(self.MEASURE_SEPARATOR_CLSNAMES)
        output.update(self.TIME_SIGNATURES)
        output.add('repeat_measure')
        return output

    @property
    def clsnames_bearing_duration(self):
        """Returns the list of classes that actually bear duration,
        i.e. contribute to onsets of their descendants in the precedence
        graph."""
        output = set()
        output.update(self.NOTEHEAD_CLSNAMES)
        output.update(self.REST_CLSNAMES)
        return output


class PitchInferenceEngineState(object):
    """This class represents the state of the MIDI pitch inference
    engine during inference.

    Reading pitch is a stateful operations. One needs to remember
    how stafflines and staffspaces map to pitch codes. This is governed
    by two things:

    * The clef, which governs
    * The accidentals: key signatures and inline accidentals.

    Clef and key signature have unlimited scope which only changes when
    a new key signature is introduced (or the piece ends). The key
    signature affects all pitches in the given step class (F, C, G, ...)
    regardless of octave. Inline accidentals have scope until the next
    measure separator, and they are only valid within their own octave.

    The pitch inference algorithm is run for each staff separately.

    Base pitch representation
    -------------------------

    The base pitch corresponds to the pitch encoded by a notehead
    simply sitting on the given staffline/staffspace, without any
    regard to accidentals (key signature or inline). It is computed
    by *distance from center staffline* of a staff, with positive
    distance corresponding to going *up* and negative for going *down*
    from the center staffline.

    Accidentals representation
    --------------------------

    The accidentals are associated also with each staffline/staffspace,
    as counted from the current center. (This means i.a. that
    the octave periodicity is 7, not 12.)

    There are two kinds of accidentals based on scope: key signature,
    and inline. Inline accidentals are valid only up to the next
    measure_separator, while key signature accidentals are valid
    up until the key signature changes. Key signature accidentals
    also apply across all octaves, while inline accidentals only apply
    on the specific staffline.

    Note that inline accidentals may *cancel* key signature
    accidentals: they override the key signature when given.

    Key accidentals are given **mod 7**.

    Pitch inference procedure
    -------------------------

    Iterate through the relevant objects on a staff, sorted left-to-right
    by left edge.
    """
    def __init__(self):

        self.base_pitch = None
        '''The MIDI code corresponding to the middle staffline,
        without modification by key or inline accidentals.'''

        self.base_pitch_step = None
        '''The name of the base pitch: C, D, E, etc.'''

        self.base_pitch_octave = None
        '''The octave where the pitch resides. C4 = c', the middle C.'''

        self._current_clef = None
        '''Holds the clef CropObject that is currently valid.'''

        self._current_delta_steps = None
        '''Holds for each staffline delta step (i.e. staffline delta mod 7)
        the MIDI pitch codes.'''

        self.key_accidentals = {}
        self.inline_accidentals = {}

    def reset(self):
        self.base_pitch = None
        self._current_clef = None
        self._current_delta_steps = None
        self.key_accidentals = {}
        self.inline_accidentals = {}

    def init_base_pitch(self, clef=None):
        """Based on the clef, initialize the base pitch.
        By default, initializes as though given a g-clef."""
        if (clef is None) or (clef.clsname == 'g-clef'):
            new_base_pitch = 71
            new_delta_steps = [0, 1, 2, 2, 1, 2, 2, 2]
            new_base_pitch_step = 6  # Index into pitch steps.
            new_base_pitch_octave = 4
        elif clef.clsname == 'c-clef':
            new_base_pitch = 60
            new_delta_steps = [0, 2, 2, 1, 2, 2, 2, 1]
            new_base_pitch_step = 0
            new_base_pitch_octave = 4
        elif clef.clsname == 'f-clef':
            new_base_pitch = 50
            new_delta_steps = [0, 2, 1, 2, 2, 2, 1, 2]
            new_base_pitch_step = 1
            new_base_pitch_octave = 3
        else:
            raise ValueError('Unrecognized clef clsname: {0}'
                             ''.format(clef.clsname))

        # Shift the key and inline accidental deltas
        # according to the change.
        if self._current_clef is not None:
            if clef.clsname != self._current_clef.clsname:
                # From G to C clef: everything is now +4
                new_key_accidentals = {
                    (d + 4) % 7: v for d, v in self.key_accidentals.items()
                }
                new_inline_accidentals = {
                    d + 4: v for d, v in self.inline_accidentals.items()
                }
                self.key_accidentals = new_key_accidentals
                self.inline_accidentals = new_inline_accidentals

        self.base_pitch = new_base_pitch
        self.base_pitch_step = new_base_pitch_step
        self.base_pitch_octave = new_base_pitch_octave
        self._current_clef = clef
        self._current_delta_steps = new_delta_steps

    def set_key(self, n_sharps=0, n_flats=0):
        """Initialize the staffline delta --> key accidental map.
        Currently works only on standard key signatures, where
        there are no repeating accidentals, no double sharps/flats,
        and the order of accidentals is the standard major/minor system.

        However, we can deal at least with key signatures that combine
        sharps and flats (if not more than 7), as seen e.g. in harp music.

        :param n_sharps: How many sharps are there in the key signature?

        :param n_flats: How many flats are there in the key signature?
        """
        if n_flats + n_sharps > 7:
            raise ValueError('Cannot deal with key signature that has'
                             ' more than 7 sharps + flats!')

        if self.base_pitch is None:
            raise ValueError('Cannot initialize key if base pitch is not known.')

        new_key_accidentals = {}

        # The pitches (F, C, G, D, ...) have to be re-cast
        # in terms of deltas, mod 7.
        if self._current_clef.clsname == 'g-clef':
            deltas_sharp = [4, 1, 5, 2, 6, 3, 0]
            deltas_flat = [0, 3, 6, 2, 5, 1, 4]
        elif self._current_clef.clsname == 'c-clef':
            deltas_sharp = [3, 0, 4, 1, 5, 2, 6]
            deltas_flat = [6, 2, 5, 1, 4, 0, 3]
        elif self._current_clef.clsname == 'f-clef':
            deltas_sharp = [2, 6, 3, 0, 4, 1, 5]
            deltas_flat = [5, 1, 4, 0, 3, 6, 2]

        for d in deltas_sharp[:n_sharps]:
            new_key_accidentals[d] = 'sharp'
        for d in deltas_flat[:n_flats]:
            new_key_accidentals[d] = 'flat'

        self.key_accidentals = new_key_accidentals

    def set_inline_accidental(self, delta, accidental):
        self.inline_accidentals[delta] = accidental.clsname

    def reset_inline_accidentals(self):
        self.inline_accidentals = {}

    def accidental(self, delta):
        """Returns the modification, in MIDI code, corresponding
        to the staffline given by the delta."""
        pitch_mod = 0

        step_delta = delta % 7
        if step_delta in self.key_accidentals:
            if self.key_accidentals[step_delta] == 'sharp':
                pitch_mod = 1
            elif self.key_accidentals[step_delta] == 'double_sharp':
                pitch_mod = 2
            elif self.key_accidentals[step_delta] == 'flat':
                pitch_mod = -1
            elif self.key_accidentals[step_delta] == 'double_flat':
                pitch_mod = -2

        # Inline accidentals override key accidentals.
        if delta in self.inline_accidentals:
            if self.inline_accidentals[delta] == 'natural':
                logging.info('Natural at delta = {0}'.format(delta))
                pitch_mod = 0
            elif self.inline_accidentals[delta] == 'sharp':
                pitch_mod = 1
            elif self.inline_accidentals[delta] == 'double_sharp':
                pitch_mod = 2
            elif self.inline_accidentals[delta] == 'flat':
                pitch_mod = -1
            elif self.inline_accidentals[delta] == 'double_flat':
                pitch_mod = -2
        return pitch_mod

    def pitch(self, delta):
        """Given a staffline delta, returns the current MIDI pitch code.

        (This method is the main interface of the PitchInferenceEngineState.)

        :delta: Distance in stafflines + staffspaces from the middle staffline.
            Negative delta means distance *below*, positive delta is *above*.

        :returns: The MIDI pitch code for the given delta.
        """

        # Split this into octave and step components.
        delta_step = delta % 7
        delta_octave = delta // 7

        # From the base pitch and clef:
        step_pitch = self.base_pitch \
                     + sum(self._current_delta_steps[:delta_step+1]) \
                     + (delta_octave * 12)
        accidental_pitch = self.accidental(delta)

        return step_pitch + accidental_pitch

    def pitch_name(self, delta):
        """Given a staffline delta, returns the name of the corrensponding pitch."""
        output_step = PitchInferenceEngineConstants.PITCH_STEPS[(self.base_pitch_step + delta) % 7]
        output_octave = self.base_pitch_octave + (delta // 7)
        output_mod = ''

        accidental = self.accidental(delta)
        if accidental == 1:
            output_mod = PitchInferenceEngineConstants.ACCIDENTAL_CODES['sharp']
        elif accidental == 2:
            output_mod = PitchInferenceEngineConstants.ACCIDENTAL_CODES['double_sharp']
        elif accidental == -1:
            output_mod = PitchInferenceEngineConstants.ACCIDENTAL_CODES['flat']
        elif accidental == 2:
            output_mod = PitchInferenceEngineConstants.ACCIDENTAL_CODES['double_flat']

        return output_step + output_mod, output_octave


class MIDIInferenceEngine(object):
    """The Pitch Inference Engine extracts MIDI from the notation
    graph. To get the MIDI, there are two streams of information
    that need to be combined: pitches and onsets, where the onsets
    are necessary both for ON and OFF events.

    Pitch inference is done through the ``infer_pitches()`` method.

    Onsets inference is done in two stages. First, the durations
    of individual notes (and rests) are computed, then precedence
    relationships are found and based on the precedence graph
    and durations, onset times are computed.

    Onset inference
    ---------------

    Onsets are computed separately by measure, which enables time
    signature constraint checking.

    (This can be implemented in the precedence graph structure,
    by (a) not allowing precedence edges to cross measure separators,
    (b) chaining measure separators, or it can be implemented
    directly in code. The first option is way more elegant.)

    Creating the precedence graph
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * Get measure separators.
    * Chain measure separators in precedence relationships.
    * Group cropobjects by bins between measure separators.
    * For each staff participating in the current measure
      (as defined by the relevant measure separator outlinks):

        * Infer precedence between the participating notes & rests,
        * Attach the sources of the resulting DAG to the leftward
          measure_separator (if there is none, just leave them
          as sources).

    Invariants
    ^^^^^^^^^^

    * There is exactly one measure separator starting each measure,
      except for the first measure, which has none. That implies:
      when there are multiple disconnected barlines marking the interface
      of the same two measures within a system, they are joined under
      a single measure_separator anyway.
    * Staff groupings are correct, and systems are read top-down.

    """
    def __init__(self):
        # Inference engine constants
        self._CONST = PitchInferenceEngineConstants()

        # Static temp data from which the pitches are inferred
        self._cdict = {}

        self.staves = None

        self.clefs = None
        self.clef_to_staff_map = None
        self.staff_to_clef_map = None

        self.key_signatures = None
        self.key_to_staff_map = None
        self.staff_to_key_map = None

        self.measure_separators = None
        self.staff_to_msep_map = None

        self.noteheads = None
        self.staff_to_noteheads_map = None

        # Dynamic temp data: things that change as the pitches are inferred.
        self.pitch_state = PitchInferenceEngineState()

        # Results
        self.pitches = None
        self.pitches_per_staff = None

        self.pitch_names = None
        self.pitch_names_per_staff = None

        self.durations_beats = None
        self.durations_beats_per_staff = None

    def reset(self):
        self.__init__()

    def infer_pitches(self, cropobjects, with_names=False):
        """The main workhorse for pitch inference.
        Gets a list of CropObjects and for each notehead-type
        symbol, outputs a MIDI code corresponding to the pitch
        encoded by that notehead.

        Notehead
        --------

        * Check for ties; if there is an incoming tie, apply
          the last pitch. (This is necessary because of ties
          that go across barlines and maintain inline accidentals.)
        * Determine its staffline delta from the middle staffline.
        * Check for inline accidentals, apply them to inference state.
        * Query pitch state with this staffline delta.

        Ties are problematic, because they may reach across
        staff breaks. This can only be resolved after all staves
        are resolved and assigned to systems, because until then,
        it is not clear which staff corresponds to which in the next
        system. Theoretically, this is near-impossible to resolve,
        because staves may not continue on the next system (e.g.,
        instruments that do not play for some time in orchestral scores),
        so simple staff counting is not foolproof. Some other matching
        mechanism has to be found, e.g. matching outgoing and incoming
        ties on the end and beginning of adjacent systems.

        Measure separator
        -----------------

        * Reset all inline accidentals to empty.

        Clef change
        -----------

        * Change base pitch
        * Recompute the key and inline signature delta indexes

        Key change
        ----------

        * Recompute key deltas

        :param with_names: If set, will return also a dict of
            objid --> pitch names (e.g., {123: 'F#3'}).

        :returns: A dict of ``objid`` to MIDI pitch code, with
            an entry for each (pitched) notehead. If ``with_names``
            is given, returns a tuple with the objid --> MIDI
            and objid --> pitch name dicts.

        """
        self._cdict = {c.objid: c for c in cropobjects}

        # Initialize pitch temp data.
        self._collect_symbols_for_pitch_inference(cropobjects)

        # Staff processing: this is where the inference actually
        # happens.
        self.pitches_per_staff = {}
        self.pitches = {}
        self.pitch_names_per_staff = {}
        self.pitch_names = {}
        self.durations_beats = {}
        self.durations_beats_per_staff = {}

        for staff in self.staves:
            self.process_staff(staff)
            self.pitches.update(self.pitches_per_staff[staff.objid])

        if with_names:
            return copy.deepcopy(self.pitches), copy.deepcopy(self.pitch_names)
        else:
            return copy.deepcopy(self.pitches)

    def process_staff(self, staff):

        self.pitches_per_staff[staff.objid] = {}
        self.pitch_names_per_staff[staff.objid] = {}

        self.durations_beats_per_staff[staff.objid] = {}

        self.pitch_state.reset()
        self.pitch_state.init_base_pitch()

        queue = sorted(
                    self.staff_to_clef_map[staff.objid]
                    + self.staff_to_key_map[staff.objid]
                    + self.staff_to_msep_map[staff.objid]
                    + self.staff_to_noteheads_map[staff.objid],
                    key=lambda x: x.left)

        for q in queue:
            if q.clsname in self._CONST.CLEF_CLSNAMES:
                self.process_clef(q)
            elif q.clsname in self._CONST.KEY_SIGNATURE_CLSNAMES:
                self.process_key_signature(q)
            elif q.clsname in self._CONST.MEASURE_SEPARATOR_CLSNAMES:
                self.process_measure_separator(q)
            elif q.clsname in self._CONST.NOTEHEAD_CLSNAMES:
                p, pn = self.process_notehead(q, with_name=True)
                self.pitches[q.objid] = p
                self.pitches_per_staff[staff.objid][q.objid] = p
                self.pitch_names[q.objid] = pn
                self.pitch_names_per_staff[staff.objid][q.objid] = pn

                b = self.beats(q)
                self.durations_beats[q.objid] = b
                self.durations_beats_per_staff[staff.objid][q.objid] = b

        return self.pitches_per_staff[staff.objid]

    def process_notehead(self, notehead, with_name=False):
        """This is the main workhorse of the pitch inference engine.

        :param notehead: The notehead-class CropObject for which we
            want to infer pitch.

        :param with_name: If set, will return not only the MIDI pitch
            code, but the name of the encoded note (e.g., F#3) as well.
        """
        # Processing ties
        # ---------------
        ties = self.__children(notehead, 'tie')
        for t in ties:
            tied_noteheads = self.__parents(t, self._CONST.NOTEHEAD_CLSNAMES)

            # Corner cases: mistakes and staff breaks
            if len(tied_noteheads) > 2:
                raise ValueError('Tie {0}: joining together more than 2'
                                 ' noteheads!'.format(t.uid))
            if len(tied_noteheads) < 2:
                logging.warning('Tie {0}: only one notehead. Staff break?'
                                ''.format(t.uid))
                continue

            left_tied_notehead = min(tied_noteheads, key=lambda x: x.left)
            if left_tied_notehead.objid != notehead.objid:
                try:
                    p = self.pitches[left_tied_notehead.objid]
                    if with_name:
                        pn = self.pitch_names[left_tied_notehead.objid]
                        return p, pn
                    else:
                        return p

                except KeyError:
                    raise KeyError('Processing tied notehead {0}:'
                                   ' preceding notehead {1} has no pitch!'
                                   ''.format(notehead.uid, left_tied_notehead.uid))

            # If the condition doesn't hold, then this is the leftward
            # note in the tie, and its pitch needs to be determined.

        # Obtain notehead delta
        # ---------------------
        delta = self.staffline_delta(notehead)

        # ### DEBUG
        # if notehead.objid == 200:
        #     logging.info('Notehead {0}: delta {1}'.format(notehead.uid, delta))
        #     logging.info('\tdelta_step: {0}'.format(delta % 7))
        #     logging.info('\tdelta_step pitch sum: {0}'
        #                  ''.format(sum(self.pitch_state._current_delta_steps[:(delta % 7)+1])))

        # Processing inline accidentals
        # -----------------------------
        accidentals = self.__children(notehead, self._CONST.ACCIDENTAL_CLSNAMES)

        if len(accidentals) > 0:

            # Sanity checks
            if len(accidentals) > 2:
                raise ValueError('More than two accidentals attached to notehead'
                                 ' {0}'.format(notehead.uid))
            elif len(accidentals) == 2:
                naturals = [a for a in accidentals if a.clsname == 'natural']
                non_naturals = [a for a in accidentals if a.clsname != 'natural']
                if len(naturals) == 0:
                    raise ValueError('More than one non-natural accidental'
                                     ' attached to notehead {0}'
                                     ''.format(notehead.uid))
                if len(non_naturals) == 0:
                    raise ValueError('Two naturals attached to one notehead {0}'
                                     ''.format(notehead.uid))

                self.pitch_state.set_inline_accidental(delta, non_naturals[0])

            elif len(accidentals) == 1:
                self.pitch_state.set_inline_accidental(delta, accidentals[0])

        # Get the actual pitch
        # --------------------
        p = self.pitch_state.pitch(delta)

        if with_name is True:
            pn = self.pitch_state.pitch_name(delta)
            return p, pn
        else:
            return p

    def staffline_delta(self, notehead):
        """Computes the staffline delta (distance from middle stafflines,
        measured in stafflines and staffspaces) for the given notehead.
        Accounts for ledger lines.
        """
        current_staff = self.__children(notehead, ['staff'])[0]
        staffline_objects = self.__children(notehead,
                                            self._CONST.STAFFLINE_CROPOBJECT_CLSNAMES)

        # Ledger lines
        # ------------
        if len(staffline_objects) == 0:

            # Processing ledger lines:
            #  - count ledger lines
            lls = self.__children(notehead, 'ledger_line')
            n_lls = len(lls)
            if n_lls == 0:
                raise ValueError('Notehead with no staffline or staffspace,'
                                 ' but also no ledger lines: {0}'
                                 ''.format(notehead.uid))

            #  Determine: is notehead above or below staff?
            is_above_staff = (notehead.top < current_staff.top)

            #  Determine: is notehead on/next to (closest) ledger line?
            #    This needs to be done *after* we know whether the notehead
            #    is above/below staff: if the notehead is e.g. above,
            #    then it would be weird to find out it is in the
            #    mini-staffspace *below* the closest ledger line,
            #    signalling a mistake in the data.
            closest_ll = min(lls, key=lambda x: x.top - notehead.top)
            # Determining whether the notehead is on a ledger
            # line or in the adjacent temp staffspace.
            # This uses a magic number, ON_STAFFLINE_RATIO_THRESHOLD.
            _on_ledger_line = True
            # Weird situation with notehead vertically *inside* bbox
            # of ledger line (could happen with slanted LLs and very small
            # noteheads).
            if closest_ll.top <= notehead.top <= notehead.bottom <= closest_ll.bottom:
                _on_ledger_line = True
            elif closest_ll.top > notehead.bottom:
                _on_ledger_line = False
            elif notehead.top > closest_ll.bottom:
                _on_ledger_line = False
            else:
                if notehead.top < closest_ll.top <= closest_ll.bottom < notehead.bottom:
                    dtop = closest_ll.top - notehead.top
                    dbottom = notehead.bottom - closest_ll.bottom
                elif notehead.top < closest_ll.top <= notehead.bottom <= closest_ll.bottom:
                    dtop = closest_ll.top - notehead.top
                    dbottom = max(closest_ll.bottom - notehead.bottom, 1)
                elif closest_ll.top <= notehead.top <= closest_ll.bottom < notehead.bottom:
                    dtop = max(notehead.top - closest_ll.top, 1)
                    dbottom = notehead.bottom - closest_ll.bottom
                else:
                    raise ValueError('Strange notehead {0} vs. ledger line {1}'
                                     ' situation: bbox notehead {2}, LL {3}'
                                     ''.format(notehead.uid, closest_ll.uid,
                                               notehead.bounding_box,
                                               closest_ll.bounding_box))

                if min(dtop, dbottom) / max(dtop, dbottom) \
                        < PitchInferenceEngineConstants.ON_STAFFLINE_RATIO_TRHESHOLD:
                    _on_ledger_line = False

                    # Check orientation congruent with rel. to staff.
                    # If it is wrong (e.g., notehead mostly under LL
                    # but above staffline, and looks like off-LL),
                    # change back to on-LL.
                    if (dtop > dbottom) and not is_above_staff:
                        _on_ledger_line = True
                        logging.debug('Notehead in LL space with wrong orientation '
                                      'w.r.t. staff:'
                                      ' {0}'.format(notehead.uid))
                    if (dbottom > dtop) and is_above_staff:
                        _on_ledger_line = True
                        logging.debug('Notehead in LL space with wrong orientation '
                                      'w.r.t. staff:'
                                      ' {0}'.format(notehead.uid))

            delta = (2 * n_lls - 1) + 5
            if not _on_ledger_line:
                delta += 1

            if not is_above_staff:
                delta *= -1

            return delta

        elif len(staffline_objects) == 1:
            current_staffline = staffline_objects[0]

            # Count how far from the current staffline we are.
            #  - Collect staffline objects from the current staff
            all_staffline_objects = self.__children(current_staff,
                                                    self._CONST.STAFFLINE_CROPOBJECT_CLSNAMES)

            #  - Determine their ordering, top to bottom
            sorted_staffline_objects = sorted(all_staffline_objects,
                                              key=lambda x: (x.top + x.bottom) / 2.)

            delta = None
            for i, s in enumerate(sorted_staffline_objects):
                if s.objid == current_staffline.objid:
                    delta = 5 - i

            if delta is None:
                raise ValueError('Notehead {0} attached to staffline {1},'
                                 ' which is however not a child of'
                                 ' the notehead\'s staff {2}!'
                                 ''.format(notehead.uid, current_staffline.uid,
                                           current_staff.uid))

            return delta

        else:
            raise ValueError('Notehead {0} attached to more than one'
                             ' staffline/staffspace!'.format(notehead.uid))

    def process_measure_separator(self, measure_separator):
        self.pitch_state.reset_inline_accidentals()

    def process_key_signature(self, key_signature):
        sharps = self.__children(key_signature, ['sharp'])
        flats =  self.__children(key_signature, ['flat'])
        self.pitch_state.set_key(len(sharps), len(flats))

    def process_clef(self, clef):
        self.pitch_state.init_base_pitch(clef=clef)

    def _collect_symbols_for_pitch_inference(self, cropobjects):
        """Extract all symbols from the document relevant for pitch
        inference and index them in the Engine's temp data structures."""
        # Collect staves.
        self.staves = [c for c in cropobjects if c.clsname == 'staff']
        logging.info('We have {0} staves.'.format(len(self.staves)))

        # Collect clefs and key signatures per staff.
        self.clefs = [c for c in cropobjects
                      if c.clsname in self._CONST.CLEF_CLSNAMES]
        self.key_signatures = [c for c in cropobjects
                          if c.clsname == 'key_signature']

        self.clef_to_staff_map = {}
        # There may be more than one clef per staff.
        self.staff_to_clef_map = collections.defaultdict(list)
        for c in self.clefs:
            # Assuming one staff per clef
            s = self.__children(c, ['staff'])[0]
            self.clef_to_staff_map[c.objid] = s
            self.staff_to_clef_map[s.objid].append(c)

        self.key_to_staff_map = {}
        # There may be more than one key signature per staff.
        self.staff_to_key_map = collections.defaultdict(list)
        for k in self.key_signatures:
            s = self.__children(k, ['staff'])[0]
            self.key_to_staff_map[k.objid] = s
            self.staff_to_key_map[s.objid].append(k)

        # Collect measure separators.
        self.measure_separators = [c for c in cropobjects
                              if c.clsname == 'measure_separator']
        self.staff_to_msep_map = collections.defaultdict(list)
        for m in self.measure_separators:
            _m_staves = self.__children(m, ['staff'])
            # (Measure separators might belong to multiple staves.)
            for s in _m_staves:
                self.staff_to_msep_map[s.objid].append(m)
                # Collect accidentals per notehead.

        # Collect noteheads.
        self.noteheads = [c for c in cropobjects
                          if c.clsname in self._CONST.NOTEHEAD_CLSNAMES]
        self.staff_to_noteheads_map = collections.defaultdict(list)
        for n in self.noteheads:
            s = self.__children(n, ['staff'])[0]
            self.staff_to_noteheads_map[s.objid].append(n)

    def __children(self, c, clsnames):
        """Retrieve the children of the given CropObject ``c``
        that have class in ``clsnames``."""
        return [self._cdict[o] for o in c.outlinks
                if self._cdict[o].clsname in clsnames]

    def __parents(self, c, clsnames):
        """Retrieve the parents of the given CropObject ``c``
        that have class in ``clsnames``."""
        return [self._cdict[i] for i in c.inlinks
                if self._cdict[i].clsname in clsnames]

    ##########################################################################
    # Durations inference

    def beats(self, cropobject):
        if cropobject.clsname in self._CONST.NOTEHEAD_CLSNAMES:
            return self.notehead_beats(cropobject)
        elif cropobject.clsname in self._CONST.REST_CLSNAMES:
            return self.rest_beats(cropobject)
        else:
            raise ValueError('Cannot compute beats for object {0} of class {1};'
                             ' beats only available for notes and rests.'
                             ''.format(cropobject.uid, cropobject.clsname))

    def notehead_beats(self, notehead):
        """Retrieves the duration for the given notehead, in beats.

        It is possible that the notehead has two stems.
        In that case, we return all the possible durations:
        usually at most two, but if there is a duration dot, then
        there can be up to 4 possibilities.

        Grace notes currently return 0 beats.

        :returns: A list of possible durations for the given notehead.
            Mostly its length is just 1; for multi-stem noteheads,
            you might get more.
        """
        beat = [0]

        stems = self.__children(notehead, ['stem'])
        flags_and_beams = self.__children(
            notehead,
            PitchInferenceEngineConstants.FLAGS_AND_BEAMS)

        if notehead.clsname.startswith('grace-notehead'):
            logging.warn('Notehead {0}: Grace notes get zero duration!'
                         ''.format(notehead.uid))
            beat = [0]

        elif len(stems) > 1:
            logging.warn('Inferring duration for multi-stem notehead: {0}'
                         ''.format(notehead.uid))
            raise NotImplementedError()

        elif len(stems) == 0:
            if notehead.clsname == 'notehead-full':
                raise ValueError('Full notehead {0} has no stem!'.format(notehead.uid))
            beat = [4]

        elif notehead.clsname == 'notehead-empty':
            if len(flags_and_beams) != 0:
                raise ValueError('Notehead {0} is empty, but has {1} flags and beams!'
                                 ''.format(notehead.uid))
            beat = [2]

        elif notehead.clsname == 'notehead-full':
            beat = [0.5**len(flags_and_beams)]

        else:
            raise ValueError('Notehead {0}: unknown clsname {1}'
                             ''.format(notehead.uid, notehead.clsname))

        duration_modifier = self.compute_duration_modifier(notehead)

        beat = [b * duration_modifier for b in beat]

        return beat

    def compute_duration_modifier(self, notehead):
        """Computes the duration modifier (multiplicative, in beats)
        for the given notehead (or rest) from the tuples and duration dots.

        Can handle duration dots within tuples.
        """
        duration_modifier = 1
        # Dealing with tuples:
        tuples = self.__children(notehead, ['tuple'])
        if len(tuples) > 1:
            raise ValueError('Notehead {0}: Cannot deal with more than one tuple'
                             ' simultaneously.'.format(notehead.uid))
        if len(tuples) == 1:
            tuple = tuples[0]

            # Find the number in the tuple.
            numerals = sorted([self._cdict[o] for o in tuple.outlinks
                               if o.clsname.startswith('numeral')],
                              key=lambda x: x.left)
            tuple_number = int(''.join([num[-1] for num in numerals]))

            if tuple_number == 2:
                # Duola makes notes *longer*
                duration_modifier = 3 / 2
            elif tuple_number == 3:
                duration_modifier = 2 / 3
            elif tuple_number == 4:
                # This one also makes notes longer
                duration_modifier = 4 / 3
            elif tuple_number == 5:
                duration_modifier = 4 / 5
            elif tuple_number == 6:
                # Most often done for two consecutive triolas,
                # e.g. 16ths with a 6-tuple filling one beat
                duration_modifier = 3 / 2
            elif tuple_number == 7:
                # Here we get into trouble, because this one
                # can be both 4 / 7 (7 16th in a beat)
                # or 8 / 7 (7 32nds in a beat).
                # In the same vein, we cannot resolve higher
                # tuples unless we establish precedence/simultaneity.
                logging.warn('Cannot really deal with higher tuples than 6.')
                # For MUSCIMA++ specifically, we can cheat: there is only one
                # septuple, which consists of 7 x 32rd in 1 beat, so they
                # get 8 / 7.
                logging.warn('MUSCIMA++ cheat: we know there is only 7 x 32rd in 1 beat'
                             ' in page 14.')
                duration_modifier = 8 / 7
            elif tuple_number == 10:
                logging.warn('MUSCIMA++ cheat: we know there is only 10 x 32rd in 1 beat'
                             ' in page 04.')
                duration_modifier = 4 / 5
            else:
                raise NotImplementedError('Notehead {0}: Cannot deal with tuple '
                                          'number {1}'.format(notehead.uid,
                                                              tuple_number))

        # Duration dots
        ddots = self.__children(notehead, ['duration-dot'])
        dot_duration_modifier = 1
        for i, d in enumerate(ddots):
            dot_duration_modifier += 1 / (2 ** (i + 1))
        duration_modifier *= dot_duration_modifier

        return duration_modifier

    def rest_beats(self, rest):
        rest_beats_dict = {'whole_rest': 4,   # !!! We should find the TS.
                           'half_rest': 2,
                           'quarter_rest': 1,
                           '8th_rest': 0.5,
                           '16th_rest': 0.25,
                           '32th_rest': 0.125,
                           '64th_and_higher_rest': 0.0625,
                           # Technically, these two should just apply time sig.,
                           # but the measure-factorized precedence graph
                           # means these durations never have descendants anyway.
                           'multi-measure_rest': 4,
                           'repeat-measure': 4,
                           }

        try:
            base_rest_duration = rest_beats_dict[rest.clsname]
        except KeyError:
            raise KeyError('Symbol {0}: Unknown rest type {1}!'
                           ''.format(rest.uid, rest.clsname))

        duration_modifier = self.compute_duration_modifier(rest)

        return [base_rest_duration * duration_modifier]

    ##########################################################################
    # Onsets inference

    def infer_precedence(self, cropobjects):

        if not self.measure_separators:
            self._collect_symbols_for_pitch_inference(cropobjects)

        ######################################################################
        # An important feature of measure-factorized onset inference
        # instead of going left-to-right per part throughout is resistance
        # to staves appearing & disappearing on line breaks (e.g. orchestral
        # scores). Measures are (very, very often) points of synchronization
        #  -- after all, that is their purpose.

        # We currently DO NOT aim to process renaissance & medieval scores:
        # especially motets may often have de-synchronized measure separators.

        # Add the relationships between the measure separator nodes.
        #  - Get staves to which the mseps are connected
        msep_staffs = {m.objid: self.__children(m, ['staff'])
                       for m in self.measure_separators}
        #  - Sort first by bottom-most staff to which the msep is connected
        #    to get systems
        #  - Sort left-to-right within systems to get final ordering of mseps
        ordered_mseps = sorted(self.measure_separators,
                               key=lambda m: (max([s.bottom
                                                   for s in msep_staffs[m.objid]]),
                                              m.left))
        ordered_msep_nodes = [PrecedenceGraphNode(cropobject=m,
                                                  inlinks=[],
                                                  outlinks=[],
                                                  onset=None,
                                                  duration=0)
                              for m in ordered_mseps]

        # Add root node: like measure separator, but for the first measure.
        # This one is the only one which is initialized with onset,
        # with the value onset=0.
        root_msep = PrecedenceGraphNode(objid=-1,
                                        cropobject=None,
                                        inlinks=[], outlinks=[],
                                        duration=0,
                                        onset=0)

        # Create measure bins. i-th measure ENDS at i-th ordered msep.
        # We assume that every measure has a rightward separator.
        measures = [(None, ordered_mseps[0])] + [(ordered_mseps[i], ordered_mseps[i+1])
                                                 for i in range(len(ordered_mseps) - 1)]
        measure_nodes = [PrecedenceGraphNode(objid=None,
                                             cropobject=None,
                                             inlinks=[root_msep],
                                             outlinks=[ordered_msep_nodes[0]],
                                             duration=0,  # Durations will be filled in
                                             onset=None)] + \
                        [PrecedenceGraphNode(objid=None,
                                             cropobject=None,
                                             inlinks=[ordered_msep_nodes[i+1]],
                                             outlinks=[ordered_msep_nodes[i+2]],
                                             duration=0,  # Durations will be filled in
                                             onset=None)
                         for i in range(len(ordered_msep_nodes) - 2)]
        '''A list of PrecedenceGraph nodes. These don't really need any CropObject
        or objid, they are just introducing through their duration the offsets
        between measure separators (mseps have legit 0 duration, so that they
        do not move the notes in their note descendants).
        The list is already ordered.'''

        # Add measure separator inlinks and outlinks.
        for m_node in measure_nodes:
            r_sep = m_node.outlinks[0]
            r_sep.inliks.append(m_node)
            if len(m_node.inlinks) > 0:
                l_sep = m_node.inlinks[0]
                l_sep.outlinks.append(m_node)

        # Finally, hang the first measure on the root msep node.
        root_msep.outlinks.append(measure_nodes[0])

        ######################################################################
        # Now, compute measure node durations from time signatures.
        #  This is slightly non-trivial. Normally, a time signature is
        #  (a) at the start of the staff, (b) right before the msep starting
        #  the measure to which it should apply. However, sometimes the msep
        #  comes up (c) at the *start* of the measure to which it should
        #  apply. We IGNORE option (c) for now.
        #
        #  - Collect all time signatures
        time_signatures = [c for c in cropobjects
                           if c.clsname in self._CONST.TIME_SIGNATURES]

        #  - Assign time signatures to measure separators that *end*
        #    the bars. (Because as opposed to the starting mseps,
        #    the end mseps are (a) always there, (b) usually on the
        #    same staff, (c) if not on the same staff, then they are
        #    an anticipation at the end of a system, and will be repeated
        #    at the beginning of the next one anyway.)
        time_signatures_to_first_measure = {}
        for t in time_signatures:
            s = self.__children(t, ['staff'])[0]
            # - Find the measure pairs
            for i, (left_msep, right_msep) in enumerate(measures):
                if s not in msep_staffs[right_msep.objid]:
                    continue
                if (left_msep is None) or (s not in msep_staffs[left_msep.objid]):
                    # Beginning of system, valid already for the current bar.
                    time_signatures_to_first_measure[t.objid] = i
                else:
                    # Use i + 1, because the time signature is valid
                    # for the *next* measure.
                    time_signatures_to_first_measure[t.objid] = i + 1

        # - Interpret time signatures.
        time_signature_durations = {t.objid: self.interpret_time_signature(t)
                                    for t in time_signatures}

        # - Reverse map: for each measure, the time signature valid
        #   for the measure.
        measure_to_time_signature = [None for _ in measures]
        time_signatures_sorted = sorted(time_signatures,
                                        key=lambda x: time_signatures_to_first_measure[x.objid])
        for t1, t2 in zip(time_signatures_sorted[:-1], time_signatures_sorted[1:]):
            affected_measures = range(time_signatures_to_first_measure[t1.objid],
                                      time_signatures_to_first_measure[t2.objid])
            for i in affected_measures:
                # Check for conflicting time signatures previously
                # assigned to this measure.
                if measure_to_time_signature[i] is not None:
                    _competing_time_sig = measure_to_time_signature[i]
                    if (time_signature_durations[t1.objid] !=
                            time_signature_durations[_competing_time_sig.objid]):
                        raise ValueError('Trying to overwrite time signature to measure'
                                         ' assignment at measure {0}: new time sig'
                                         ' {1} with value {2}, previous time sig {3}'
                                         ' with value {4}'
                                         ''.format(i, t1.uid,
                                                   time_signature_durations[t1.objid],
                                                   _competing_time_sig.uid,
                                                   time_signature_durations[_competing_time_sig.objid]))

                measure_to_time_signature[i] = t1

        logging.debug('Checking that every measure has a time signature assigned.')
        for i, (msep1, msep2) in enumerate(measures):
            if measure_to_time_signature[i] is None:
                raise ValueError('Measure without time signature: {0}, between'
                                 'separators {1} and {2}'
                                 ''.format(i, msep1.uid, msep2.uid))

        # - Apply to each measure node the duration corresponding
        #   to its time signature.
        for i, m in enumerate(measure_nodes):
            _tsig = measure_to_time_signature[i]
            m.duration = time_signature_durations[_tsig.objid]

        # ...
        # Now, the "skeleton" of the precedence graph consisting
        # pf measure separator and measure nodes is complete.
        ######################################################################

        ######################################################################
        # Collecting onset-carrying objects (at this point, noteheads
        # and rests; the repeat-measure object that would normally
        # affect duration is handled through measure node durations.
        onset_objs = [c for c in cropobjects
                      if c.clsname in self._CONST.clsnames_bearing_duration]

        # Assign onset-carrying objects to measures (their left msep).
        # (This is *not* done by assigning outlinks to measure nodes,
        # we are now just factorizing the space of possible precedence
        # graphs.)
        #  - This is done by iterating over staves.
        staff_to_objs_map = collections.defaultdict(list)
        for c in onset_objs:
            ss = self.__children(c, ['staff'])
            for s in ss:
                staff_to_objs_map[s.objid].append(c)

        #  - Noteheads and rests are all connected to staves,
        #    which immediately gives us for each staff the subset
        #    of eligible symbols for each measure.
        #  - We can just take the vertical projection of each onset
        #    object and find out which measures it overlaps with.
        #    To speed this up, we can just check whether the middles
        #    of objects fall to the region delimited by the measure
        #    separators. Note that sometimes the barlines making up
        #    the measure separator are heavily bent, so it would
        #    be prudent to perhaps use just the intersection of
        #    the given barline and the current staff.

        # Preparation: we need for each valid (staff, msep) combination
        # the bounding box of their intersection, in order to deal with
        # more curved measure separators.

        msep_to_staff_projections = {}
        '''For each measure separator, for each staff it connects to,
        the bounding box of the measure separator's intersection with
        that staff.'''
        for msep in self.measure_separators:
            msep_to_staff_projections[msep.objid] = {}
            for s in msep_staffs[msep.objid]:
                intersection_bbox = self.msep_staff_overlap_bbox(msep, s)
                msep_to_staff_projections[msep.objid][s.objid] = intersection_bbox

        staff_and_measure_to_objs_map = collections.defaultdict(
                                            collections.defaultdict(list))
        '''Per staff (indexed by objid) and measure (by order no.), keeps a list of
        CropObjects from that staff that fall within that measure.'''

        # Iterate over objects left to right, shift measure if next object
        # over bound of current measure.
        ordered_objs_per_staff = {s_objid: sorted(s_objs, key=lambda x: x.left)
                                  for s_objid, s_objs in staff_to_objs_map.items()}
        for s_objid, objs in ordered_objs_per_staff.items():
            # Vertically, we don't care -- the attachment to staff takes
            # care of that, we only need horizontal placement.
            _c_m_idx = 0   # Index of current measure
            _c_msep_right = measure_nodes[_c_m_idx].outlinks[0]
            # Left bound of current measure's right measure separator
            _c_m_right = msep_to_staff_projections[_c_msep_right.objid][s_objid][1]
            for _c_o_idx, o in objs:
                # If we are out of bounds, move to next measure
                while o.left > _c_m_right:
                    _c_m_idx += 1
                    if _c_m_idx >= len(measure_nodes):
                        raise ValueError('Object {0}: could not assign to any measure,'
                                         ' ran out of measures!'.format(o.objid))
                    _c_msep_right = measure_nodes[_c_m_idx].outlinks[0]
                    _c_m_right = msep_to_staff_projections[_c_msep_right.objid][s_objid][1]
                    staff_and_measure_to_objs_map[s_objid][_c_m_right] = []

                staff_and_measure_to_objs_map[s_objid][_c_m_right].append(o)

        # Infer precedence within the measure.
        #  - This is the difficult part.
        #  - First: check the *sum* of durations assigned to the measure
        #    against the time signature. If it fits only once, then it is
        #    a monophonic measure and we can happily read it left to right.
        #  - If the measure is polyphonic, the fun starts!
        #    With K graph nodes, how many prec. graphs are there?
        for s_objid in staff_and_measure_to_objs_map:
            for measure_idx in staff_and_measure_to_objs_map[s_objid]:
                _c_objs = staff_and_measure_to_objs_map[s_objid][measure_idx]
                measure_graph = self.measure_precedence_graph(_c_objs)

                # Connect the measure graph source nodes to their preceding
                # measure separator.
                l_msep_node = measure_nodes[measure_idx].inlinks[0]
                for source_node in measure_graph:
                    l_msep_node.outlinks.append(source_node)
                    source_node.inlinks.append(l_msep_node)

        return root_msep

    def measure_precedence_graph(self, cropobjects):
        """Indexed by staff objid and measure number, holds the precedence graph
        for the given measure in the given staff as a list of PrecedenceGraphNode
        objects that correspond to the source nodes of the precedence subgraph.
        These nodes then get connected to their leftwards measure separator node.

        :param cropobjects: List of CropObjects, assumed to be all from one
            measure.

        :returns: A list of PrecedenceGraphNode objects that correspond
            to the source nodes in the precedence graph for the (implied)
            measure. (In monophonic music, the list will have one element.)
            The rest of the measure precedence graph nodes is accessible
            through the sources' outlinks.

        """
        _is_monody = self.is_measure_monody(cropobjects)
        if _is_monody:
            source_nodes = self.monody_measure_precedence_graph(cropobjects)
            return source_nodes

        else:
            raise ValueError('Cannot deal with onsets in polyphonic music yet.')

    def monody_measure_precedence_graph(self, cropobjects):
        """Infers the precedence graph for a plain monodic measure.
        The resulting structure is very simple: it's just a chain
        of the onset-carrying objects from left to right."""
        nodes = []
        for c in sorted(cropobjects, key=lambda x: x.left):
            potential_durations = self.beats(c)

            # In monody, there should only be one duration
            if len(potential_durations) > 1:
                raise ValueError('Object {0}: More than one potential'
                                 ' duration, even though the measure is'
                                 ' determined to be monody.'.format(c.uid))
            duration = potential_durations[0]

            node = PrecedenceGraphNode(objid=c.objid,
                                       cropobject=c,
                                       inlinks=[],
                                       outlinks=[],
                                       duration=duration,
                                       onset=None)
            nodes.append(node)
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            n1.outlinks.append(n2)
            n2.inlinks.append(n1)
        source_nodes = [nodes[0]]
        return source_nodes

    def is_measure_monody(self, cropobjects):
        """Checks whether the given measure is written as simple monody:
        no two of the onset-carrying objects are active simultaneously.

        Assumptions
        -----------

        * Detecting monody without looking at the time signature:
            * All stems in the same direction? --> NOPE: Violin chords in Bach...
            * All stems in horizontally overlapping noteheads in the same direction?
              --> NOPE: Again, violin chords in Bach...
            * Overlapping noteheads share a beam, but not a stem? --> this works,
              but has false negatives: overlapping quarter notes
        """
        raise NotImplementedError()

    def is_measure_chord_monody(self, cropobjects):
        """Checks whether the given measure is written as monody potentially
        with chords. That is: same as monody, but once all onset-carrying objects
        that share a stem are merged into an equivalence class."""
        raise NotImplementedError()

    def msep_staff_overlap_bbox(self, measure_separator, staff):
        """Computes the bounding box for the part of the input
        ``measure_separator`` that actually overlaps the ``staff``.
        This is implemented to deal with mseps that curve a lot,
        so that their left/right bounding box may mistakenly
        exclude some symbols from their preceding/following measure.

        Returns the (T, L, B, R) bounding box.
        """
        intersection = measure_separator.bbox_intersection(staff)
        if intersection is None:
            # Corner case: measure separator is connected to staff,
            # but its bounding box does *not* overlap the bbox
            # of the staff.
            output_bbox = staff.top, measure_separator.left, \
                          staff.bottom, measure_separator.right
        else:
            # The key step: instead of using the bounding
            # box intersection, first crop the zeros from
            # msep intersection mask (well, find out how
            # many left and right zeros there are).
            it, il, ib, ir = intersection
            msep_crop = measure_separator.mask[it, il, ib, ir]

            if msep_crop.sum() == 0:
                # Corner case: bounding box does encompass staff,
                # but there is msep foreground pixel in that area
                # (could happen e.g. with mseps only drawn *around*
                # staffs).
                output_bbox = staff.top, measure_separator.left, \
                              staff.bottom, measure_separator.right
            else:
                # The canonical case: measure separator across the staff.
                msep_crop_vproj = msep_crop.sum(axis=0)
                _dl = 0
                _dr = 0
                for i, v in enumerate(msep_crop_vproj):
                    if v != 0:
                        _dl = i
                        break
                for i in range(1, len(msep_crop_vproj)):
                    if msep_crop_vproj[-i] != 0:
                        _dr = i
                        break
                output_bbox = staff.top, measure_separator.left + _dl, \
                              staff.bottom, measure_separator.right - _dr
        return output_bbox

    def interpret_time_signature(self, time_signature):
        """Converts the time signature into the beat count (in quarter
        notes) it assigns to its following measures.

        Dealing with numeric time signatures
        ------------------------------------

        * Is there both a numerator and a denominator?
        * If yes: assign numerals to either num. (top), or denom. (bottom)
        * If not: assume the number is no. of beats. (In some scores, the
          base indicator may be attached in form of a note instead of a
          denumerator, like e.g. scores by Janacek, but we ignore this for now.
          In early music, 3 can mean "tripla", which is 3/2.)

        Dealing with non-numeric time signatures
        ----------------------------------------

        * whole-time mark is interpreted as 4/4
        * alla breve mark is interpreted as 2/4

        """
        raise NotImplementedError()

    def onsets(self, cropobjects):
        """Infers the onsets of notes in the given cropobjects.

        The onsets are measured in beats.

        :returns: A objid --> onset dict for all notehead-type
            CropObjects.
        """

        # TODO: Don't infer_pitches(), remove dep. on self.measure_separators
        # Technicality, shortcut, to fill up all the internal dicts.
        # This does not take long.
        self.infer_pitches(cropobjects, with_names=True)

        # We first find the precedence graph. (This is the hard
        # part.)
        # The precedence graph is a dict of CropObjects
        # that have additional prec_inlinks and prec_outlinks
        # attributes.
        #
        # Note that measure separators should participate in the precedence
        # graph...
        precedence_graph = self.infer_precedence(cropobjects)

        # Once we have the precedence graph, we need to walk it.
        # It is a DAG, so we simply do a BFS from each source.
        # Whenever a node has more incoming predecessors,
        # we need to wait until they are *all* resolved,
        # and check whether they agree.
        queue = []
        for node in precedence_graph:
            if len(node.inlinks) == 0:
                queue.append(node)

        onsets = {}

        # We will only be appending to the queue, so the
        # start of the queue is defined simply by the index.
        __qstart = 0
        while len(queue) > 0:
            q = queue[__qstart]
            __qstart += 1
            prec_qs = q.inlinks
            prec_onsets = [pq.onset for pq in prec_qs]
            prec_durations = [pq.duration for pq in prec_qs]

            onset_proposals = [o + d for o, d in zip(prec_onsets, prec_durations)]
            if min(onset_proposals) != max(onset_proposals):
                raise ValueError('Object {0}: onsets not synchronized from'
                                 ' predecessors: {1}'.format(q.obj.uid,
                                                             onset_proposals))
            onset = onset_proposals[0]
            q.onset = onset
            # Some nodes do not have a CropObject assigned.
            if q.obj is not None:
                onsets[q.obj.objid] = onset

            for post_q in q.outlinks:
                queue.append(post_q)

            __qstart += 1

        return onsets


class PrecedenceGraphNode:
    """A helper plain-old-data class for onset extraction.
    The ``inlinks`` and ``outlinks`` attributes are lists
    of other ``PrecedenceGraphNode`` instances.
    """
    def __init__(self, objid=None, cropobject=None, inlinks=None, outlinks=None,
                 onset=None, duration=0):
        # Optional link to CropObjects, or just a placeholder ID.
        self.obj = cropobject
        if objid is None and cropobject is not None:
            objid = cropobject.objid
        self.node_id = objid

        self.inlinks = []
        if inlinks:
            self.inlinks = inlinks
        self.outlinks = []
        if outlinks:
            self.outlinks = outlinks

        self.onset = onset
        '''Counting from the start of the musical sequence, how many
        beat units pass before this object?'''

        self.duration = duration
        '''By how much musical time does the object delay the onsets
        of its descendants in the precedence graph?'''


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

    inference_engine = MIDIInferenceEngine()

    logging.info('Running pitch inference.')
    pitches, pitch_names = inference_engine.infer_pitches(cropobjects,
                                                          with_names=True)
    durations = inference_engine.durations_beats

    # Logging
    #pitch_names = {objid: midi2pitch_name(midi_code)
    #               for objid, midi_code in pitches.items()}

    # Export
    logging.info('Adding pitch information to <Data> attributes.')
    for c in cropobjects:
        if c.objid in pitches:
            midi_pitch_code = pitches[c.objid]
            pitch_step, pitch_octave = pitch_names[c.objid]
            beats = durations[c.objid]
            if len(beats) > 1:
                logging.warn('Notehead {0}: multiple possible beats: {1}'
                             ''.format(c.uid, beats))
                b = beats[0]
            else:
                b = beats[0]
            c.data = {'midi_pitch_code': midi_pitch_code,
                      'normalized_pitch_step': pitch_step,
                      'pitch_octave': pitch_octave,
                      'duration_beats': b}

    if args.export is not None:
        with open(args.export, 'w') as hdl:
            hdl.write(export_cropobject_list(cropobjects))
            hdl.write('\n')
    else:
        print(export_cropobject_list(cropobjects))

    _end_time = time.clock()
    logging.info('infer_pitches.py done in {0:.3f} s'.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
