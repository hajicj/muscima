"""This module implements an abstraction over a notation graph, and
functions for manipulating notation graphs."""
from __future__ import print_function, unicode_literals

import copy
import logging

import operator

from muscima.cropobject import CropObject
from muscima.inference_engine_constants import _CONST
from muscima.utils import resolve_notehead_wrt_staffline

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class NotationGraphError(ValueError):
    pass


class NotationGraphUnsupportedError(NotImplementedError):
    pass


class NotationGraph(object):
    """The NotationGraph class is the abstraction for a notation graph."""

    def __init__(self, cropobjects):
        """Initialize the notation graph with a list of CropObjects."""
        self.cropobjects = cropobjects
        self._cdict = {c.objid: c for c in self.cropobjects}

    def __len__(self):
        return len(self.cropobjects)

    def __to_objid(self, cropobject_or_objid):
        if isinstance(cropobject_or_objid, CropObject):
            objid = cropobject_or_objid.objid
        else:
            objid = cropobject_or_objid
        return objid

    def children(self, cropobject_or_objid, classes=None):
        """Find all children of the given node."""
        objid = self.__to_objid(cropobject_or_objid)
        if objid not in self._cdict:
            raise ValueError('CropObject {0} not in graph!'.format(self._cdict[objid].uid))

        c = self._cdict[objid]
        output = []
        for o in c.outlinks:
            if o in self._cdict:
                if classes is None:
                    output.append(self._cdict[o])
                elif self._cdict[o].clsname in classes:
                    output.append(self._cdict[o])
        return output

    def parents(self, cropobject_or_objid, classes=None):
        """Find all parents of the given node."""
        objid = self.__to_objid(cropobject_or_objid)
        if objid not in self._cdict:
            raise ValueError('CropObject {0} not in graph!'.format(self._cdict[objid].uid))

        c = self._cdict[objid]
        output = []
        for i in c.inlinks:
            if i in self._cdict:
                if classes is None:
                    output.append(self._cdict[i])
                elif self._cdict[i].clsname in classes:
                    output.append(self._cdict[i])
        return output

    def descendants(self, cropobject_or_objid, classes=None):
        """Find all descendants of the given node."""
        objid = self.__to_objid(cropobject_or_objid)

        descendant_objids = []
        queue = [objid]
        __q_start = 0
        while __q_start < len(queue):
            current_objid = queue[__q_start]
            __q_start += 1

            if current_objid != objid:
                descendant_objids.append(current_objid)
            children = self.children(current_objid, classes=classes)
            children_objids = [ch.objid for ch in children]
            for o in children_objids:
                if o not in queue:
                    queue.append(o)

        return [self._cdict[o] for o in descendant_objids]

    def ancestors(self, cropobject_or_objid, classes=None):
        """Find all ancestors of the given node."""
        objid = self.__to_objid(cropobject_or_objid)

        ancestor_objids = []
        queue = [objid]
        __q_start = 0
        while __q_start < len(queue):
            current_objid = queue[__q_start]
            __q_start += 1

            if current_objid != objid:
                ancestor_objids.append(current_objid)
            parents = self.parents(current_objid, classes=classes)
            parent_objids = [p.objid for p in parents]
            for o in parent_objids:
                if o not in queue:
                    queue.append(o)

        return [self._cdict[objid] for objid in ancestor_objids]

    def has_child(self, cropobject_or_objid, classes=None):
        children = self.children(cropobject_or_objid, classes=classes)
        return len(children) > 0

    def has_parent(self, cropobject_or_objid, classes=None):
        parents = self.parents(cropobject_or_objid, classes=classes)
        return len(parents) > 0

    def is_child_of(self, cropobject_or_objid, other_cropobject_or_objid):
        """Check whether the first symbol is a child of the second symbol."""
        to_objid = self.__to_objid(cropobject_or_objid)
        from_objid = self.__to_objid(other_cropobject_or_objid)

        c = self._cdict[from_objid]
        if to_objid in c.outlinks:
            return True
        else:
            return False

    def is_parent_of(self, cropobject_or_objid, other_cropobject_or_objid):
        """Check whether the first symbol is a parent of the second symbol."""
        from_objid = self.__to_objid(cropobject_or_objid)
        to_objid = self.__to_objid(other_cropobject_or_objid)

        c = self._cdict[from_objid]
        if to_objid in c.outlinks:
            return True
        else:
            return False

    def __getitem__(self, objid):
        """Returns a CropObject based on its objid."""
        return self._cdict[objid]

    def is_stem_direction_above(self, notehead, stem):
        """Determines whether the given stem of the given notehead
        is above it or below. This is not trivial due to chords.
        """
        if notehead.objid not in self._cdict:
            raise NotationGraphError('Asking for notehead which is not'
                                     ' in graph: {0}'.format(notehead.uid))

        # This works even if there is just one. There should always be one.
        sibling_noteheads = self.parents(stem, classes=_CONST.NOTEHEAD_CLSNAMES)
        if notehead not in sibling_noteheads:
            raise ValueError('Asked for stem direction, but notehead {0} is'
                             ' unrelated to given stem {1}!'
                             ''.format(notehead.uid, stem.uid))

        topmost_notehead = min(sibling_noteheads, key=lambda x: x.top)
        bottom_notehead = max(sibling_noteheads, key=lambda x: x.bottom)

        d_top = topmost_notehead.top - stem.top
        d_bottom = stem.bottom - bottom_notehead.bottom

        return d_top > d_bottom

    def is_symbol_above_notehead(self, notehead, other, compare_on_intersect=False):
        """Determines whether the given other symbol is above
        the given notehead.

        This is non-trivial because the other may reach above *and* below
        the given notehead, if it is long and slanted (beam, slur, ...).
        A horizontally intersecting subset of the mask of the other symbol
        is used to determine its vertical bounds relevant to the given object.
        """
        # if other not in self.children(notehead, [other.clsname]):
        #     raise NotationGraphUnsupportedError('Resolving other direction:'
        #                                         ' assumes other {0} child of'
        #                                         ' notehead {1}.'
        #                                         ''.format(other.uid,
        #                                                   notehead.uid))

        if notehead.right <= other.left:
            # No horizontal overlap, notehead to the left
            beam_submask = other.mask[:, :1]
        elif notehead.left >= other.right:
            # No horizontal overlap, notehead to the right
            beam_submask = other.mask[:, -1:]
        else:
            h_bounds = (max(notehead.left, other.left),
                        min(notehead.right, other.right))

            beam_submask = other.mask[:,
                           (h_bounds[0] - other.left):(h_bounds[1] - other.left)]

        # Get vertical bounds of beam submask
        other_submask_hsum = beam_submask.sum(axis=1)
        other_submask_top = min([i for i in range(beam_submask.shape[0])
                                if other_submask_hsum[i] != 0]) + other.top
        other_submask_bottom = max([i for i in range(beam_submask.shape[0])
                                   if other_submask_hsum[i] != 0]) + other.top
        if (notehead.top <= other_submask_top <= notehead.bottom) \
                or (other_submask_bottom <= notehead.top <= other_submask_bottom):
            if compare_on_intersect:
                logging.warn('Notehead {0} intersecting other.'
                             ' Returning false.'
                             ''.format(notehead.uid))
                return False

        if notehead.bottom < other_submask_top:
            return False

        elif notehead.top > other_submask_bottom:
            return True

        else:
            raise NotationGraphError('Weird relative position of notehead'
                                     ' {0} and other {1}.'.format(notehead.uid,
                                                                 other.uid))

    def remove_vertex(self, objid):

        self.remove_edges_for_vertex(objid)

        c = self._cdict[objid]
        self.cropobjects.remove(c)
        del self._cdict[objid]

    def remove_edge(self, fr, to):
        if fr not in self._cdict:
            raise ValueError('Cannot remove edge from objid {0}: not in graph!'
                             ''.format(fr))
        if to not in self._cdict:
            raise ValueError('Cannot remove edge to objid {0}: not in graph!'
                             ''.format(to))

        # print('removing edge {0}'.format((fr, to)))
        f = self._cdict[fr]
        # print('\tf outlinks before: {0}'.format(f.outlinks))
        f.outlinks.remove(to)
        # print('\tf outlinks after: {0}'.format(f.outlinks))
        t = self._cdict[to]
        # print('\tt outlinks before: {0}'.format(t.outlinks))
        t.inlinks.remove(fr)
        # print('\tt outlinks after: {0}'.format(t.outlinks))

    def remove_edges_for_vertex(self, objid):
        if objid not in self._cdict:
            raise ValueError('Cannot remove vertex with objid {0}: not in graph!'
                             ''.format(objid))
        c = self._cdict[objid]

        # print('Removing edges for vertex: {0}.'.format(objid))
        # print('\tInlinks: {0}'.format(c.inlinks))
        # print('\tOutlinks: {0}'.format(c.outlinks))

        # Remove from inlinks and outlinks:
        for o in copy.deepcopy(c.outlinks):
            self.remove_edge(objid, o)
        for i in copy.deepcopy(c.inlinks):
            self.remove_edge(i, objid)

    def remove_classes(self, clsnames):
        """Remove all vertices with these clsnames."""
        to_remove = [c.objid for c in self.cropobjects if c.clsname in clsnames]
        for objid in to_remove:
            self.remove_vertex(objid)


def group_staffs_into_systems(cropobjects,
                              use_fallback_measure_separators=True):
    """Returns a list of lists of ``staff`` CropObjects
    grouped into systems. Uses the outer ``staff_grouping``
    symbols (or ``measure_separator``) symbols.

    Currently cannot deal with a situation where a system consists of
    interlocking staff groupings and measure separators, and cannot deal
    with system separator markings.

    :param cropobjects: The complete list of CropObjects in the current
        document.

    :param use_fallback_measure_separators: If set and no staff groupings
        are found, will use measure separators instead to group
        staffs. The algorithm is to find the leftmost measure
        separator for each staff and use this set instead of staff
        groupings: measure separators also have outlinks to all
        staffs that they are relevant for.

    :returns: A list of systems, where each system is a list of ``staff``
        CropObjects.
    """
    graph = NotationGraph(cropobjects)
    _cdict = {c.objid: c for c in cropobjects}
    staff_groups = [c for c in cropobjects
                    if c.clsname == 'staff_grouping']

    # Do not consider staffs that have no notehead or rest children.
    empty_staffs = [c for c in cropobjects if (c.clsname == 'staff') and
                    (len([i for i in c.inlinks
                          if ((_cdict[i].clsname in _CONST.NOTEHEAD_CLSNAMES) or
                              (_cdict[i].clsname in _CONST.REST_CLSNAMES))])
                     == 0)]
    print('Empty staffs: {0}'.format('\n'.join([c.uid for c in empty_staffs])))

    # There might also be non-empty staffs that are nevertheless
    # not covered by a staff grouping, only measure separators.

    if use_fallback_measure_separators: # and (len(staff_groups) == 0):
        # Collect measure separators, sort them left to right
        measure_separators = [c for c in cropobjects
                              if c.clsname in _CONST.MEASURE_SEPARATOR_CLSNAMES]
        measure_separators = sorted(measure_separators,
                                    key=operator.attrgetter('left'))
        # Use only the leftmost measure separator for each staff.
        staffs = [c for c in cropobjects
                  if c.clsname in [_CONST.STAFF_CLSNAME]]
        leftmost_measure_separators = set()
        for s in staffs:
            if s in empty_staffs:
                continue
            for m in measure_separators:
                if graph.is_child_of(s, m):
                    leftmost_measure_separators.add(m)
                    break
        staff_groups += leftmost_measure_separators

    if len(staff_groups) != 0:
        staffs_per_group = {c.objid: [_cdict[i] for i in sorted(c.outlinks)
                                      if _cdict[i].clsname == 'staff']
                            for c in staff_groups}
        # Build hierarchy of staff_grouping based on inclusion
        # between grouped staff sets.
        outer_staff_groups = []
        for sg in sorted(staff_groups, key=lambda c: c.left):
            sg_staffs = staffs_per_group[sg.objid]
            is_outer = True
            for other_sg in staff_groups:
                if sg.objid == other_sg.objid:
                    continue
                other_sg_staffs = staffs_per_group[other_sg.objid]
                if len([s for s in sg_staffs
                        if s not in other_sg_staffs]) == 0:
                    # If the staff groups are equal (can happen with
                    # a mixture of measure-separators and staff-groupings),
                    # only the leftmost should be an outer grouping.
                    if set(sg_staffs) == set(other_sg_staffs):
                        if other_sg in outer_staff_groups:
                            is_outer = False
                    else:
                        is_outer = False
            if is_outer:
                outer_staff_groups.append(sg)
        #
        # outer_staff_groups = [c for c in staff_groups
        #                       if len([_cdict[i] for i in c.inlinks
        #                               if _cdict[i].clsname == 'staff_group']) == 0]
        systems = [[c for c in cropobjects
                    if (c.clsname == 'staff') and (c.objid in sg.outlinks)]
                   for sg in outer_staff_groups]
    else:
        # Here we use the empty staff fallback
        systems = [[c] for c in cropobjects
                   if (c.clsname == 'staff') and (c not in empty_staffs)]

    return systems


def group_by_staff(cropobjects):
    """Returns one NotationGraph instance for each staff and its associated
    CropObjects. "Associated" means:

    * the object is a descendant of the staff,
    * the object is an ancestor of the staff, or
    * the object is a descendant of an ancestor of the staff, *except*
      measure separators and staff groupings.
    """
    g = NotationGraph(cropobjects=cropobjects)

    staffs = [c for c in cropobjects if c.clsname == _CONST.STAFF_CLSNAME]
    objects_per_staff = dict()
    for staff in staffs:
        descendants = g.descendants(staff)
        ancestors = g.ancestors(staff)
        a_descendants = []
        for ancestor in ancestors:
            if ancestor.clsname in _CONST.SYSTEM_LEVEL_CLSNAMES:
                continue
            _ad = g.descendants(ancestor)
            a_descendants.extend(_ad)
        staff_related = set()
        for c in descendants + ancestors + a_descendants:
            staff_related.add(c)

        objects_per_staff[staff.objid] = list(staff_related)

    return objects_per_staff


##############################################################################
# Graph search utilities

def find_related_staffs(query_cropobjects, all_cropobjects,
                        with_stafflines=True):
    """Find all staffs that are related to any of the cropobjects
    in question. Ignores whether these staffs are already within
    the list of ``query_cropobjects`` passed to the function.

    Finds all staffs that are ancestors or descendants of at least
    one of the query CropObjects, and if ``with_stafflines`` is requested,
    all stafflines and staffspaces that are descendants of at least one
    of the related staffs as well.

    :param query_cropobjects: A list of CropObjects for which we want
        to find related staffs. Subset of ``all_cropobjects``.

    :param all_cropobjects: A list of all the CropObjects in the document
        (or directly a NotationGraph object). Assumes that the query
        cropobjects are a subset of ``all_cropobjects``.

    :param with_stafflines: If set, will also return all stafflines
        and staffspaces related to the discovered staffs.

    :returns: List of staff (and, if requested, staffline/staffspace)
        CropObjects that are relate to the query CropObjects.
    """
    if not isinstance(all_cropobjects, NotationGraph):
        graph = NotationGraph(all_cropobjects)
    else:
        graph = all_cropobjects

    related_staffs = set()
    for c in query_cropobjects:
        desc_staffs = graph.descendants(c, classes=[_CONST.STAFF_CLSNAME])
        anc_staffs =  graph.ancestors(c, classes=[_CONST.STAFF_CLSNAME])
        current_staffs = set(desc_staffs + anc_staffs)
        related_staffs = related_staffs.union(current_staffs)

    if with_stafflines:
        related_stafflines = set()
        for s in related_staffs:
            staffline_objs = graph.descendants(s,
                                               _CONST.STAFFLINE_CROPOBJECT_CLSNAMES)
            related_stafflines = related_stafflines.union(set(staffline_objs))
        related_staffs = related_staffs.union(related_stafflines)

    return list(related_staffs)


##############################################################################
# Graph validation/fixing.
# An invariant of these methods should be that they never remove a correct
# edge. There is a known problem in this if a second stem is marked across
# staves: the beam orientation misfires.


def find_beams_incoherent_with_stems(cropobjects):
    """Searches the graph for edges where a notehead is connected to a stem
    in one direction, but is connected to beams that are in the
    other direction.

    If a notehead has zero or more than one stem, it is ignored.

    :returns: A list of (notehead, beam) pairs such that the beam
        is not coherent with the stem direction for the notehead.
    """
    graph = NotationGraph(cropobjects)
    noteheads = [c for c in cropobjects if c.clsname in _CONST.NOTEHEAD_CLSNAMES]

    incoherent_pairs = []
    for n in noteheads:
        stems = graph.children(n, classes=['stem'])
        if len(stems) != 1:
            continue
        stem = stems[0]

        beams = graph.children(n, classes=['beam'])
        if len(beams) == 0:
            continue

        # Is the stem above the notehead, or not?
        # This is not trivial because of chords.
        is_stem_above = graph.is_stem_direction_above(n, stem)
        logging.info('IncoherentBeams: stem of {0} is above'.format(n.objid))

        for b in beams:
            is_beam_above = graph.is_symbol_above_notehead(n, b)
            logging.info('IncoherentBeams: beam {0} of {1} is above'.format(b.objid, n.objid))
            if is_stem_above != is_beam_above:
                incoherent_pairs.append([n, b])

    return incoherent_pairs


# Ledger lines often cause problems with autoparser.
# They should be always linked from noteheads in a consistent
# direction (from outside inwards to the staff).
# Also, no notehead should be connected to both a staffline/staffspace
# *AND* a ledger line.

def find_ledger_lines_with_noteheads_from_both_directions(cropobjects):
    """Looks for ledger lines that have inlinks from noteheads
    on both sides. Returns a list of ledger line CropObjects."""
    graph = NotationGraph(cropobjects)

    problem_ledger_lines = []

    for c in cropobjects:
        if c.clsname != 'ledger_line':
            continue

        noteheads = graph.parents(c, classes=_CONST.NOTEHEAD_CLSNAMES)

        if len(noteheads) < 2:
            continue

        positions = [resolve_notehead_wrt_staffline(n, c) for n in noteheads]
        positions_not_on_staffline = [p for p in positions if p != 0]
        unique_positions = set(positions_not_on_staffline)
        if len(unique_positions) > 1:
            problem_ledger_lines.append(c)

    return problem_ledger_lines


def find_noteheads_with_ledger_line_and_staff_conflict(cropobjects):
    """Find all noteheads that have a relationship both to a staffline
    or staffspace and to a ledger line.

    Assumes (obviously) that staffline relationships have already been
    resolved. Useful in a workflow where autoparsing is applied *after*
    staff inference.
    """
    graph = NotationGraph(cropobjects)

    problem_noteheads = []

    for c in cropobjects:
        if c.clsname not in _CONST.NOTEHEAD_CLSNAMES:
            continue

        lls = graph.children(c, ['ledger_line'])
        staff_objs = graph.children(c, _CONST.STAFFLINE_CROPOBJECT_CLSNAMES)
        if lls and staff_objs:
            problem_noteheads.append(c)

    return problem_noteheads


def find_misdirected_ledger_line_edges(cropobjects):
    """Finds all edges that connect to ledger lines, but do not
    lead in the direction of the staff.

    Silently assumes that all noteheads are connected to the correct staff.
    """
    graph = NotationGraph(cropobjects)

    misdirected_object_pairs = []

    for c in cropobjects:
        if c.clsname not in _CONST.NOTEHEAD_CLSNAMES:
            continue

        lls = graph.children(c, ['ledger_line'])
        if not lls:
            continue

        staffs = graph.children(c, ['staff'])
        if not staffs:
            logging.warn('Notehead {0} not connected to any staff!'
                         ''.format(c.uid))
            continue
        staff = staffs[0]

        # Determine whether notehead is above or below staff.
        # Because of mistakes in notehead-ll edges, can actually be
        # *on* the staff. (If it is on a staffline, then the edge is
        # definitely wrong.)
        stafflines = sorted(graph.children(staff, [_CONST.STAFFLINE_CLSNAME]),
                            key=lambda x: x.top)
        p_top = resolve_notehead_wrt_staffline(c, stafflines[0])
        p_bottom = resolve_notehead_wrt_staffline(c, stafflines[-1])
        # Notehead actually located on the staff somewhere:
        # all of the LL rels. are false.
        if (p_top != p_bottom) or (p_top == 0) or (p_bottom == 0):
            for ll in lls:
                misdirected_object_pairs.append([c, ll])
            continue

        notehead_staff_direction = 1
        if p_bottom == -1:
            notehead_staff_direction = -1

        for ll in lls:
            ll_direction = resolve_notehead_wrt_staffline(c, ll)
            if (ll_direction != 0) and (ll_direction != notehead_staff_direction):
                misdirected_object_pairs.append([c, ll])

    return misdirected_object_pairs


def resolve_ledger_line_or_staffline_object(cropobjects):
    """If staff relationships are created before notehead to ledger line
    relationships, then there will be noteheads on ledger lines that
    are nevertheless connected to staffspaces. This function should be
    applied after both staffspace and ledger line relationships have been
    inferred, to guess whether the notehead's relationship to the staff
    object should be discarded.

    Has no dependence on misdirected edge detection (handles this as a part
    of the conflict resolution).
    """
    graph = NotationGraph(cropobjects)

    problem_object_pairs = []

    for c in cropobjects:
        if c.clsname not in _CONST.NOTEHEAD_CLSNAMES:
            continue

        lls = graph.children(c, ['ledger_line'])
        stafflines = graph.children(c, _CONST.STAFFLINE_CROPOBJECT_CLSNAMES)
        staff = graph.children(c, _CONST.STAFF_CLSNAME)

        if len(lls) == 0:
            continue
        if len(stafflines) == 0:
            continue

        if len(staff) == 0:
            logging.warn('Notehead {0} not connected to any staff!'
                         ' Unable to resolve ll/staffline.'.format(c.uid))
            continue

        # Multiple LLs: must check direction
        # Multiple stafflines: ???
        if len(stafflines) > 1:
            logging.warn('Notehead {0} is connected to multiple staffline'
                         ' objects!'.format(c.uid))


##############################################################################

def group_by_measure(cropobjects):
    """Groups the objects into measures.
    Assumes the measures are consistent across staffs: no polytempi.

    If there are objects that span multiple measures, they are assigned
    to all the measures they intersect.

    If no measure separators are found, assumes everything belongs
    to one measure.

    :returns: A list of CropObject lists corresponding to measures. The list
        is ordered left-to-right.
    """
    graph = NotationGraph(cropobjects)
    logging.debug('Find measure separators.')

    measure_separators = [c for c in cropobjects
                          if c.clsname in _CONST.MEASURE_SEPARATOR_CLSNAMES]

    if len(measure_separators) == 0:
        return cropobjects

    logging.debug('Order measure separators by precedence.')
    # Systems
    # measure seps. by system
    measure_separators = sorted(measure_separators, key=lambda m: m.left)

    logging.debug('Denote measure areas: bounding boxes and masks.')
    logging.debug('Assign objects to measures, based on overlap.')

    raise NotImplementedError()