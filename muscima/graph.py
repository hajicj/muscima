"""This module implements an abstraction over a notation graph, and
functions for manipulating notation graphs."""
from __future__ import print_function, unicode_literals

import copy

from muscima.cropobject import CropObject
from muscima.inference_engine_constants import _CONST

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class NotationGraphError(ValueError):
    pass


class NotationGraph(object):
    """The NotationGraph class is the abstraction for a notation graph."""

    def __init__(self, cropobjects):
        """Initialize the notation graph with a list of CropObjects."""
        self.cropobjects = cropobjects
        self._cdict = {c.objid: c for c in self.cropobjects}

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
            ch = self.children(current_objid, classes=classes)
            for o in ch:
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
            ch = self.parents(current_objid, classes=classes)
            for o in ch:
                if o not in queue:
                    queue.append(o)

        return [self._cdict[objid] for objid in ancestor_objids]

    def __getitem__(self, objid):
        """Returns a CropObject based on its objid."""
        return self._cdict[objid]


def group_staffs_into_systems(cropobjects):
    """Returns a list of lists of ``staff`` CropObjects
    grouped into systems. Uses the outer ``staff_grouping``
    symbols."""
    _cdict = {c.objid: c for c in cropobjects}
    staff_groups = [c for c in cropobjects
                    if c.clsname == 'staff_grouping']

    if len(staff_groups) != 0:
        staffs_per_group = {c.objid: [_cdict[i] for i in c.outlinks
                                      if _cdict[i].clsname == 'staff']
                            for c in staff_groups}
        # Build hierarchy of staff_grouping based on inclusion.
        outer_staff_groups = []
        for sg in staff_groups:
            sg_staffs = staffs_per_group[sg.objid]
            is_outer = True
            for other_sg in staff_groups:
                if sg.objid == other_sg.objid: continue
                other_sg_staffs = staffs_per_group[other_sg.objid]
                if len([s for s in sg_staffs
                        if s not in other_sg_staffs]) == 0:
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
        systems = [[c] for c in cropobjects if c.clsname == 'staff']
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

