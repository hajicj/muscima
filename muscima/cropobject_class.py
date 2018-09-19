"""This module implements the :class:`CropObjectClass`, which
represents one possible :class:`CropObject` class, such as
a notehead or a time signature. Aside from defining the "vocabulary"
of available object classes for annotation, it also contains
some information about how objects of the given class should
be displayed in the MUSCIMarker annotation software (ordering
related object classes together in menus, implementing a sensible
color scheme, etc.). There is nothing interesting about this class,
we pulled it into the ``muscima`` package because the object
grammar (i.e. which relationships are allowed and which are not)
depends on having CropObjectClass object as its "vocabulary",
and you will probably want to manipulate the data somehow based
on the objects' relationships (like reassembling notes from notation
primitives: notehead plus stem plus flags...), and the grammar
file is a reference for doing that.

CropObjectClass is a plain old data class, nothing interesting
about it. The only catch is that colors for rendering
in MUSCIMarker are kept as a ``#RRGGBB`` string in the XML
file, but represented in the ``CropObjectClass.color`` attribute
as a triplet of floats between 0 (``00``) and 255 (``ff``).


The ``___str__()`` method of the class will output the correct
XML representation.

**XML example**

This is what a single CropObjectClass element might look like::

    <CropObjectClass>
        <Id>1</Id>
        <Name>notehead-empty</Name>
        <GroupName>note-primitive/notehead-empty</GroupName>
        <Color>#FF7566</Color>
        </CropObjectClass>

See e.g. ``test/test_data/mff-muscima-classes-annot.xml``,
which is incidentally the real CropObjectClass list used
for annotating MUSCIMA++.

"""
import logging
from typing import Tuple

__version__ = "1.0"
__author__ = "Jan Hajic jr."


#######################################################################

class CropObjectClass(object):
    """Information about the annotation class. We're using it
    mostly to get the color of rendered CropObjects.

    CropObjectClass is a Plain Old Data class, there is no other
    functionality beyond simply existing and writing itself
    out in the appropriate XML format.
    """
    def __init__(self, clsid, name, group_name, color):
        # type: (int, str, str, str) -> None
        self.clsid = clsid
        self.name = name
        self.group_name = group_name
        # Parse the string into a RGB spec.
        r, g, b = hex2rgb(color)
        logging.debug('CropObjectClass {0}: color {1}'.format(name, (r, g, b)))
        self.color = (r, g, b)

    def __str__(self):
        lines = []
        lines.append('<CropObjectClass>')
        lines.append('\t<Id>{0}</Id>'.format(self.clsid))
        lines.append('\t<Name>{0}</Name>'.format(self.name))
        lines.append('\t<GroupName>{0}</GroupName>'.format(self.group_name))
        lines.append('\t<Color>{0}</Color>'.format(rgb2hex(self.color)))
        lines.append('\t</CropObjectClass>')
        return '\n'.join(lines)

#######################################################################
# Utility functions for name/writer conversions
_hex_tr = {
    '0': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15,
}
_hex_itr = {v: k for k, v in _hex_tr.items()}


def parse_hex(hstr):
    # type: (str) -> int
    """Convert a hexadecimal number string to integer.

    >>> parse_hex('33')
    51
    >>> parse_hex('abe8')
    44008

    """
    out = 0
    for i, l in enumerate(reversed(hstr)):
        out += (16**i) * _hex_tr[l]
    return out


def hex2rgb(hstr):
    # type: (str) -> Tuple[float, float, float]
    """Parse a hex-coded color like '#AA0202' into a floating-point representation.

    >>> hex2rgb('#abe822')
    (0.6705882352941176, 0.9098039215686274, 0.13333333333333333)

    """
    if hstr.startswith('#'):
        hstr = hstr[1:]
    rs, gs, bs = hstr[:2], hstr[2:4], hstr[4:]
    r, g, b = parse_hex(rs), parse_hex(gs), parse_hex(bs)
    return r / 255.0, g / 255.0, b / 255.0


def rgb2hex(rgb):
    # type: (Tuple[float, float, float]) -> str
    """Convert a floating-point representation of R, G, B values
    between 0 and 1 (inclusive) to a hex string (strating with a
    hashmark). Will use uppercase letters for 10 - 15.

    >>> rgb = (0.6705882352941176, 0.9098039215686274, 0.13333333333333333)
    >>> rgb2hex(rgb)
    '#ABE822'

    """
    rgb_int = [int(ch * 255) for ch in rgb]
    return '#' + ''.join(['{:02X}'.format(ch) for ch in rgb_int])
