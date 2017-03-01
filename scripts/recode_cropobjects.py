#!/usr/bin/env python
"""This is a script that ensures the specified CropObjectList files are
formatted up-to-date:

* Uses ``<Top>`` and ``<Left>``, not ``<X>`` and ``<Y>``
* Does not use ``<Selected>``
* Does not use ``<MLClassId>``

You can either provide a root directory, individual files, and ``--outdir``,
which takes the files together with their filenames and creates the re-coded
copies in the output directory (including paths), or you can provide
``--inplace`` and the script modifies the file in-place.

Example::

  recode_xy_to_topleft.py -r /my/data/cropobjects -i /my/data/cropobjects/*.xml
                          -o /my/data/recoded-cropobjects
"""
from __future__ import print_function, unicode_literals
import argparse
import copy
import logging
import os
import time

from muscima.io import parse_cropobject_list, export_cropobject_list

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################

def get_document_namespace(filename, root=None, output_dir=None):
    """Derives the document namespace for a CropObjectList file
    with the given filename, optionally with a given root
    and output dir.

    In fact, only takes ``os.path.splitext(os.path.basename(filename))[0]``.
    """
    return os.path.splitext(os.path.basename(filename))[0]


def recode_ids(cropobjects,
               document_namespace,
               dataset_namespace):
    """Recode all IDs of the given CropObjects, so that they are (hopefully)
    globally unique. That is, from e.g. ``611``, we get
    ``MUSCIMA++_1.0::CVC-MUSCIMA_W-35_N-08_D-ideal::611.

    Creates new CropObjects, does *not* modify the input in-place.

    :param cropobjects: A list of CropObject instances.

    :param document_namespace: An identifier of the given
        CropObjectList. It should be unique for each dataset,
        i.e. ``absolute_dataset_namespace``.

    :param dataset_namespace: An identifier of the given
        dataset. It should be globally unique (which is impossible
        to guarantee, but at least within further versions of MUSCIMA++,
        it should hold).
    """
    output_cropobjects = []
    for c in cropobjects:
        c_out = copy.deepcopy(c)
        uid = c.UID_DELIMITER.join([dataset_namespace,
                                    document_namespace,
                                    str(c.objid)])
        c_out.set_uid(uid)
        output_cropobjects.append(c_out)
    return output_cropobjects

##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-r', '--root', action='store', default=None,
                        help='Root directory of the CropObjectList files.'
                             ' The paths of the input files will be interpreted'
                             ' relative to this directory in order to place'
                             ' the output files, unless \'--inplace\' is given.'
                             ' If \'--output_dir\' is given but \'--root\''
                             ' is not, the ')
    parser.add_argument('-o', '--output_dir', action='store',
                        help='Output directory for the recoded files.'
                             ' Equivalent role to the \'--root\': if you'
                             ' supply a file /my/root/subdir/somefile.xml,'
                             ' root /my/root/, and output dir /my/output, the'
                             ' output file will be /my/output/subdir/somefile.xml.'
                             ' If the output dir does not exist, it will be'
                             ' created')
    parser.add_argument('-i', '--input_files', action='store', nargs='+',
                        help='Input files. Full paths, *including* root dir:'
                             ' the root is only there for retaining directory'
                             ' structure, if applicable. (This is because you'
                             ' will probably want to use shell wildcards, and'
                             ' it would not work if you did not supply the'
                             ' full paths to the input directory/directories.)')
    parser.add_argument('--inplace', action='store_true',
                        help='Modify input files in-place.')

    parser.add_argument('--recode_uids', action='store_true',
                        help='Add UIDs to CropObjects. The dataset namespace'
                             ' is given by \'--uid_global_namespace\', the'
                             ' document namespace is derived from filenames'
                             ' (as basename, without filetype extension).')
    parser.add_argument('--uid_dataset_namespace', action='store',
                        default='MUSCIMA-pp_1.0',
                        help='If UIDs are getting added, this is their global'
                             ' namespace.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    ##########################################################################
    logging.info('Converting to absolute paths...')
    root = None
    if args.root is not None:
        root = os.path.abspath(args.root)
    output_dir = os.path.abspath(args.output_dir)
    input_files = [os.path.abspath(f) for f in args.input_files]
    logging.info('Root: {0}'.format(root))
    logging.info('Output: {0}'.format(output_dir))
    logging.info('Example input: {0}'.format(input_files[0]))

    ##########################################################################
    # Get output filenames,
    # fail on non-corresponding input file and root.
    logging.info('Inferring output pathnames...')
    if args.inplace:
        output_files = input_files
    else:
        if args.root is None:
            relative_files = input_files
        else:
            len_root = len(root)
            relative_files = []
            for f in input_files:
                if not os.path.samefile(os.path.commonpath([f, root]),
                                        root):
                    raise ValueError('Input file {0} does not seem to'
                                     ' come from the root directory {1}.'
                                     ''.format(f, root))
                relative_files.append(f[len_root+1:])

        # Ensure output dir exists
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        logging.debug('Making output file names. Output dir: {0}'.format(output_dir))
        logging.debug('Example rel file: {0}'.format(relative_files[0]))
        logging.debug('Ex. output: {0}'.format(os.path.join(output_dir, relative_files[0])))
        output_files = [os.path.join(output_dir, f)
                        for f in relative_files]
        logging.debug('Local Example output file: {0}'.format(output_files[0]))

    logging.info('Example output file: {0}'.format(output_files[0]))

    ##########################################################################
    # Parse cropobjects
    logging.info('Parsing cropobject files ({0} total)...'.format(len(input_files)))
    cropobjects_for_files = []
    for i, f in enumerate(input_files):
        cropobjects_for_files.append(parse_cropobject_list(f))
        if (i > 0) and (i % 10 == 0):
            logging.info('Parsed {0} files.'.format(i))

        if args.recode_uids:
            dataset_namespace = args.uid_dataset_namespace
            document_namespace = get_document_namespace(filename=f,
                                                        root=root,
                                                        output_dir=output_dir)
            recoded_cropobjects = recode_ids(cropobjects_for_files[-1],
                                             document_namespace=document_namespace,
                                             dataset_namespace=dataset_namespace)
            cropobjects_for_files[-1] = recoded_cropobjects


    ##########################################################################
    logging.info('Exporting cropobjects...')
    _i = 0
    for output_file, c in zip(output_files, cropobjects_for_files):
        s = export_cropobject_list(c)
        with open(output_file, 'w') as hdl:
            hdl.write(s)
            hdl.write('\n')

        _i += 1
        if (_i % 10) == 0:
            logging.info('Done: {0} files'.format(_i))

    _end_time = time.clock()
    logging.info('recode_xy_to_topleft.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
