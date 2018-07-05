#!/usr/bin/env python
# -*-coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import logging
import os
import sys

import muscima

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


def get_long_description():
    readme = os.path.join(here, 'README.md')
    changes = os.path.join(here, 'CHANGES.md')

    if os.path.isfile(readme) and os.path.isfile(changes):
        long_description = read(readme, changes)
    else:
        logging.warn('Could not find README.md and CHANGES.md file'
                     ' in directory {0}. Contents:'
                     ' {1}'.format(here, os.listdir(here)))
        long_description = 'Tools for the MUSCIMA++ dataset of music notation.' \
                           ' [README.md and CHNAGES.md not found]'
    return long_description

setup(
    name='muscima',
    version=muscima.__version__,
    url='https://muscima.readthedocs.io',
    license='MIT Software License',
    author='Jan Hajič jr.',
    install_requires=['numpy>=1.11.1',
                      'lxml>=3.6.4'],
    author_email='hajicj@ufal.mff.cuni.cz',
    description='Tools for the MUSCIMA++ dataset of music notation.',
    long_description=get_long_description(),
    packages=['muscima'],
    include_package_data=True,
    scripts=['scripts/analyze_agreement.py',
             'scripts/analyze_annotations.py',
             'scripts/analyze_tracking_log.py',
             'scripts/get_images_from_muscima.py',
             'scripts/add_staffline_symbols.py',
             'scripts/add_staff_relationships.py',
             'scripts/strip_staffline_symbols.py'],
    platforms='any',
    test_suite='test.test_muscima',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Web Environment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        ],
)
