'''Unit tests for the muscima package.

Runs doctests.
'''
from __future__ import print_function, unicode_literals, division
import unittest
import doctest
import os
import logging

import muscima.cropobject
import muscima.cropobject_class
import muscima.dataset
import muscima.grammar
import muscima.io


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(muscima.cropobject))
    tests.addTests(doctest.DocTestSuite(muscima.cropobject_class))
    tests.addTests(doctest.DocTestSuite(muscima.dataset))
    tests.addTests(doctest.DocTestSuite(muscima.grammar))
    tests.addTests(doctest.DocTestSuite(muscima.io))
    logging.warning('Loading tests from doctests')
    return tests


def test_muscima():
    logging.warning('Running test_muscima: unittest.main()')
    unittest.main(module=__name__, exit=False)


if __name__ == '__main__':
    unittest.main()
