'''Unit tests for the muscima package.

Runs doctests.
'''
from __future__ import print_function, unicode_literals, division
import unittest
import doctest
import os

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
    return tests

if __name__ == '__main__':
    unittest.main()
