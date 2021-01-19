"""
This file imports content into the keras.tests directory, making this a Python package.
E.g.: Only that way can Rational be imported 'directly' in the test files
"""

from ..rationals import Rational
from ._helpers import _activation, _test_template
