#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement


def avg(values):
    """Return the mean of a list or generator. The numpy methods break when
    generators are supplied.
    """
    total, n = 0.0, 0
    for x in values:
        total += x
        n += 1
    return float(total) / n
