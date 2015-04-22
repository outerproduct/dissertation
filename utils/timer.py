#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import sys
import time


class Timer(object):
    """A simple class for timing the execution of code snippets. To use, import
    this class and place the code snippet under "with Timer():". Source:
    http://mrooney.blogspot.com/2009/07/simple-timing-of-python-code.html
    """
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        sys.stderr.write("\033[0;32m[%(diff).2gs]\033[0m\n"
                            % {"diff": time.time() - self.start})


class AvgTimer(object):
    """A variation on the Timer which can display the average amount of time
    that a repeated process takes.
    """
    def __init__(self, count=1):
        if count > 0:
            self.count = count
        else:
            print "ERROR: Inappropriate count value", count, "for AvgTimer"
            self.count = 1

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        sys.stderr.write("\033[0;32m[%(diff).2gs on avg]\033[0m\n"
                            % {"diff": (time.time()-self.start) / self.count})
