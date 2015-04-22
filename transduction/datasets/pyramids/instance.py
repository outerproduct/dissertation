#!/usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import numpy as np


class FusionInstance(object):
    """A class representing a fusion instance with metadata.
    """
    def __init__(self, input_sents, output_sent, filepaths, year, labels=None):
        """Store the instance.
        """
        self.year = year
        self.input_sents = input_sents
        self.output_sent = output_sent
        self.filepaths = filepaths
        self.labels = labels

    def get_cardinality(self):
        """Return the number of input sentences.
        """
        return len(self.input_sents)

    def get_length_ratio(self):
        """Return the ratio of length of average input sentence to length of
        output sentence (in words).
        """
        return np.mean([len(input_sent.split())
                        for input_sent in self.input_sents]) / \
                                len(self.output_sent.split())

    def print_summary(self):
        """Print a summary of the instance to STDOUT.
        """
        print '\n'.join(self.filepaths)

        for input_sent in self.input_sents:
            print "IN:", input_sent
        print

        if self.labels is not None:
            for label in self.labels:
                print "LBL:", label
            print

        print "OUT:", self.output_sent
        print
        print
