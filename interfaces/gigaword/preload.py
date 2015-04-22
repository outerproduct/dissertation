#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import argparse
from interfaces import gigaword
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Compute and save Gigaword corpus statistics')
    parser.add_argument('--stemming', action='store_true',
            help="use word stems instead of raw words")
    parser.add_argument('--docs_path', action='store',
            help="path to the corpus documents",
            default='/proj/fluke/users/kapil/resources/gigaword_eng_4/' +
                    'preprocessed/pdata/')
    parser.add_argument('--pickle_path', action='store',
            help="path to store the statistics",
            default='/proj/fluke/users/kapil/resources/gigaword_eng_4/' +
                    'gigaword.pickle')
    args = parser.parse_args()

    if os.path.exists(args.pickle_path):
        print "Found existing file at", args.pickle_path
        print "Delete this or change --pickle_path"
        sys.exit()
    else:
        gw = gigaword.CorpusStats(docs_path=args.docs_path,
                                  stemming=args.stemming)
        gw.save(args.pickle_path)
