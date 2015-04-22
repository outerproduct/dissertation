#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import argparse
from collections import defaultdict
import numpy as np
import os
import re
from stemming import porter2
import sys
from unidecode import unidecode
from utils import jsonrpc, timer


class DependencyModel(object):
    """A simple arc-factored dependency model built from a treebank parsed
    with Stanford.
    """
    def __init__(self, treebank_path):
        """Load the treebank from the given filepath.
        """
        self.stem_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.num_dependencies = 0

        self.all_labels = set()
        self.all_words = set()
        self.all_stems = set()

        # Regular expression for the typed dependency format
        self.dep_re = re.compile(
                r"^([\w\d]+)\((.+)\-([\d]+), (.+)\-([\d]+)\)$")

        # Load the dependency model from a treebank file
        self.load_treebank(treebank_path)

    @classmethod
    def from_server(cls, host_port, timeout=60):
        """Return a proxy DependencyModel object with bound methods to
        support retrieval of dependency probabiities.
        """
        if host_port is None:
            return None
        host, port = host_port.split(':')
        return jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                jsonrpc.TransportTcpIp(addr=(host, int(port)),
                                       timeout=timeout))

    @classmethod
    def process_token(cls, token):
        """Clean up word and/or POS tokens.
        """
        if token == '``' or token == '\'\'':
            return '\"'
        return token

    @classmethod
    def serialize(cls, *args):
        return '_'.join(unidecode(arg) if isinstance(arg, basestring)
                        else str(arg) for arg in args)

    @classmethod
    def add_to_counter(cls, counter, label, parent_token, child_token,
            direction):
        """Add an instance of the given dependency to the given counter.
        """
        for l in (label, None):
            for p in (parent_token, None):
                for c in (child_token, None):
                    for d in (direction, None):
                        counter[cls.serialize(l,p,c,d)] += 1

    def load_treebank(self, treebank_path):
        """Load dependencies from a file containing Stanford-style dependency
        parses.
        """
        with timer.Timer():
            num_sents = 0
            with open(treebank_path) as f:
                started_sent = False
                for line in f:
                    if line == '\n':
                        # Ignore line but note if a sentence was just
                        # completed
                        if started_sent:
                            num_sents += 1
                            sys.stdout.write("Loading treebank sentences: " +
                                    str(num_sents) + "\r")
                        started_sent = False
                        continue

                    started_sent = True
                    match = re.match(self.dep_re, line)
                    if match is None:
                        print "ERROR: Unexpected Stanford dependency format"
                        print line
                        continue
                    label, token0, t0, token1, t1 = match.groups()

                    # In the Stanford typed dependency format, token0 is
                    # the governor/head and token1 is the dependent
                    direction = None
                    if t0 > t1:
                        # Head follows dependent: left attachment
                        direction = -1
                    elif t0 < t1:
                        # Head precedes dependent: right attachment
                        direction = 1
                    else:
                        print "ERROR: Unexpected token indices"
                        print line
                        continue

                    # Note counts of words
                    token0 = token0.lower() if token0 != 'ROOT' else token0
                    token1 = token1.lower()
                    self.add_to_counter(self.word_counts,
                                        label,
                                        token0,
                                        token1,
                                        direction)

                    # Note counts of stems
                    stem0 = porter2.stem(token0)
                    stem1 = porter2.stem(token1)
                    self.add_to_counter(self.stem_counts,
                                        label,
                                        stem0,
                                        stem1,
                                        direction)

                    # Note total number of unique labels, words and stems
                    self.all_labels.add(label)
                    self.all_words.update((token0, token1))
                    self.all_stems.update((stem0, stem1))

        print
        self.num_labels = len(self.all_labels)
        self.num_words = len(self.all_words)
        self.num_stems = len(self.all_stems)

    def get_joint_logprob(self, label=None, parent=None, child=None,
            direction=None, token_type='word', alpha=0.001):
        """Return the joint probability of a dependency configuration under
        the model. Add-alpha smoothing is performed by default. Without
        this (i.e., when alpha=0), the log probability of an unseen
        configuration will be -inf.
        """
        counter = getattr(self, token_type + '_counts')
        num_tokens = getattr(self, 'num_' + token_type + 's')

        config_deps = counter[self.serialize(label, parent, child, direction)]
        total_deps = counter[self.serialize(None, None, None, None)]

        smoothing_term = (self.num_labels if label is not None else 1) * \
                         (num_tokens if parent is not None else 1) * \
                         (num_tokens if child is not None else 1) * \
                         (2 if direction is not None else 1)

        return np.log(config_deps + 1 * alpha) - \
                np.log(total_deps + smoothing_term * alpha)

    def get_logprob(self, label=None, parent=None, child=None, direction=None,
            given_label=None, given_parent=None, given_child=None,
            given_direction=None, token_type='word', alpha=0.001):
        """Return some conditional (or joint) log probability under the model.
        At least one non-conditioned attribute should be specified. Add-alpha
        smoothing is performed by default in the joint probability computation.
        Without this (i.e., when alpha=0), the log probability of an unseen
        configuration will be -inf and conditioning on an unseen
        configuration will yield nan.
        """
        if label is None and parent is None and child is None and \
                direction is None:
            print "ERROR: specify at least one variable to compute probability"
            raise Exception

        joint_logprob = self.get_joint_logprob(
                label=label if label is not None else given_label,
                parent=parent if parent is not None else given_parent,
                child=child if child is not None else given_child,
                direction=direction if direction is not None else
                given_direction,
                alpha=alpha,
                token_type=token_type)

        if given_label is None and given_parent is None and \
                given_child is None and given_direction is None:
            return joint_logprob

        marginalized_logprob = self.get_joint_logprob(
                label=given_label if label is None else None,
                parent=given_parent if parent is None else None,
                child=given_child if child is None else None,
                direction=given_direction if direction is None else None,
                alpha=alpha,
                token_type=token_type)

        return joint_logprob - marginalized_logprob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Start a dependency model server')
    parser.add_argument('--treebank_path', action='store',
            help="path to the WSJ in Stanford parse format",
            default='/proj/fluke/users/kapil/resources/' +
                    'treebank-depenified-stanford/wsj_gold_stanford_all.deps')
    parser.add_argument('--host', action='store',
            help="host to serve on (default localhost; 0.0.0.0 for public)",
            default=os.environ['HOSTNAME'])
    parser.add_argument('--port', action='store', type=int,
            help="port to serve on (default 8082)",
            default=8082)
    parser.add_argument('--timeout', action='store', type=int,
            help="time limit for responses",
            default=60)
    args = parser.parse_args()

    server = jsonrpc.Server(jsonrpc.JsonRpc20(),
                            jsonrpc.TransportTcpIp(addr=(args.host,
                                                         args.port),
                                                   timeout=args.timeout))
    dm = DependencyModel(args.treebank_path)
    server.register_function(dm.get_logprob)
    print 'Serving dependencies from %s on http://%s:%s' % \
            (args.treebank_path, args.host, args.port)
    server.serve()
