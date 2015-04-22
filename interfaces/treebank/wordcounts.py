#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import argparse
import re
from stemming import porter2
import sys
from utils import jsonrpc

class WordCounter(object):
    """A simple counter built on part of speech tags from any corpus.
    """
    def __init__(self, filepath='/proj/fluke/users/kapil/resources/' +
            'treebank-depenified/final.deps', mica=True):
        """Load the corpus from the given filepath.
        """
        # Counts for adjacent parts of speech. Note that total counts for a
        # term can be retrieved by replacing the other term with ******.
        self.noun_verb_counts = {'******': 0}

        # Load the model from a treebank file
        if mica:
            self.load_mica_style(filepath)
        else:
            print "ERROR: not supported yet"
            raise Exception

    @classmethod
    def from_server(cls, host_port):
        """Return a proxy WordCounter object with bound methods to
        retrieve counts.
        """
        host, port = host_port.split(':')
        return jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                jsonrpc.TransportTcpIp(addr=(host, int(port))))

    def load_mica_style(self, filepath):
        """Load a treebank from a file containing MICA-style parses.
        """
        # End of sentence marker
        eos = "...EOS..."

        # A regex for the basic parse content
        num = "([0-9]+)"
        POS = "([A-Z0-9\`\'\$\-_\/\,\.\:\#]+)"
        anything = "([^ ]+)"
        parse_re = re.compile(' '.join((num, anything, POS, num)))

        num_sents = 0
        with open(filepath) as f:
            for line in f:
                if line.startswith(eos):
                    # Ignore EOS markers since we're just building a model
                    num_sents += 1
                    sys.stdout.write("Loading treebank sentences: " +
                            str(num_sents) + "\r")
                    continue
                match = re.match(parse_re, line)
                if match is None:
                    print "ERROR: unrecognized treebank parse format"
                    print line
                    continue

                word = match.group(2)
                stem = porter2.stem(word.lower())
                pos = match.group(3)
                if not pos.startswith('N') and not pos.startswith('V'):
                    continue

                if stem not in self.noun_verb_counts:
                    self.noun_verb_counts[stem] = 1
                else:
                    self.noun_verb_counts[stem] += 1

                # Note the total count of all nouns and verbs
                self.noun_verb_counts['******'] += 1
        print

    def get_noun_verb_counts(self):
        """Return the map of noun and verb counts.
        """
        return self.noun_verb_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Start a dependency model server')
    parser.add_argument('--treebank_path', action='store',
            help="path to the WSJ in Mica parse format",
            default='/proj/fluke/users/kapil/resources/' +
            'treebank-depenified/final.deps')
    parser.add_argument('--host', action='store',
            help="Host to serve on (default localhost; 0.0.0.0 for public)",
            default='127.0.0.1')
    parser.add_argument('--port', action='store', type=int,
            help="Port to serve on (default 8082)",
            default=8082)
    args = parser.parse_args()

    server = jsonrpc.Server(jsonrpc.JsonRpc20(),
                            jsonrpc.TransportTcpIp(addr=(args.host,
                                                         args.port)))
    wc = WordCounter(filepath=args.treebank_path)
    server.register_function(wc.get_noun_verb_counts)
    print "Serving word counts from %s on http://%s:%s" % \
            (args.treebank_path, args.host, args.port)
    server.serve()
