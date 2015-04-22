#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import argparse
from collections import defaultdict
import cPickle
import glob
import numpy as np
import os
import re
from stemming import porter2
import sys
from utils import jsonrpc, timer


# A symbol to represent any word, thereby storing counts for normalization
anyword = '******'


class CorpusStats(object):
    """A container for statistics from the Gigaword corpus.
    """
    def __init__(self, docs_path, stemming=False):
        """Initialize the corpus from a raw text version and store counts
        for words and stems.
        """
        self.stemming = stemming

        # For word counts and pairwise counts within a window
        self.word_counts = defaultdict(int)
#            def dictint():
#                return defaultdict(int)
#            self.word_pair_counts = defaultdict(dictint)

        # For inverse document frequency
        self.docs_with_word = defaultdict(int)

        # Read in the corpus and save it
        self.process_all_docs(docs_path)

    @classmethod
    def from_pickle(cls, pickle_path):
        """Return a pickled Gigaword object.
        """
        print "Loading Gigaword corpus from", pickle_path
        with open(pickle_path) as f:
            return cPickle.load(f)

    @classmethod
    def from_server(cls, host_port):
        """Return a proxy Gigaword object with bound methods to
        retrieve counts.
        """
        host, port = host_port.split(':')
        return jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                jsonrpc.TransportTcpIp(addr=(host, int(port))))

    def save(self, pickle_path):
        """Pickle the Gigaword object.
        """
        print "Saving Gigaword corpus to", pickle_path
        with open(pickle_path, 'wb') as f:
            cPickle.dump(self, f, 2)

    def process_all_docs(self, docs_path, **kwargs):
        """Read in all the documents.
        """
        # Walk through once just to figure out the total number of documents
        doc_filepaths = glob.glob(docs_path + '/*/*.txt')
        self.num_docs = len(doc_filepaths)

        with timer.AvgTimer(self.num_docs):
            for d, doc_filepath in enumerate(doc_filepaths):
                docid = self.get_docid(doc_filepath)
                sys.stdout.write("Loading " +
                        str(d) + "/" + str(self.num_docs) +
                        " Gigaword documents: " + docid +
                        " " * 10 + "\r")
                sys.stdout.flush()
                self.process_doc(doc_filepath, **kwargs)

    def get_docid(self, doc_filepath):
        """Return a document id from a filepath.
        """
        filename_offset = doc_filepath.rfind('/') + 1
        return doc_filepath[filename_offset:-7]

    def process_doc(self, docpath, window=None, **kwargs):
        """Process a single document.
        """
        words_in_doc = set()

        with open(docpath) as f:
            for line in f:
                words = re.split('\s', line.lower().strip())
                if self.stemming and window is not None:
                    # Need to stem in advance
                    words = [porter2.stem(word) for word in words]

                # Compute pairwise cooccurrences
                for w, word in enumerate(words):
                    if self.stemming and window is None:
                        # In this case, better to stem inline
                        word = porter2.stem(word)

                    self.word_counts[word] += 1
                    self.word_counts[anyword] += 1
                    words_in_doc.add(word)

                    if window is None:
                        continue

                    # Include following words within window
                    for u in range(w + 1, min(w + window + 1, len(words))):
                        other_word = words[u]

                        # Save space by symmetrizing
                        if word < other_word:
                            self.word_pair_counts[word][other_word] += 1
                        else:
                            self.word_pair_counts[other_word][word] += 1

        for word in words_in_doc:
            self.docs_with_word[word] += 1

    def has_word(self, word):
        """Return whether the word exists in the corpus or not.
        """
        word = word.lower()
        if self.stemming:
            word = porter2.stem(word)
        return word in self.word_counts

    def get_prob(self, word, alpha=0):
        """Return the probability of a word in the corpus with optional
        additive smoothing.
        """
        word = word.lower()
        if self.stemming:
            word = porter2.stem(word)
        return (self.word_counts[word] + alpha) / \
                (self.word_counts[anyword] + len(self.word_counts) * alpha)

    def get_logprob(self, word, alpha=0):
        """Return the log probability of a word in the corpus with optional
        additive smoothing.
        """
        word = word.lower()
        if self.stemming:
            word = porter2.stem(word)
        return np.log(self.word_counts[word] + alpha) - \
                np.log(self.word_counts[anyword] +
                       len(self.word_counts) * alpha)

    def get_idf(self, word):
        """Return the inverse document frequency of a word.
        """
        word = word.lower()
        return np.log(self.num_docs / (1 + self.docs_containing_word[word]))

    def get_pmi(self, word0, word1):
        """Return the pointwise mutual information, a measure of word
        association within a window, for two words. This is normalized
        using Bouma (2009) to avoid infinite values for OOV terms.
        """
        word0 = word0.lower()
        word1 = word1.lower()

        if self.stemming:
            word0 = porter2.stem(word0)
            word1 = porter2.stem(word1)

        if word0 not in self.word_counts or word1 not in self.word_counts:
            return -1

        if word0 < word1:
            pair_counts = self.word_pair_counts[word0][word1]
        else:
            pair_counts = self.word_pair_counts[word0][word1]

        if pair_counts == 0:
            return -1

        num_words = self.word_counts[anyword]

        # TODO: confirm normalization. Currently assuming words are
        # normalized by num_words and pairs by num_words^2.
        ratio = pair_counts / (self.word_counts[word0] *
                               self.word_counts[word1])
        pmi = np.log(ratio)
        normalized_pmi = - pmi / np.log(pair_counts / (num_words * num_words))

        return normalized_pmi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Start a Gigaword data server')
    parser.add_argument('--pickle_path', action='store',
            help="path to load the statistics",
            default='/proj/fluke/users/kapil/resources/gigaword_eng_4/' +
                    'gigaword.pickle')
    parser.add_argument('--host', action='store',
            help="Host to serve on (default localhost; 0.0.0.0 for public)",
            default='127.0.0.1')
    parser.add_argument('--port', action='store', type=int,
            help="Port to serve on (default 8086)",
            default=8086)
    args = parser.parse_args()

    server = jsonrpc.Server(jsonrpc.JsonRpc20(),
                            jsonrpc.TransportTcpIp(addr=(args.host,
                                                         args.port)))

    if not os.path.exists(args.pickle_path):
        print "Can't load Gigaword statistics from", args.pickle_path
        print "Try rerunning gigaword/preload.py"
        sys.exit()

    gw = CorpusStats.from_pickle(args.pickle_path)
    server.register_function(gw.get_prob)
    server.register_function(gw.get_logprob)
    server.register_function(gw.get_idf)
    server.register_function(gw.get_pmi)
    print "Serving Gigaword statistics on http://%s:%s" % \
            (args.host, args.port)
    server.serve()
