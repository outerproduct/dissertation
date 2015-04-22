#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import argparse
import numpy as np
import os
from srilm_swig import srilm
from unidecode import unidecode
from utils import jsonrpc, timer


class LangModel(object):
    """A class that intializes and maintains SRILM language models for scoring
    n-grams.
    """
    logprob_funcs = [srilm.getUnigramProb,
                     srilm.getBigramProb,
                     srilm.getTrigramProb,
                     srilm.getQuadrigramProb,
                     srilm.getPentagramProb]

    def __init__(self, n,
            lm_path='/proj/fluke/resources/LMs/en.giga.noUN.5gram.lm.bin'):
        """Initialize language models.
        """
        # Record the maximum size of ngrams in the stored LM
        self.n = n

        with timer.Timer():
            print "Initializing language models",

            self.lm = srilm.initLM(n)
            srilm.readLM(self.lm, lm_path)
            print

    def __del__(self):
        """Unload the SRILM language models.
        """
        srilm.deleteLM(self.lm)

    @classmethod
    def from_server(cls, host_port, timeout=60):
        """Return a proxy LM object with bound methods to support ngram
        scoring.
        """
        if host_port is None:
            return None
        host, port = host_port.split(':')
        return jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                jsonrpc.TransportTcpIp(addr=(host, int(port)),
                                       timeout=timeout))

    @classmethod
    def get_ngrams(cls, n, phrase):
        """Generate a list of n-grams from a phrase.
        """
        return [phrase[i:i+n] for i in range(0, len(phrase)-n+1)]

    @classmethod
    def get_join_ngrams(cls, n, phrase0, phrase1):
        """Generate a list of n-grams that span the point of connection between
        the two given phrases.
        """
        return [list(phrase0[i:]) + list(phrase1[:i+n])
                for i in range(max(-len(phrase0), -n+1), 0)
                if len(phrase1) >= i+n]

    def score_ngram(self, ngram):
        """Return the score of a single ngram using an LM with the same order
        as the size of the ngram.
        """
        n = len(ngram)
        if n > self.n or n > 5:
            print "ERROR: %d-gram LM can't directly score %d-grams" % \
                            (self.n, n)
            raise Exception

        # SRILM uses base-10 logarithms, so all probabilities should be
        # converted to base-e by dividing by np.log10(e) or, equivalently,
        # multiplying by np.log(10).
        return LangModel.logprob_funcs[n-1](self.lm,
                ' '.join(unidecode(word) for word in ngram)) * np.log(10)

    def score_ngrams(self, ngrams, normalize=False):
        """Return the sum or average log-probability of a list of n-grams
        under the n-gram LM.
        """
        logprob_sum = sum(self.score_ngram(ngram) for ngram in ngrams)
        if normalize:
            return logprob_sum / len(ngrams)
        else:
            return logprob_sum

    def score_sent(self, words, n=None, normalize=False):
        """Return the sum or average log-probability of a sentence.
        """
        if n is None:
            n = self.n

        # XXX srilm.getSentenceProb doesn't work for sentences with more than
        # 15 words.
        sentence = ['<s>'] + list(words[:]) + ['</s>']
        return self.score_ngrams(self.get_ngrams(n, sentence),
                                 normalize=normalize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Start an LM server')
    parser.add_argument('--ngram_order', action='store', type=int,
            help="order of n-grams to serve",
            default=3)
    parser.add_argument('--lm_path', action='store',
            help="path to the trained SRILM language model",
            default='/proj/fluke/resources/LMs/en.giga.noUN.5gram.lm.bin')
    parser.add_argument('--host', action='store',
            help="host to serve on (default localhost; 0.0.0.0 for public)",
            default=os.environ['HOSTNAME'])
    parser.add_argument('--port', action='store', type=int,
            help="port to serve on (default 8081)",
            default=8081)
    parser.add_argument('--timeout', action='store', type=int,
            help="time limit for responses",
            default=60)
    args = parser.parse_args()

    server = jsonrpc.Server(jsonrpc.JsonRpc20(),
                            jsonrpc.TransportTcpIp(addr=(args.host,
                                                         args.port),
                                                   timeout=args.timeout))
    lm = LangModel(args.ngram_order, lm_path=args.lm_path)
    server.register_function(lm.score_ngram)
    server.register_function(lm.score_ngrams)
    server.register_function(lm.score_sent)
    print 'Serving %s-grams from %s on http://%s:%s' % (args.ngram_order,
            args.lm_path, args.host, args.port)
    server.serve()
