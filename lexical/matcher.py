#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import re
from resources import wordnet
import string   # for maketrans


class Matcher(object):
    """A class for determining whether tokens and phrases match under various
    processing conditions.
    """
    # List available functions for matching
    matchers = [
              'exact',
              'lowercase',
              'stem',
              'lemma',
              'abbrev',
              'negation',
              'wordnet',
              ]
    # TODO: numeric terms, dates, times, quantities, NEs (especially locations)
    # TODO: restrict matches to alphanumeric tokens? or perhaps separate out
    # punctuation?

    # Regular expression to capture abbreviated named entities
    abbrev_re = re.compile(r'[A-Z].*\.')

    # Regular expression to capture negations
    negation_re = re.compile(r'^n[\'o]t$', re.IGNORECASE)

    @classmethod
    def check(cls, t0, sentence0, t1, sentence1, matchers=None):
        """Return True if the given tokens match under the specified matchers.
        """
        # Use default matchers unless a list is provided
        if matchers is None:
            matchers = cls.matchers

        token0 = sentence0.tokens[t0]
        token1 = sentence1.tokens[t1]

        value = False
        for matcher in matchers:
            value |= getattr(cls, matcher)(token0, t0, sentence0,
                                           token1, t1, sentence1)
        return value

###############################################################################
# Matchers

    @classmethod
    def exact(cls, token0, t0, sentence0, token1, t1, sentence1):
        """Return True if the tokens are identical.
        """
        return token0 == token1

    @classmethod
    def lowercase(cls, token0, t0, sentence0, token1, t1, sentence1):
        """Return True if the lowercased tokens match.
        """
        return token0.lower() == token1.lower()

    @classmethod
    def stem(cls, token0, t0, sentence0, token1, t1, sentence1):
        """Return True if the token stems match using the Porter stemmer.
        """
        return sentence0.stems[t0] == sentence1.stems[t1]

    @classmethod
    def lemma(cls, token0, t0, sentence0, token1, t1, sentence1):
        """Return True if the token lemmas match usung Wordnet's Morphy
        lemmatizer.
        """
        if token0.lower() == token1.lower():
            return True

        lemma0 = wordnet.get_lemma(t0, sentence0)
        lemma1 = wordnet.get_lemma(t1, sentence1)
        return lemma0 is not None and lemma1 is not None and lemma0 == lemma1

    @classmethod
    def abbrev(cls, token0, t0, sentence0, token1, t1, sentence1):
        """Return True if both tokens are abbreviations that appear to match.
        """
        # Return False unless both terms are potential abbreviations
        if not cls.is_abbrev(token0) or not cls.is_abbrev(token1):
            return False

        # Lowercase the abbreviations and strip periods to compare
        # TODO: no_trans can just be None in version 2.6 onward
        no_trans = string.maketrans('','')
        return token0.lower().translate(no_trans, '.') \
            == token1.lower().translate(no_trans, '.')

    @classmethod
    def negation(cls, token0, t0, sentence0, token1, t1, sentence1):
        """Return True if both tokens are negations.
        """
        return cls.is_negation(token0) and cls.is_negation(token1)

    @classmethod
    def wordnet(cls, token0, t0, sentence0, token1, t1, sentence1):
        """Return True if at least one of the tokens shares at least
        a third of its synsets with the other token.
        """
        synsets0 = set(wordnet.get_synsets([t0], sentence0))
        synsets1 = set(wordnet.get_synsets([t1], sentence1))
        if len(synsets0) == 0 or len(synsets1) == 0:
            return False
        common = synsets0.intersection(synsets1)
        return len(common) / min(len(synsets0), len(synsets1)) > 0.33

###############################################################################
# Helpers
# TODO: move to separate Tagger class?

    @classmethod
    def is_abbrev(cls, token):
        """Return True if the given token is an abbreviation; False otherwise.
        """
        return len(token) > 1 and \
            (re.match(cls.abbrev_re, token) or token.isupper())

    @classmethod
    def is_negation(cls, token):
        """Return True if the given token is a negation; False otherwise.
        """
        return re.match(cls.negation_re, token) is not None
