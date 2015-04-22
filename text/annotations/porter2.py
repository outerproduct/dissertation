#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from stemming import porter2
import text.sentence
import unicodedata


class Porter2(object):
    """Annotator interface to the Porter2 stemmer.
    """
    @staticmethod
    def normalize(text):
        """Normalize unicode text as ASCII.
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore') \
                if isinstance(text, unicode) \
                else text

    def run_on_corpus(self, corpus):
        """Run the Porter2 stemmer on the words.
        """
        # Check if the corpus consists of Sentences or MultiSentences, and
        # get a single list of Sentences either way
        sentences = []
        if corpus[0].__class__ == text.sentence.MultiSentence:
            for multisentence in corpus:
                # Collect the Sentence objects from each MultiSentence
                sentences.extend(multisentence.sentences)
        else:
            sentences = corpus

        for sentence in sentences:
            stems = [porter2.stem(self.normalize(token.lower()))
                    for token in sentence.tokens]
            sentence.add_token_tags(stems, name='stems', annotator='porter2')
