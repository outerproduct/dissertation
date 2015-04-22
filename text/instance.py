#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import sentence


class Instance(object):
    """An instance consisting of one or more Sentence-like objects. Subclass
    for further specialization.
    """
    def __init__(self, sentences, **kwargs):
        """Initialize a collection of Sentence-like objects with optional
        label fields.
        """
        # Store the sentences along with optional labels
        self._process_sentences(sentences)
        self.__dict__.update(kwargs)

    def _process_sentences(self, sentences):
        """Generate a list of Sentence or MultiSentence objects from a variety
        of inputs.
        """
        if len(sentences) == 0:
            self.sentences = sentences
            return

        # NOTE: We base all checks on the first tuple in order to ensure
        # consistent (and more efficient) processing.

        # If we already have Sentence-like objects, do nothing
        if isinstance(sentences[0], sentence.Sentence) or \
                isinstance(sentences[0], sentence.MultiSentence):
            self.sentences = sentences

        # If we have a list of strings, assume that each element represents a
        # separate sentence (we expect a list of sentences as input)
        elif isinstance(sentences[0], basestring):
            self.sentences = [sentence.Sentence(s) for s in sentences]

        # If we have a list of lists/tuples, look deeper
        elif isinstance(sentences[0], list) or isinstance(sentences[0], tuple):

            # If it's a list/tuple of strings, check the last string.
            if isinstance(sentences[0][0], basestring):

                # If the last string is sentence-terminating punctuation
                # or doesn't feature a space, assume each string represents
                # a word, and therefore each list/tuple represents a
                # single sentence.
                if sentences[0][-1] in ('.', '?', '!', '\"', '\'') or \
                        (len(sentences[0]) > 1 and \
                        ' ' not in sentences[0][-1]):
                    self.sentences = [sentence.Sentence(s) for s in sentences]

                # Otherwise, assume that each string represents a full
                # sentence and therefore each list/tuple represents a group of
                # multiple connected sentences.
                else:
                    self.sentences = [sentence.MultiSentence(
                                      map(sentence.Sentence, ms))
                                      for ms in sentences]

            # If it's a list/tuple of lists/tuples, just assume that they each
            # contain strings representing words. The inner lists should
            # represent sentences while the outer lists should represent
            # groups of multiple connected sentences.
            elif isinstance(sentences[0][0], list) or \
                    isinstance(sentences[0][0], tuple):
                self.sentences = [sentence.MultiSentence(
                                  map(sentence.Sentence, ms))
                                  for ms in sentences]
            else:
                print "ERROR: unknown type", str(sentences[0].__class__)
                print "Expected Sentence-like objects or lists of strings",
                print "convertible to Sentence-like objects"
                raise TypeError
        else:
            print "ERROR: unknown type", str(sentences[0].__class__)
            print "Expected Sentence-like objects or lists of strings",
            print "convertible to Sentence-like objects"
            raise TypeError

    def get_sentences(self):
        """Retrieve all Sentence-like objects for annotation.
        """
        return self.sentences

    def get_size(self):
        """Retrieve the size of the instance for load balancing in parallel
        learning. The default approximation for this is the number of tokens
        in the instance Sentences.
        """
        return sum(len(sentence.tokens) for sentence in self.sentences)
