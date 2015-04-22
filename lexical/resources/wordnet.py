#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from nltk.corpus import wordnet


# Mapper from standard PTB POS tags to Wordnet-style POS tags
# NOTE: NLTK's Wordnet interface considers adjectve satellites 's' as
# regular adjectives 'a' by default
tag_mapper = {'NN'  : 'n',
                'NNS' : 'n',
                'NP'  : 'n',
                'NPS' : 'n',
                'NNP' : 'n',
                'NNPS': 'n',
                'VB'  : 'v',
                'VBD' : 'v',
                'VBG' : 'v',
                'VBN' : 'v',
                'VBP' : 'v',
                'VBZ' : 'v',
                'JJ'  : 'a',
                'JJR' : 'a',
                'JJS' : 'a',
                'RB'  : 'r',
                'RBR' : 'r',
                'RBS' : 'r'}


# Mapper from standard chunk tags to Wordnet-style POS tags
# NOTE: NLTK's Wordnet interface considers adjectve satellites 's' as
# regular adjectives 'a' by default
chunk_mapper = {'NP'   : 'n',
                'VP'   : 'v',
                'ADJP' : 'a',
                'ADVP' : 'r'}


def get_lemma(w_idx, sentence, relaxed=False):
    """Return the lemma for a word in an annotated Sentence using Wordnet's
    Morphy.
    """
    pos = sentence.pos_tags[w_idx]

    wn_pos = None
    if pos in tag_mapper:
        wn_pos = tag_mapper[pos]
    elif not relaxed:
        return None

    word = sentence.tokens[w_idx].lower()
    return wordnet.morphy(word, wn_pos)


def get_synsets(w_idxs, sentence, relaxed=False):
    """Return the Wordnet synsets for a phrase in an annotated Sentence.
    """
    if len(w_idxs) == 1:
        # For a single word, use its POS tag
        pos = sentence.pos_tags[w_idxs[0]]
        wn_pos = None
        if pos in tag_mapper:
            wn_pos = tag_mapper[pos]
        elif not relaxed:
            return []

        word = sentence.tokens[w_idxs[0]]
        return wordnet.synsets(word, wn_pos)
    else:
        # For a multi-word phrase, use its chunk tag if it has one
        span = (w_idxs[0], w_idxs[-1])
        wn_pos = None
        if span in sentence.chunks:
            chunk_tag = sentence.chunks[span]
            if chunk_tag in chunk_mapper:
                wn_pos = chunk_mapper[chunk_tag]
            elif not relaxed:
                return []
        elif not relaxed:
            return []

        phrase = '_'.join(sentence.tokens[w].lower() for w in w_idxs)
        return wordnet.synsets(phrase, wn_pos)
