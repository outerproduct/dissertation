#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from nltk.tokenize import punkt


punkt_splitter = punkt.PunktSentenceTokenizer()

# Suffixes observed to frequently cause incorrect splits
# (derived from observations in parsing WikiNews).
bad_suffixes = ['Mr.',
                'Ms.',
                'Mrs.',
                'Dr.',
                'Lt.',
                'Sgt.',
                'Maj.',
#                'Col.',
#                'Gen.',
#                'Adm.',
                'Sen.',
                'Rep.',
                'U.S.',
                'U.N.',
                'U.K.',
                'E.U.',
                'Jan.',
                'Feb.',
                'Mar.',
                'Apr.',
                'Jun.',
                'Jul.',
                'Aug.',
                'Sep.',
                'Oct.',
                'Nov.',
                'Dec.',
                'No.',
                'a.m.',
                'p.m.',
                'sq.',
                'ft.',
                'vs.'
                ]

bad_prefixes = [')',
                ']',
                '}',
                ').',
#                '].',
#                '}.',
#                ')!',
#                ']!',
#                '}!',
#                ')?',
#                ']?',
#                '}?',
                '\"',
                '\'\'',
                '\'s',
                ]


def split(text, fix=True):
    """Split text into sentences using the Punkt Sentence
    Tokenizer from NLTK with some post-processing.
    """
    sents = punkt_splitter.tokenize(text)

    if not fix:
        return sents
    return fix_boundaries(sents, text)


def fix_boundaries(sents, text):
    """Re-merge sentences which seem to have been incorrectly split based
    on their suffixes and move around new sentence prefixes.
    """
    fixed_sents = []
    prev_sent = ''
    prev_gap_len = 0
    i = 0  # offset in text where the next sentence would start
    for s, sent in enumerate(sents):
        i += len(sent)

        if len(prev_sent) > 0:
            # Merge entire sentence back
            gap = ' ' * prev_gap_len
            sent = gap.join((prev_sent, sent))

        elif len(fixed_sents) > 0:
            # Merge back bad prefixes and keep the rest of the sentence
            has_bad_prefix = False
            for bad_prefix in bad_prefixes:
                if sent.startswith(bad_prefix) and \
                        (len(sent) == len(bad_prefix) or
                        sent[len(bad_prefix)] in ' '):
                    has_bad_prefix = True
                    break

            if has_bad_prefix:
                # Merge prefix back
                gap = ' ' * prev_gap_len
                fixed_sents[-1] += gap + bad_prefix

                # Remove prefix from current sentence
                j = len(bad_prefix)
                j += consume_whitespace(text, j)
                sent = sent[j:]

        # Update i and determine the amount of trailing whitespace for
        # forthcoming merges
        prev_gap_len = 0
        if s < len(sents) - 1:
            prev_gap_len += consume_whitespace(text, i)
            i += prev_gap_len

        # Stripping the bad prefix may have resulted in a blank sentence
        if len(sent) == 0:
            prev_sent = ''
            continue

        has_bad_suffix = False
        for bad_suffix in bad_suffixes:
            # If the sentence ends with a bad suffix word, mark it for
            # merging entirely with the next sentence
            if sent.endswith(bad_suffix) and \
                    (len(sent) == len(bad_suffix) or
                    sent[-len(bad_suffix)-1] == ' '):
                has_bad_suffix = True
                break

        if has_bad_suffix:
            # Awaiting future merge
            prev_sent = sent
        else:
            # Looks like a good split so far (may merge in a prefix later)
            prev_sent = ''
            fixed_sents.append(sent)

    assert i == len(text) or text[i] == ' '

    # Last sentence
    if len(prev_sent) > 0:
        fixed_sents.append(prev_sent)

    return fixed_sents


def consume_whitespace(text, idx):
    """Return the amount of whitespace at and following the given character
    index.
    """
    whitespace = 0
    while idx < len(text) and text[idx] == ' ':
        idx += 1
        whitespace += 1
    return whitespace
