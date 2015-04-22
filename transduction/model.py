#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
from interfaces import srilm
from interfaces import treebank
import itertools
from learning import features as featuremod
from lexical.resources import framenet
import numpy as np
from operator import itemgetter
from pyutilib.enum import Enum
from random import choice
from utils import jsonrpc


# An enum for token indices that represent special variables when paired
# with a sentence index of None
metaidx = Enum('START', 'END', 'ROOT', 'LEAF')

# TODO: confirm that these POS tags are adequate for all taggers used
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
            'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$',
            'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
            'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', ':',
            '#', '$', '``', '\'\'', '(', ')']

# Add prefix tag categories. We use startswith() for POS tag comparison
# when the tag is a single character long so these act like category labels
# for broad parts of speech, e.g., all nouns or WH-words collectively.
prefix_tags = ['N', 'V', 'J', 'R', 'W']
pos_tags.extend(prefix_tags)
prefix_tag_set = set(prefix_tags)

# Chunk labels that we know TagChunk handles.
chunk_tags = ['NP', 'VP', 'PP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'CONJP',
        'INTJ', 'LST', 'UCP']

# Stanford dependency labels arranged according to the hierarchy described
# in the Stanford dependencies manual. The hierarchy position of 'pcomp'
# is inferred since it is not in the manual, likely because this label is
# eliminated when preposition edges are collapsed.
label_hierarchy = {'root':      ['root'],
                   'dep':       ['dep'],
                   'aux':       ['dep', 'aux'],
                   'auxpass':   ['dep', 'aux', 'auxpass'],
                   'cop':       ['dep', 'aux', 'cop'],
                   'arg':       ['dep', 'arg'],
                   'agent':     ['dep', 'arg', 'agent'],
                   'comp':      ['dep', 'arg', 'comp'],
                   'acomp':     ['dep', 'arg', 'comp', 'acomp'],
                   'attr':      ['dep', 'arg', 'comp', 'attr'],
                   'ccomp':     ['dep', 'arg', 'comp', 'ccomp'],
                   'xcomp':     ['dep', 'arg', 'comp', 'xcomp'],
                   'complm':    ['dep', 'arg', 'comp', 'complm'],
                   'obj':       ['dep', 'arg', 'comp', 'obj'],
                   'dobj':      ['dep', 'arg', 'comp', 'obj', 'dobj'],
                   'iobj':      ['dep', 'arg', 'comp', 'obj', 'iobj'],
                   'pobj':      ['dep', 'arg', 'comp', 'obj', 'pobj'],
                   'pcomp':     ['dep', 'arg', 'comp', 'pcomp'], # inferred
                   'mark':      ['dep', 'arg', 'comp', 'mark'],
                   'rel':       ['dep', 'arg', 'comp', 'rel'],
                   'subj':      ['dep', 'arg', 'subj'],
                   'nsubj':     ['dep', 'arg', 'subj', 'nsubj'],
                   'nsubjpass': ['dep', 'arg', 'subj', 'nsubj', 'nsubjpass'],
                   'csubj':     ['dep', 'arg', 'subj', 'csubj'],
                   'csubjpass': ['dep', 'arg', 'subj', 'csubj', 'csubjpass'],
                   'cc':        ['dep', 'cc'],
                   'conj':      ['dep', 'conj'],
                   'expl':      ['dep', 'expl'],
                   'mod':       ['dep', 'mod'],
                   'abbrev':    ['dep', 'mod', 'abbrev'],
                   'amod':      ['dep', 'mod', 'amod'],
                   'appos':     ['dep', 'mod', 'appos'],
                   'advcl':     ['dep', 'mod', 'advcl'],
                   'purpcl':    ['dep', 'mod', 'purpcl'],
                   'det':       ['dep', 'mod', 'det'],
                   'predet':    ['dep', 'mod', 'predet'],
                   'preconj':   ['dep', 'mod', 'preconj'],
                   'infmod':    ['dep', 'mod', 'infmod'],
                   'mwe':       ['dep', 'mod', 'mwe'],
                   'partmod':   ['dep', 'mod', 'partmod'],
                   'advmod':    ['dep', 'mod', 'advmod'],
                   'neg':       ['dep', 'mod', 'advmod', 'neg'],
                   'rcmod':     ['dep', 'mod', 'rcmod'],
                   'quantmod':  ['dep', 'mod', 'quantmod'],
                   'nn':        ['dep', 'mod', 'nn'],
                   'npadvmod':  ['dep', 'mod', 'npadvmod'],
                   'tmod':      ['dep', 'mod', 'npadvmod', 'tmod'],
                   'num':       ['dep', 'mod', 'num'],
                   'number':    ['dep', 'mod', 'number'],
                   'prep':      ['dep', 'mod', 'prep'],
                   'poss':      ['dep', 'mod', 'poss'],
                   'possessive':['dep', 'mod', 'possessive'],
                   'prt':       ['dep', 'mod', 'prt'],
                   'parataxis': ['dep', 'parataxis'],
                   'punct':     ['dep', 'punct'],
                   'ref':       ['dep', 'ref'],
                   'sdep':      ['dep', 'sdep'],
                   'xsubj':     ['dep', 'sdep', 'xsubj'],
                   }

# RASP grammatical relation types
gr_types = ['ta', 'det', 'aux', 'conj', 'ncmod', 'xmod', 'cmod', 'pmod',
            'ncsubj', 'xsubj', 'csubj', 'dobj', 'obj2', 'iobj', 'pcomp',
            'xcomp', 'ccomp']

# Prespecified lexical tokens used in word features
negations = set(['not', 'n\'t', 'never', 'no'])
open_parens = set(('(', '[', '{', '<'))
close_parens = set((')', ']', '}', '>'))

# A mapping from a normalization category to the corresponding instance
# attribute.
norm_attrib_map = {'1': None,
                   'num': 'num_sents_reciprocal',
                   'avg': 'avg_len_reciprocal',
                   'sum': 'sum_len_reciprocal',
                   }

##############################################################################
# Helper resources needed for computing features. These are not recorded as
# members of the features class in order to prevent them being written
# to disk when the model is saved.
lm_servers = []
lm_proxy = None

dep_servers = []
dep_proxy = None

# Number of failures permitted for remote models
failure_limit = 10

def init_servers(given_lm_servers, given_dep_servers):
    """Restore models that are needed for computing features.
    """
    lm_servers.extend(given_lm_servers)
    dep_servers.extend(given_dep_servers)

    lm_server = choice(lm_servers)
    print "Using", lm_server, "as an LM server"
    globals()['lm_proxy'] = srilm.LangModel.from_server(lm_server)

    dep_server = choice(dep_servers)
    print "Using", dep_server, "as a dependency model server"
    globals()['dep_proxy'] = treebank.DependencyModel.from_server(dep_server)

##############################################################################

class TransductionFeatures(featuremod.Features):
    """A collection of classmethods that specify individual features and
    groups of features over transduction instances. Can be instantiated
    in order to incorporate more involved feature organization.
    """
    def __init__(self, feature_conf, instances=None, norm_conf=('1',),
            ngram_order=3, standardize=False):
        """Initialize and store the full list of features.
        """
        # Extract lexical or syntactic occurrences to populate feature
        # templates.
        if instances is not None:
            occurrences = self.extract_occurrences(instances)
        else:
            print "WARNING: no instances provided;"
            print "feature templates may be empty"
            occurrences = {}

        # Instance attribute names that indicate different normalization
        # values for scale-dependent features. (None -> 1)
        normalizers = sorted([norm_attrib_map[name] for name in norm_conf])
        if len(normalizers) == 0:
            print "WARNING: no normalizers were defined;",
            print "all scale-dependent features will be dropped"

        # Retrieve full list of features and note categories
        featuremod.Features.__init__(self,
                                     feature_conf,
                                     standardize=standardize,
                                     normalizers=normalizers,
                                     ngram_order=ngram_order,
                                     **occurrences)

    @classmethod
    def extract_occurrences(cls, instances):
        """Extract occurrences of lexical or syntactic information for
        feature templates which avoid sparsity.
        """
        # Specific tag sequences extracted from training data
        corpus_pos_pairs = set()
        corpus_pos_triples = set()

        # Specific conjoinings of labels and child/parent POS tags from
        # training data
        corpus_label_pos = set()

        # Specific word classes extracted from training data for lexical
        # features
        lex_classes = (
                       #'CC',
                       #'DT',
                       #'IN',
                       #'PDT',
                       'SYM',
                       )
        lex_words = set()
        verb_stems = set()
        prep_stems = set()
        preps = set()
        fn_words = set()

        for instance in instances:
            for sent in instance.input_sents:
                prev_tags = ['START']
                prev_prev_tags = []
                for t, token in enumerate(sent.tokens):
                    tag = sent.pos_tags[t]
                    if tag.startswith('VB'):
                        verb_stems.add(sent.stems[t])
                    elif tag.startswith('IN'):
                        preps.add(token.lower())
                        prep_stems.add(sent.stems[t])
                    elif tag[:2] not in ('NN', 'JJ', 'VB', 'CD', 'RB', 'UH'):
                        fn_words.add(token.lower())
                    else:
                        for lex_class in lex_classes:
                            if tag.startswith(lex_class):
                                lex_words.add(token.lower())

                    # POS tag sequences
                    cur_tags = cls.get_implied_tags(tag)
                    corpus_pos_pairs.update((prev_tag, cur_tag)
                                    for prev_tag in prev_tags
                                    for cur_tag in cur_tags)
                    corpus_pos_triples.update((prev_prev_tag, prev_tag,
                                               cur_tag)
                                    for prev_prev_tag in prev_prev_tags
                                    for prev_tag in prev_tags
                                    for cur_tag in cur_tags)

                    prev_prev_tags = prev_tags
                    prev_tags = cur_tags

                    # POS tags and labels on dependency arcs
                    # NOTE: not handling leaves here since this is only
                    # used for word features
                    if sent.dparse.is_root(t):
                        labels = ['root']
                        gov_tags = ['ROOT']
                    else:
                        node = sent.dparse.nodes[t]
                        labels = cls.get_implied_labels(
                                node.get_incoming_attribs('label')[0])
                        gov_tags = cls.get_implied_tags(
                                sent.pos_tags[sent.dparse.get_parent_idx(t)])

                    # To avoid very sparse cubic features, we only record
                    # pairwise occurrences between POS tags, governor POS tags
                    # and labels rather than all three. In addition, we note
                    # features for labels and governor POS tags by themselves.
                    for cur_tag in cur_tags:
                        corpus_label_pos.update(itertools.product(
                            labels, [None], cur_tags + [None]))
                        corpus_label_pos.update(itertools.product(
                            labels + [None], gov_tags, [None]))
                        corpus_label_pos.update(itertools.product(
                            [None], gov_tags, cur_tags))

                # Add a terminal pseudo-token at the end of each sentence.
                corpus_pos_pairs.update((prev_tag, 'END')
                                    for prev_tag in prev_tags)
                corpus_pos_triples.update((prev_prev_tag, prev_tag, 'END')
                                    for prev_prev_tag in prev_prev_tags
                                    for prev_tag in prev_tags)

        # Return a dictionary that can be directly used for kwargs
        return {'corpus_label_pos': corpus_label_pos,
                'corpus_pos_pairs': corpus_pos_pairs,
                'corpus_pos_triples': corpus_pos_triples,
                'lex_words': sorted(lex_words),
                'preps': sorted(preps),
                'prep_stems': sorted(prep_stems),
                'verb_stems': sorted(verb_stems),
                'fn_words': sorted(fn_words)}

    @classmethod
    def get_implied_labels(cls, label, shallowest=1, deepest=4):
        """Return a list of all labels implied by the given label.
        """
        all_implied = label_hierarchy[label]
        if len(all_implied) <= shallowest:
            return [label]
        else:
            return all_implied[shallowest:deepest+1]

    @classmethod
    def get_implied_tags(cls, tag):
        """Return a list of all POS tags implied by the given POS tag.
        """
        if len(tag) > 1 and tag[0] in prefix_tag_set:
            return [tag, tag[0]]
        else:
            return [tag]

##############################################################################
# Features

    @classmethod
    def word_norm(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return 1 for every additional word, thereby functioning
        as a sort of bias offset for the learning algorithm.
        """
        feats = []
        for k, norm in enumerate(normalizers):
            feats.append((ext_offset + k,
                1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def word_capitalization(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return 1 if the word contains any capitalization else return 0.
        We use this as an approximation for named entities and events.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        word = instance.input_sents[s].tokens[w]

        feats = []
        if word != word.lower():
            for k, norm in enumerate(normalizers):
                feats.append((ext_offset + k,
                    1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def word_capitalization_seq(cls, instance, word_idxs, reverse=False,
            **kwargs):
        """Return the position of this word in a sequence of capitalized
        words. This allows the model to prefer leading or trailing words.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        sent = instance.input_sents[s]
        seq_length = 0
        if reverse:
            while w < len(sent.tokens) and \
                    sent.tokens[w] != sent.tokens[w].lower():
                w += 1
                seq_length += 1
        else:
            while w >= 0 and \
                    sent.tokens[w] != sent.tokens[w].lower():
                w -= 1
                seq_length += 1
        return seq_length

    @classmethod
    def word_negation(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return 1 if the word is a negation, otherwise return 0.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        word = instance.input_sents[s].tokens[w]

        feats = []
        if word in negations:
            for k, norm in enumerate(normalizers):
                feats.append((ext_offset + k,
                    1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def word_in_parens(cls, instance, word_idxs, **kwargs):
        """Return 1 if the word is within parentheses (inclusive), 0
        otherwise.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        sent = instance.input_sents[s]

        # Both opening and closing parentheses are also considered to be
        # parenthesized, so the first token should be checked to see whether
        # it ends a parenthesized clause.
        if sent.tokens[w] in close_parens:
            return 1

        # Return 1 if an unmatched open parentheses is found before this token
        nesting = 0
        while w >= 0:
            if sent.tokens[w] in open_parens:
                nesting += 1
            elif sent.tokens[w] in close_parens:
                nesting -= 1

            if nesting > 0:
                return 1
            w -= 1
        return 0

    @classmethod
    def word_tfidf(cls, instance, word_idxs, **kwargs):
        """Return the tf-idf of the word if the word is a noun or verb.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        word = instance.input_sents[s].tokens[w].lower()
        return instance.tfidf[word]

    @classmethod
    def word_depth(cls, instance, word_idxs, normalize=True,
            **kwargs):
        """Determine the depth of a token in its tree, optionally normalizing
        by the maximum depth of the tree.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        depth = instance.input_sents[s].dparse.nodes[w].depth
        if normalize:
            max_depth = instance.input_sents[s].dparse.max_depth
            if max_depth == 0:
                return 0
            return depth / max_depth
        else:
            return depth

    @classmethod
    def word_significance(cls, instance, word_idxs, scale=1, normalize=True,
            **kwargs):
        """Provide a measure of how relevant this word might be for a
        compressed sentence using the significance score of
        Clarke & Lapata (2008).
        """
        assert len(word_idxs) == 1

        return scale * \
                cls.word_depth(instance,
                                word_idxs,
                                normalize=normalize,
                                **kwargs) * \
                cls.word_tfidf(instance,
                                word_idxs,
                                **kwargs)

    @classmethod
    def word_significance_fixed(cls, instance, word_idxs, scale=1, **kwargs):
        """A version of word significance that will not be weighted
        by parameters in order to mimic Clarke's tuning.
        """
        return cls.word_significance(instance, word_idxs, scale=scale,
                **kwargs)

    @classmethod
    def word_fidelity(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return whether the word appears in the input or not.
        """
        assert len(word_idxs) == 1

        found_match = False
        ss, ww = word_idxs[0]
        matching_word_idxs = [(s,w)
                for s, w_list in instance.input_maps[ss][ww].iteritems()
                for w in w_list]

        # Check for a word from the original input sentences
        for s, w in matching_word_idxs:
            if s in instance.original_sent_idxs and \
                    w >= 0 and w < instance.sent_lens[s]:
                found_match = True
                break

        feats = []
        if found_match:
            for k, norm in enumerate(normalizers):
                feats.append((ext_offset + k,
                    1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def word_label_pos(cls, instance, word_idxs, label_pos_indicators,
            shallowest=1, normalizers=(None,), ext_offset=0, **kwargs):
        """Return matches for dependency labels and POS tags.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        sent = instance.input_sents[s]
        parse = sent.dparse

        dep_tags = cls.get_implied_tags(sent.pos_tags[w])

        if parse.is_root(w):
            gov_tags = ['ROOT']
            labels = ['root']
        else:
            node = parse.nodes[w]
            gov_tags = cls.get_implied_tags(sent.pos_tags[node.parent_idx])
            labels = cls.get_implied_labels(
                    node.get_incoming_attribs('label')[0],
                    shallowest=shallowest)

        occurrence_tuples = itertools.product(labels + [None],
                                              gov_tags + [None],
                                              dep_tags + [None])
        feats = []
        for occurrence in occurrence_tuples:
            try:
                i = label_pos_indicators[occurrence]
                feat_offset = ext_offset + len(normalizers) * i
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                # Occurrence not present in features, so we ignore it
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def word_pos_seq(cls, instance, word_idxs, pos_indicators, w_offsets,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the word is surrounded by or a part of the specified
        sequence of part of speech tags.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        sent = instance.input_sents[s]

        feats = []
        for o, offset_seq in enumerate(w_offsets):
            w_idxs = [w + w_offset for w_offset in offset_seq]
            if w_idxs[0] < -1 or w_idxs[-1] > len(sent.tokens):
                # These won't exist in the indicator list by design
                continue

            partial_pos_seqs = [None] * len(w_idxs)
            for i, idx in enumerate(w_idxs):
                # This will become a list of POS tag sequences of
                # length i+1
                partial_pos_seqs[i] = []

                if idx == -1:
                    partial_pos_seqs[i].append(['START'])
                elif idx == len(sent.tokens):
                    for seq in partial_pos_seqs[i-1]:
                        partial_pos_seqs[i].append(seq + ['END'])
                else:
                    tags = cls.get_implied_tags(sent.pos_tags[idx])
                    if i == 0:
                        partial_pos_seqs[i].extend([tag] for tag in tags)
                    else:
                        # Combine the current token's POS tags with the
                        # previous partial POS sequences
                        partial_pos_seqs[i].extend(seq + [tag]
                                for seq in partial_pos_seqs[i-1]
                                for tag in tags)

            # The last column of partial POS sequences after considering
            # all words in the offset range is the final set of POS
            # sequences
            final_pos_seqs = partial_pos_seqs[-1]

            for pos_seq in final_pos_seqs:
                pos_seq = tuple(pos_seq)
                try:
                    j = pos_indicators[pos_seq]
                    feat_offset = ext_offset + \
                            (o * len(pos_indicators)
                                * len(normalizers)) + \
                            (j * len(normalizers))
                    for k, norm in enumerate(normalizers):
                        feats.append((feat_offset + k,
                            1 if norm is None
                            else getattr(instance, norm)))
                except KeyError:
                    pass

        return sorted(feats, key=itemgetter(0))

    @classmethod
    def word_lex(cls, instance, word_idxs, word_indicators, w_offsets,
            use_stem=False, normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the word matches the given target word.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        sent = instance.input_sents[s]
        tokens = sent.stems if use_stem else sent.tokens

        feats = []
        for o, w_offset in enumerate(w_offsets):
            w_idx = w + w_offset
            if w_idx <= 0 or w_idx >= len(sent.tokens):
                # These won't exist in the tuple list by design
                continue
            token = tokens[w_idx].lower()

            try:
                j = word_indicators[token]
                feat_offset = ext_offset + \
                        (o * len(word_indicators) * len(normalizers)) + \
                        (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def word_support(cls, instance, word_idxs, pos_threshold_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the word in this sentence is supported by
        a similar word's occurrence in other sentences. This is shown by
        binary features over whether the support exceeds a threshold
        or a raw occurrence count if the given threshold is None.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        if s is None:
            return []

        sent = instance.input_sents[s]

        # Support is only defined for nouns, verbs, adjectives and adverbs
        tag = sent.pos_tags[w].upper()
        if tag[:2] not in ('NN', 'VB', 'JJ', 'RB'):
            return []

        # Ignore unsupported words
        degree_of_support = sent.support[w]
        if degree_of_support < 2:
            return []

        implied_tags = sorted(set([tag, tag[:2]])) + [None]
        implied_support = range(2, degree_of_support+1)  + [None]

        occurrence_tuples = itertools.product(implied_tags, implied_support)

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = pos_threshold_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                # Return a boolean for thresholds otherwise the raw degree of
                # support (minus 1 since minimum recorded support is 2)
                feat_value = degree_of_support - 1 \
                                if occurrence[1] is None else 1
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        feat_value if norm is None
                        else feat_value * getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def word_gov_support(cls, instance, word_idxs, pos_threshold_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return the support of the governor of this word.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        if s is None:
            return []

        dparse = instance.input_sents[s].dparse
        if dparse.is_root(w):
            return []

        p = dparse.get_parent_idx(w)
        return cls.word_support(instance, ((s, p),), pos_threshold_indicators,
                normalizers=normalizers, ext_offset=ext_offset, **kwargs)

    @classmethod
    def word_subtree_support(cls, instance, word_idxs,
            pos_threshold_indicators, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return the ratio of support for the subtree of nodes rooted at
        the word.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        if s is None:
            return []

        sent = instance.input_sents[s]
        tag = sent.pos_tags[w].upper()

        implied_tags = sorted(set([tag, tag[:2]])) + [None]
        implied_subtree_support = [] if len(sent.subtree_support[w]) == 0 \
                else range(2, max(sent.subtree_support[w].iterkeys())+1)

        # This includes word w and so is always non-zero
        subtree_size = len(sent.dparse.get_descendant_idxs(w)) + 1

        occurrence_tuples = itertools.product(implied_tags,
                                              implied_subtree_support)

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = pos_threshold_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                # Return the ratio of supporting nodes to non-supporting
                # nodes in the subtree.
                feat_value = sent.subtree_support[w][occurrence[1]] / \
                                         subtree_size
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        feat_value if norm is None
                        else feat_value * getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def ngram_norm(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return 1 for every additional ngram, thereby functioning
        as a sort of bias offset for the learning algorithm.
        """
        feats = []
        for k, norm in enumerate(normalizers):
            feats.append((ext_offset + k,
                1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def ngram_lm(cls, instance, word_idxs, ngram_order=3, **kwargs):
        """Return log-probability under an LM.
        """
        assert len(word_idxs) == ngram_order

        words = []
        for s, w in word_idxs:
            if w == metaidx.START:
                words.append('<s>')
            elif w == metaidx.END:
                words.append('</s>')
            else:
                words.append(instance.input_sents[s].tokens[w])

        # Avoid dying on weird JSON-RPC failures
        num_failures = 0
        while num_failures < failure_limit:
            try:
                return lm_proxy.score_ngram(words)
            except jsonrpc.RPCTransportError:
                num_failures += 1
                if num_failures == failure_limit:
                    raise
                # Try another server
                lm_server = choice(lm_servers)
                print "Got RPCTransportError;",
                print "now using", lm_server, "as an LM server"
                globals()['lm_proxy'] = srilm.LangModel.from_server(lm_server)

    @classmethod
    def ngram_lm_fixed(cls, instance, word_idxs, **kwargs):
        """A version of log-probability under an LM that will not be
        weighted by parameters in order to avoid implicitly imposing
        an exponential weight on n-gram probability.
        """
        return cls.ngram_lm(instance, word_idxs, **kwargs)

    @classmethod
    def ngram_lm_prob(cls, instance, word_idxs, **kwargs):
        """Raw probability under the LM instead of log-prob.
        """
        return np.exp(cls.ngram_lm(instance, word_idxs, **kwargs))

    @classmethod
    def ngram_lm_normed(cls, instance, word_idxs, **kwargs):
        """Normalized log-probability under an LM to be more compatible
        with linear models.
        """
        #lowest_logprob = -227.95592420641054
        lowest_nonoov_logprob = -20

        normalizer = lowest_nonoov_logprob * max(len(input_sent.tokens)
                for input_sent in instance.input_sents)
        return cls.ngram_lm(instance, word_idxs, **kwargs) / normalizer

    @classmethod
    def ngram_fidelity(cls, instance, word_idxs, n=None, ngram_order=3,
            **kwargs):
        """Return 1 if the ngram is in the original sentences, 0 otherwise.
        """
        # Default to the provided ngram order
        if n is None:
            n = ngram_order
        assert len(word_idxs) == n

        maps = instance.input_maps

        for s in instance.original_sent_idxs:
            if word_idxs[0][1] == metaidx.START:
                current = [-1]
            else:
                ss0, ww0 = word_idxs[0]
                current = set(maps[ss0][ww0][s])

            if word_idxs[-1][1] == metaidx.END:
                w_idx_sets = [set(maps[ss][ww][s])
                            for ss, ww in word_idxs[1:-1]] + \
                            [set([instance.sent_lens[s]])]
            else:
                w_idx_sets = [set(maps[ss][ww][s])
                            for ss, ww in word_idxs[1:]]

            for w_idx_set in w_idx_sets:
                incremented = [w + 1 for w in current]
                current = w_idx_set.intersection(incremented)

            if len(current) > 0:
                return 1

        return 0

#            # Return 1 if the words come from the same sentence and the words
#            # are consecutive
#            # TODO: Handle coreference and alignment here
#            for i in range(len(word_idxs) - 1):
#                s0, w0 = word_idxs[i]
#                s1, w1 = word_idxs[i+1]
#                if s0 is not None and s1 is not None and s0 != s1:
#                    return 0
#                if w0 == metaidx.START:
#                    if w1 != 0:
#                        return 0
#                elif w1 == metaidx.END:
#                    if w0 != instance.sent_lens[s0] - 1:
#                        return 0
#                elif w1 != w0 + 1:
#                    return 0
#            return 1

    @classmethod
    def ngram_pair_fidelity(cls, instance, word_idxs, ngram_order=3,
            **kwargs):
        """Return the average fidelity of each word pair in the ngram.
        """
        # No point calculating bigram fidelity again for bigrams
        if ngram_order == 2:
            return 0
        assert len(word_idxs) == ngram_order

        pair_fidelities = []
        for i in range(ngram_order-1):
            pair_idxs = (word_idxs[i], word_idxs[i+1])
            pair_fidelities.append(
                    cls.ngram_fidelity(instance, pair_idxs, n=2, **kwargs))
        return np.mean(pair_fidelities)

    @classmethod
    def ngram_pos_seq(cls, instance, word_idxs, pos_indicators,
            normalizers=(None,), ext_offset=0, ngram_order=3, **kwargs):
        """Return the number of repetitions of the given tag sequence in the
        ngram's part of speech tags. Note that the POS indicators must
        only contain tuples of a fixed size. To test with different-sized
        POS sequences, call this feature multiple times.
        """
        assert len(word_idxs) == ngram_order

        # Note that we assume that all tag indicators are the same size
        seq_size = len(pos_indicators.keys()[0])
        feats = []
        if seq_size > ngram_order:
            return feats

        pos_seq_counts = defaultdict(int)
        for i in range(ngram_order - seq_size + 1):
            partial_pos_seqs = [None] * seq_size
            for j in range(seq_size):
                partial_pos_seqs[j] = []

                # Get implied tags for this sequence position
                s, w = word_idxs[i+j]
                if s is None:
                    tags = [str(w)]
                else:
                    tags = cls.get_implied_tags(
                            instance.input_sents[s].pos_tags[w])

                # Add these new tags to the tags from previous positions
                if j == 0:
                    partial_pos_seqs[j].extend([tag] for tag in tags)
                else:
                    partial_pos_seqs[j].extend(seq + [tag]
                            for seq in partial_pos_seqs[j-1]
                            for tag in tags)

            # Count the final tags
            final_pos_seqs = partial_pos_seqs[-1]
            for final_pos_seq in final_pos_seqs:
                pos_seq_counts[tuple(final_pos_seq)] += 1

        for pos_seq, count in pos_seq_counts.iteritems():
            try:
                j = pos_indicators[pos_seq]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        count if norm is None
                        else count * getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def ngram_label(cls, instance, word_idxs, label_indicators,
            shallowest=1, use_gr=False, normalizers=(None,), ext_offset=0,
            ngram_order=3, **kwargs):
        """Return the number of repetitions of the given label in the
        ngram component words.
        """
        assert len(word_idxs) == ngram_order

        sents = instance.input_sents
        label_counts = defaultdict(int)
        for s, w in word_idxs:
            if s is None:
                continue

            if use_gr:
                for label in sents[s].relgraph.nodes[w].get_incoming_attribs(
                                'label'):
                    label_counts[label] += 1
            else:
                if sents[s].dparse.is_root(w):
                    label_counts['root'] += 1
                else:
                    for label in cls.get_implied_labels(
                            sents[s].dparse.nodes[w].get_incoming_attribs(
                                'label')[0],
                            shallowest=shallowest):
                        label_counts[label] += 1

        feats = []
        for label, count in label_counts.iteritems():
            try:
                j = label_indicators[label]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        count if norm is None
                        else count * getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def ngram_lex(cls, instance, word_idxs, word_indicators, use_stem=False,
            position=None, normalizers=(None,), ext_offset=0, ngram_order=3,
            **kwargs):
        """Return whether a word is present in a particular position
        in the n-gram.
        """
        assert len(word_idxs) == ngram_order
        assert position < ngram_order

        s, w = word_idxs[position]
        if s is None:
            return []

        return cls.word_lex(instance, ((s, w),), word_indicators, [0],
                use_stem=use_stem, normalizers=normalizers,
                ext_offset=ext_offset, **kwargs)

    @classmethod
    def ngram_pair_support(cls, instance, word_idxs, pos_threshold_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether this bigram is supported by bigrams in other
        sentences. This is shown by binary features over whether
        the support exceeds a threshold or a raw occurrence count if
        the given threshold is None.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        if s1 is None:
            # No support defined for END or LEAF
            return []
        if s0 is None:
            w0 = None  # convention for looking up support

        degree_of_support = instance.dep_support[((s0, w0), (s1, w1))]
        if degree_of_support < 2:
            # Ignore unsupported dependencies
            return []

        implied_tags0 = cls.get_implied_tags(
                            instance.input_sents[s0].pos_tags[w0]) \
                            if s0 is not None else []
        implied_tags1 = cls.get_implied_tags(
                            instance.input_sents[s1].pos_tags[w1])
        implied_support = range(2, degree_of_support+1)

        occurrence_tuples = itertools.product(implied_tags0 + [None],
                                              implied_tags1 + [None],
                                              implied_support + [None])

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = pos_threshold_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                # Return a boolean for thresholds otherwise the raw degree of
                # support (minus 1 since minimum recorded support is 2)
                feat_value = degree_of_support - 1 \
                                    if occurrence[2] is None else 1
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        feat_value if norm is None
                        else feat_value * getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def ngram_tok_support(cls, instance, word_idxs, pos_threshold_indicators,
            position=None, normalizers=(None,), ext_offset=0, ngram_order=3,
            **kwargs):
        """Return the support of a word at any position in the n-gram.
        """
        assert len(word_idxs) == ngram_order
        assert position < ngram_order

        s, w = word_idxs[position]
        if s is None:
            return []

        return cls.word_support(instance, ((s, w),), pos_threshold_indicators,
                normalizers=normalizers, ext_offset=ext_offset, **kwargs)

    @classmethod
    def ngram_gov_support(cls, instance, word_idxs, pos_threshold_indicators,
            position=None, normalizers=(None,), ext_offset=0, ngram_order=3,
            **kwargs):
        """Return the support of the governor of a word at any position
        in the n-gram.
        """
        assert len(word_idxs) == ngram_order
        assert position < ngram_order

        s, w = word_idxs[position]
        if s is None:
            return []

        return cls.word_gov_support(instance, ((s, w),),
                pos_threshold_indicators, normalizers=normalizers,
                ext_offset=ext_offset, **kwargs)

    @classmethod
    def ngram_subtree_support(cls, instance, word_idxs,
            pos_threshold_indicators, position=None, normalizers=(None,),
            ext_offset=0, ngram_order=3, **kwargs):
        """Return the support of the input subtree rooted at either
        word of this ngram.
        """
        assert len(word_idxs) == ngram_order
        assert position < ngram_order

        s, w = word_idxs[position]
        if s is None:
            return []

        return cls.word_subtree_support(instance, ((s, w),),
                pos_threshold_indicators, normalizers=normalizers,
                ext_offset=ext_offset, **kwargs)

    @classmethod
    def dep_norm(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return 1 for every additional dependency, thereby functioning
        as a sort of bias offset for the learning algorithm.
        """
        feats = []
        for k, norm in enumerate(normalizers):
            feats.append((ext_offset + k,
                1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def dep_fidelity(cls, instance, word_idxs, label=None, shallowest=1,
            use_gr=False, **kwargs):
        """Return 1 if the dependency is in the original sentences, 0
        otherwise.
        """
        assert 2 <= len(word_idxs) <= 3

        maps = instance.input_maps

        if use_gr:
            # Conceptually, RASP relation graphs are not trees and therefore
            # do not have leaves or roots.
            if word_idxs[0][1] == metaidx.ROOT or \
                    word_idxs[1][1] == metaidx.LEAF:
                return 0

            # Check each input sentence
            for s in instance.original_sent_idxs:
                input_parse = instance.input_sents[s].relgraph
                ss0, ww0 = word_idxs[0]
                ss1, ww1 = word_idxs[1]
                for w0 in maps[ss0][ww0][s]:
                    for w1 in maps[ss1][ww1][s]:
                        if input_parse.has_edge(w0, w1):
                            if label is None or label == \
                                    input_parse.get_edge(w0, w1).label:
                                if len(word_idxs) == 2:
                                    return 1
                                else:
                                    # Get sibling from actual parse tree
                                    input_dparse = \
                                            instance.input_sents[s].dparse
                                    ss2, ww2 = word_idxs[2]
                                    for w2 in maps[ss2][ww2][s]:
                                        if w2 in \
                                        input_dparse.get_elder_sibling(w1):
                                            return 1
        else:
            # Check each input sentence
            for s in instance.original_sent_idxs:
                input_parse = instance.input_sents[s].dparse

                if word_idxs[1][1] == metaidx.LEAF:
                    if label is not None and label.upper() != 'LEAF':
                        continue
                    if len(word_idxs) == 3 and word_idxs[0] != word_idxs[2]:
                        # LEAF node should be an only child
                        continue
                    ss0, ww0 = word_idxs[0]
                    for w in maps[ss0][ww0][s]:
                        if input_parse.is_leaf(w):
                            return 1

                elif word_idxs[0][1] == metaidx.ROOT:
                    if label is not None and label.upper() != 'ROOT':
                        continue
                    if len(word_idxs) == 3 and word_idxs[0] != word_idxs[2]:
                        # ROOT node should only have one child
                        continue
                    ss1, ww1 = word_idxs[1]
                    for w in maps[ss1][ww1][s]:
                        if input_parse.is_root(w):
                            return 1

                else:
                    ss0, ww0 = word_idxs[0]
                    ss1, ww1 = word_idxs[1]
                    for w0 in maps[ss0][ww0][s]:
                        for w1 in maps[ss1][ww1][s]:
                            if input_parse.has_edge(w0, w1):
                                if label is None or \
                                        label in cls.get_implied_labels(
                                        input_parse.get_edge(w0, w1).label,
                                        shallowest=shallowest):
                                    if len(word_idxs) == 2:
                                        return 1
                                    else:
                                        ss2, ww2 = word_idxs[2]
                                        for w2 in maps[ss2][ww2][s]:
                                            if w2 in \
                                            input_parse.get_elder_sibling(w1):
                                                return 1
        return 0

    @classmethod
    def dep_fid_label(cls, instance, word_idxs, label_indicators,
            shallowest=1, use_gr=False, normalizers=(None,), ext_offset=0,
            **kwargs):
        """Return whether the label matches that of the given original
        dependency.
        """
        assert 2 <= len(word_idxs) <= 3

        # No need to check individual labels if the None case fails. This
        # is also where we check sibling match in the case of a second-order
        # dependency.
        if cls.dep_fidelity(instance, word_idxs, label=None,
                shallowest=shallowest, use_gr=use_gr) == 0:
            return []
        occurrence_tuples = set([None])

        if len(label_indicators) > 1 or None not in label_indicators:
            # TODO: is this too slow?
            seen_labels = set([None])
            for label in label_indicators.iterkeys():
                # The indicators might contain other information for
                # second-order dependencies, etc
                if isinstance(label, tuple):
                    label = label[0]

                if label in seen_labels:
                    continue

                # Don't bother checking these labels again
                implied_labels = [label] if use_gr \
                        else cls.get_implied_labels(label,
                                                    shallowest=shallowest)
                seen_labels.update(implied_labels)

                # Only check the label and ignore siblings
                if cls.dep_fidelity(instance,
                                    word_idxs[:2],
                                    label=label,
                                    shallowest=shallowest,
                                    use_gr=use_gr) == 1:
                    occurrence_tuples.update(implied_labels)

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = label_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_label_pos(cls, instance, word_idxs, label_pos_indicators,
            shallowest=1, position=None, normalizers=(None,), ext_offset=0,
            **kwargs):
        """Return whether the labels match the incoming dependencies
        of the governor or dependent in the input dependency tree.
        """
        assert len(word_idxs) == 2
        assert position < 2

        s, w = word_idxs[position]
        if s is None:
            return []

        return cls.word_label_pos(instance, ((s,w),), label_pos_indicators,
                shallowest=shallowest, normalizers=normalizers,
                ext_offset=ext_offset, **kwargs)

    @classmethod
    def dep_pos_seq(cls, instance, word_idxs, pos_indicators, w_offsets,
            position=None, normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the governor or dependent is surrounded by or
        a part of the specified sequence of part of speech tags.
        """
        assert len(word_idxs) == 2
        assert position < 2

        s, w = word_idxs[position]
        if s is None:
            return []

        return cls.word_pos_seq(instance, ((s,w),), pos_indicators,
                w_offsets, normalizers=normalizers, ext_offset=ext_offset,
                **kwargs)

    @classmethod
    def dep_cond_prob(cls, instance, word_idxs,
            # direction_indicators, label_indicators,
            invert_indicators, token_type='word',
            direction=None, ext_offset=0, **kwargs):
        """Score dependencies based on lexical log-probabilities of a token
        given its parent or optionally inverted (parent given child).
        """
        assert len(word_idxs) == 2

        sents = instance.input_sents
        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        word0 = 'ROOT' if w0 == metaidx.ROOT else \
                sents[s0].stems[w0] if token_type == 'stem' else \
                sents[s0].tokens[w0].lower()
        word1 = 'LEAF' if w1 == metaidx.LEAF else \
                sents[s1].stems[w1] if token_type == 'stem' else \
                sents[s1].tokens[w1].lower()

        # If the direction flag is set, the output direction is
        # used for scoring, otherwise we default to the input direction
#        if direction is None:
#            if w0 == metaidx.ROOT:
#                direction = 1
#            elif w1 == metaidx.LEAF:
#                direction = -1
#            elif s0 == s1:
#                direction = int(np.sign(w1 - w0))

        feats = []
        # NOTE: input direction and range indicators do not seem useful
#        for i, input_direction in enumerate(direction_indicators):
#            for label, j in label_indicators.iteritems():
        for k, invert in enumerate(invert_indicators):

            # Avoid dying on weird JSON-RPC failures
            num_failures = 0
            while num_failures < failure_limit:
                try:
                    if invert:
                        score = dep_proxy.get_logprob(parent=word0,
                                given_child=word1,
                                direction=direction,
                                label=None,
                                token_type=token_type)
                    else:
                        score = dep_proxy.get_logprob(child=word1,
                                given_parent=word0,
                                direction=direction,
                                label=None,
                                token_type=token_type)
                    break

                except jsonrpc.RPCTransportError:
                    num_failures += 1
                    if num_failures == failure_limit:
                        raise

                    # Try another server
                    dep_server = choice(dep_servers)
                    print "Got RPCTransportError; now using",
                    print dep_server, "as a dependency server"
                    globals()['dep_proxy'] = \
                            treebank.DependencyModel.from_server(
                                    dep_server)

            feat_offset = ext_offset + k #\
#                    (i * len(label_indicators) *
#                            len(invert_indicators)) + \
#                    (j * len(invert_indicators)) + k
            feats.append((feat_offset, score))
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_dir(cls, instance, word_idxs, direction_indicators,
            direction=None, normalizers=(None,), ext_offset=0, **kwargs):
        """Return the original direction of the dependency and whether it's
        flipped in the output.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]

        # Input direction from the sentences
        if w0 == metaidx.ROOT:
            input_dir = 1
        elif w1 == metaidx.LEAF:
            input_dir = -1
        elif s0 == s1:
            input_dir = np.sign(w1 - w0)
        else:
            input_dir = 0

        # Output direction from the direction flag
        if direction is not None:
            output_dirs = [direction, None]
        else:
            output_dirs = [None]

        occurrence_tuples = itertools.product([input_dir, None],
                                               output_dirs)

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = direction_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_dist(cls, instance, word_idxs, **kwargs):
        """Return the absolute number of tokens spanned by the given
        dependency.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]

        if w0 == metaidx.ROOT or w1 == metaidx.LEAF:
            return 0
        elif s0 == s1:
            return abs(w0 - w1)
        else:
            return instance.avg_len

    @classmethod
    def dep_fid_dir_pos(cls, instance, word_idxs, fid_dir_pos_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and POS tags match those of the given
        dependency.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        sents = instance.input_sents
        fidelity = cls.dep_fidelity(instance, word_idxs, label=None, **kwargs)

        if w0 == metaidx.ROOT:
            gov_tags = ['ROOT']
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            direction = 1  # by convention, this is a right attachment
        elif w1 == metaidx.LEAF:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            dep_tags = ['LEAF']
            direction = -1
        else:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            if s0 == s1:
                direction = np.sign(w1 - w0)
            else:
                direction = 0  # across sentences

        occurrence_tuples = itertools.product([fidelity, None],
                                              [direction, None],
                                              gov_tags + [None],
                                              dep_tags + [None])

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = fid_dir_pos_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_fid_dir_span(cls, instance, word_idxs, fid_dir_span_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and POS tags of the participating
        tokens and spanned tokens match those of the given dependency.
        """
        # TODO: this generalizes over dep_fid_dir_pos() and can be merged
        # with it when new models are trained.
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        sents = instance.input_sents
        fidelity = cls.dep_fidelity(instance, word_idxs, label=None, **kwargs)

        spanned_tags = set()
        if w0 == metaidx.ROOT:
            gov_tags = ['ROOT']
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            direction = 1  # by convention, this is a right attachment
        elif w1 == metaidx.LEAF:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            dep_tags = ['LEAF']
            direction = -1
        else:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            if s0 == s1:
                sent = sents[s0]
                direction = np.sign(w1 - w0)

                for inner_tag in sent.pos_tags[w0+direction:w1:direction]:
                    spanned_tags.update(cls.get_implied_tags(inner_tag))
            else:
                direction = 0  # across sentences

        occurrence_tuples = itertools.product([fidelity, None],
                                              [direction, None],
                                              gov_tags + [None],
                                              dep_tags + [None],
                                              list(spanned_tags) + [None])

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = fid_dir_span_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_anc_dir_pos(cls, instance, word_idxs, anc_dir_pos_indicators,
            assume_intervals=False, normalizers=(None,), ext_offset=0,
            **kwargs):
        """Return whether the direction and POS tags match for an ancestral
        dependency of a particular length.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]

        # Since this is just checking for ancestral DAG links within a
        # sentence, we return nothing if the words come from different
        # sentences.
        # TODO: maybe we should still let this through to match DAG depth
        # 'None' and can then effectively replace dep_dir_pos. For now, we
        # just avoid supplying 'None' as a depth.
        if s0 != s1:
            return []
        sent = instance.input_sents[s0]
        path = sent.dparse.get_path_of_descent(w0, w1)
        path_len = len(path)

        # Assume that the provided DAG arc depths are intervals, so the
        # path length needs to be mapped to the interval that contains it.
        # We don't include zero-length paths (i.e., no ancestry relationship)
        # in this unless explicitly specified in the intervals.
        if assume_intervals and path_len > 0:
            # Collect upper bounds of all intervals
            interval_ubs = set([anc_dir_pos_conf[0]
                for anc_dir_pos_conf in anc_dir_pos_indicators.iterkeys()])
            if None in interval_ubs:
                interval_ubs.remove(None)

            # Find the smallest upper bound that contains the path length.
            # Note that the upper bounds aren't sorted here; we expect to
            # iterate through each.
            new_len = 1000
            for interval_ub in interval_ubs:
                if (interval_ub >= path_len and interval_ub < new_len):
                    new_len = interval_ub

            # Check that the intervals cover all observed path lengths.
            assert new_len != 1000
            path_len = new_len

        if w0 == metaidx.ROOT:
            gov_tags = ['ROOT']
            dep_tags = cls.get_implied_tags(sent.pos_tags[w1])
            direction = 1  # by convention, this is a right attachment
        elif w1 == metaidx.LEAF:
            gov_tags = cls.get_implied_tags(sent.pos_tags[w0])
            dep_tags = ['LEAF']
            direction = -1
        else:
            gov_tags = cls.get_implied_tags(sent.pos_tags[w0])
            dep_tags = cls.get_implied_tags(sent.pos_tags[w1])
            if s0 == s1:
                direction = np.sign(w1 - w0)
            else:
                direction = 0  # across sentences

        occurrence_tuples = itertools.product([path_len, None],
                                              [direction, None],
                                              gov_tags + [None],
                                              dep_tags + [None])
        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = anc_dir_pos_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_fid_dir_lex(cls, instance, word_idxs, fid_dir_lex_indicators,
            use_stem=False, normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and words match those of the given
        dependency.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        sents = instance.input_sents
        fidelity = cls.dep_fidelity(instance, word_idxs, label=None, **kwargs)

        if w0 == metaidx.ROOT:
            direction = 1  # by convention, this is a right attachment
            gov_token = 'ROOT'
            dep_token = sents[s1].stems[w1] if use_stem else \
                        sents[s1].tokens[w1]
        elif w1 == metaidx.LEAF:
            direction = -1
            gov_token = sents[s0].stems[w0] if use_stem else \
                        sents[s0].tokens[w0]
            dep_token = 'LEAF'
        else:
            if s0 == s1:
                direction = np.sign(w1 - w0)
            else:
                direction = 0  # across sentences

            if use_stem:
                gov_token = sents[s0].stems[w0]
                dep_token = sents[s1].stems[w1]
            else:
                gov_token = sents[s0].tokens[w0]
                dep_token = sents[s1].tokens[w1]

        occurrence_tuples = itertools.product([fidelity, None],
                                              [direction, None],
                                              [gov_token, None],
                                              [dep_token, None])

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = fid_dir_lex_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_fid_dir_chk(cls, instance, word_idxs, fid_dir_chk_indicators,
            normalizers=(), ext_offset=0, **kwargs):
        """Return whether the direction and chunk label pair matches those
        of the given dependency.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]

        # We ignore terminal pseudo-words for this feature
        if s0 is None or s1 is None:
            return []

        fidelity = cls.dep_fidelity(instance, word_idxs, label=None, **kwargs)
        if s0 == s1:
            direction = np.sign(w1 - w0)
        else:
            direction = 0  # across sentences

        token_chunks = []
        for s, w in word_idxs:
            token_chunks.append(instance.input_chunks[s][w])
        identical_symb = ['='] if token_chunks[0] == token_chunks[1] else []

        occurrence_tuples = list(itertools.product([fidelity, None],
                                                   [direction, None],
                                                   [token_chunks[0][0]],
                                                   [token_chunks[1][0]] +
                                                        identical_symb))

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = fid_dir_chk_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_support(cls, instance, word_idxs, pos_threshold_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether this dependency is supported by dependencies
        in other sentences. This is shown by binary features over whether
        the support exceeds a threshold or a raw occurrence count if
        the given threshold is None.
        """
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        if s1 is None:
            # No support defined for END or LEAF
            return []
        if s0 is None:
            w0 = None  # convention for looking up support

        degree_of_support = instance.dep_support[((s0, w0), (s1, w1))]
        if degree_of_support < 2:
            # Ignore unsupported dependencies
            return []

        implied_tags0 = cls.get_implied_tags(
                            instance.input_sents[s0].pos_tags[w0]) \
                            if s0 is not None else []
        implied_tags1 = cls.get_implied_tags(
                            instance.input_sents[s1].pos_tags[w1])
        implied_support = range(2, degree_of_support+1)

        occurrence_tuples = itertools.product(implied_tags0 + [None],
                                              implied_tags1 + [None],
                                              implied_support + [None])

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = pos_threshold_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                # Return a boolean for thresholds otherwise the raw degree of
                # support (minus 1 since minimum recorded support is 2)
                feat_value = degree_of_support - 1 \
                                    if occurrence[2] is None else 1
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        feat_value if norm is None
                        else feat_value * getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep_tok_support(cls, instance, word_idxs, pos_threshold_indicators,
            position=None, normalizers=(None,), ext_offset=0, **kwargs):
        """Return the support of the governor or dependent of this dependency.
        """
        assert len(word_idxs) == 2

        # This is operationally identical to bigram support
        if 'ngram_order' in kwargs:
            del kwargs['ngram_order']
        return cls.ngram_tok_support(instance, word_idxs,
                pos_threshold_indicators, position=position,
                normalizers=normalizers, ext_offset=ext_offset,
                ngram_order=2, **kwargs)

    @classmethod
    def dep_gov_support(cls, instance, word_idxs, pos_threshold_indicators,
            position=None, normalizers=(None,), ext_offset=0, **kwargs):
        """Return the support of the input governor for either the
        governor or dependent of this dependency.
        """
        assert len(word_idxs) == 2

        # This is operationally identical to bigram governor support
        if 'ngram_order' in kwargs:
            del kwargs['ngram_order']
        return cls.ngram_gov_support(instance, word_idxs,
                pos_threshold_indicators, position=position,
                normalizers=normalizers, ext_offset=ext_offset,
                ngram_order=2, **kwargs)

    @classmethod
    def dep_subtree_support(cls, instance, word_idxs,
            pos_threshold_indicators, position=None, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return the support of the input subtree rooted at either
        the governor or dependent of this dependency.
        """
        assert len(word_idxs) == 2

        # This is operationally identical to bigram subtree support
        if 'ngram_order' in kwargs:
            del kwargs['ngram_order']
        return cls.ngram_subtree_support(instance, word_idxs,
                pos_threshold_indicators, position=position,
                normalizers=normalizers, ext_offset=ext_offset,
                ngram_order=2, **kwargs)

    @classmethod
    def range_norm(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return 1 for every range variable.
        """
        feats = []
        for k, norm in enumerate(normalizers):
            feats.append((ext_offset + k,
                1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def range_fid_dir_pos(cls, instance, word_idxs, fid_dir_pos_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and POS tags match those of the given
        dependency for a range variable.
        """
        return cls.dep_fid_dir_pos(instance, word_idxs,
                fid_dir_pos_indicators, normalizers=normalizers,
                ext_offset=ext_offset, **kwargs)

    @classmethod
    def range_fid_dir_lex(cls, instance, word_idxs, fid_dir_lex_indicators,
            use_stem=False, normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and words match those of the given
        dependency for a range variable.
        """
        return cls.dep_fid_dir_lex(instance, word_idxs,
                fid_dir_lex_indicators, use_stem=use_stem,
                normalizers=normalizers, ext_offset=ext_offset, **kwargs)

    @classmethod
    def range_fid_dir_chk(cls, instance, word_idxs, fid_dir_chk_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and chunk labels match those of the
        given dependency for a range variable.
        """
        return cls.dep_fid_dir_chk(instance, word_idxs,
                fid_dir_chk_indicators, normalizers=normalizers,
                ext_offset=ext_offset, **kwargs)

    @classmethod
    def arity_norm(cls, instance, word_idxs, normalizers=(None,),
            ext_offset=0, **kwargs):
        """Return 1 for every arity variable.
        """
        feats = []
        for k, norm in enumerate(normalizers):
            feats.append((ext_offset + k,
                1 if norm is None else getattr(instance, norm)))
        return feats

    @classmethod
    def arity_label_pos(cls, instance, word_idxs, arity_label_pos_indicators,
            shallowest=1, normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the input arity, label and POS tag matches that
        of the given token for an arity variable.
        """
        assert len(word_idxs) == 1

        s, w = word_idxs[0]
        sent = instance.input_sents[s]
        pos_tags = cls.get_implied_tags(sent.pos_tags[w])
        parse = sent.dparse

        if parse.is_root(w):
            labels = ['root']
        else:
            node = parse.nodes[w]
            labels = cls.get_implied_labels(
                    node.get_incoming_attribs('label')[0],
                    shallowest=shallowest)

        # Also consider the original number of children
        num_children = parse.get_num_children(w)

        occurrence_tuples = itertools.product([num_children, None],
                                              labels + [None],
                                              pos_tags + [None])
        feats = []
        for occurrence in occurrence_tuples:
            try:
                i = arity_label_pos_indicators[occurrence]
                feat_offset = ext_offset + len(normalizers) * i
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep2_dir_pos_span(cls, instance, word_idxs, dir_pos_span_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and POS tags of the participating
        and spanned tokens match those of the given second-order dependency.
        """
        assert len(word_idxs) == 3

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        s2, w2 = word_idxs[2] # sibling
        sents = instance.input_sents

        gov_sib_tags = set()
        dep_sib_tags = set()
        if w0 == metaidx.ROOT:
            gov_tags = ['ROOT']
            sib_tags = ['ROOT']
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            direction = 1  # by convention, this is a right attachment
        elif w1 == metaidx.LEAF:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            sib_tags = cls.get_implied_tags(sents[s2].pos_tags[w2])
            dep_tags = ['LEAF']
            direction = -1
        else:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            sib_tags = cls.get_implied_tags(sents[s2].pos_tags[w2])
            if s0 == s1:
                sent = sents[s0]
                direction = np.sign(w1 - w0)

                for inner_tag in sent.pos_tags[w0+direction:w2:direction]:
                    gov_sib_tags.update(cls.get_implied_tags(inner_tag))

                for inner_tag in sent.pos_tags[w1+direction:w2:direction]:
                    dep_sib_tags.update(cls.get_implied_tags(inner_tag))
            else:
                direction = 0  # across sentences

        occurrence_tuples = itertools.product([direction, None],
                                              gov_tags + [None],
                                              list(gov_sib_tags) + [None],
                                              sib_tags + [None],
                                              list(dep_sib_tags) + [None],
                                              dep_tags + [None])

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = dir_pos_span_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def dep2_dir_pos_dist(cls, instance, word_idxs, dir_pos_dist_indicators,
            normalizers=(None,), ext_offset=0, **kwargs):
        """Return whether the direction and POS tags of the participating
        and spanned tokens match those of the given second-order dependency.
        """
        assert len(word_idxs) == 3

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        s2, w2 = word_idxs[2] # sibling
        sents = instance.input_sents

        gov_sib_distance = [None]
        dep_sib_distance = [None]
        if w0 == metaidx.ROOT:
            gov_tags = ['ROOT']
            sib_tags = ['ROOT']
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            direction = 1  # by convention, this is a right attachment
        elif w1 == metaidx.LEAF:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            sib_tags = cls.get_implied_tags(sents[s2].pos_tags[w2])
            dep_tags = ['LEAF']
            direction = -1
        else:
            gov_tags = cls.get_implied_tags(sents[s0].pos_tags[w0])
            dep_tags = cls.get_implied_tags(sents[s1].pos_tags[w1])
            sib_tags = cls.get_implied_tags(sents[s2].pos_tags[w2])
            if s0 == s1:
                direction = np.sign(w1 - w0)
                gov_sib_distance.append(abs(w2 - w0))
                dep_sib_distance.append(abs(w2 - w1))
            else:
                direction = 0  # across sentences

        occurrence_tuples = itertools.product([direction, None],
                                              gov_tags + [None],
                                              gov_sib_distance,
                                              sib_tags + [None],
                                              dep_sib_distance,
                                              dep_tags + [None])

        feats = []
        for occurrence in occurrence_tuples:
            try:
                j = dir_pos_dist_indicators[occurrence]
                feat_offset = ext_offset + (j * len(normalizers))
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def frame_name(cls, instance, word_idxs, frame_indicators,
            ancestor_limit=0, normalizers=(), ext_offset=0, **kwargs):
        """Return whether the frame name matches those of the given entries
        from Framenet.
        """
        assert len(word_idxs) == 1

        s, f = word_idxs[0]
        frame = instance.input_sents[s].frames.nodes[f].name

        occurrence_tuples = [None, frame]
        if ancestor_limit > 0:
            occurrence_tuples.extend(framenet.get_frame_ancestors(
                                     frame, limit=ancestor_limit))

        feats = []
        for occurrence in occurrence_tuples:
            try:
                i = frame_indicators[occurrence]
                feat_offset = ext_offset + len(normalizers) * i
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))

    @classmethod
    def fe_frame_pos_dep(cls, instance, word_idxs,
            frame_fe_pos_dep_indicators, normalizers=(), ext_offset=0,
            **kwargs):
        """Return whether the frame element label, frame name, core type,
        POS tag of the head word of the frame element, POS tag of the
        head word of the frame target and dependency label, if any, between
        the two match those of the given frame element.
        """
        assert len(word_idxs) == 2

        s0, f = word_idxs[0]
        s1, w = word_idxs[1]
        assert s0 == s1     # unless this is a MultiSentence perhaps?

        # Always consider the frame and the POS tagso of the word
        frame_node = instance.input_sents[s0].frames.nodes[f]
        pos_tags = cls.get_implied_tags(instance.input_sents[s1].pos_tags[w])

        # Check whether the word is in the target of the frame
        in_tgt = w in frame_node.tgt_idxs

        # Check whether the word is a lexicalization of any FE in the input
        word_node = instance.input_sents[s1].frames.nodes[w]
        in_lex = len(word_node.incoming_edges) > 0

        # If this frame element is in the input, record its label and coretype
        fes = []
        coretypes = []
        if w in frame_node.outgoing_edges and \
                hasattr(frame_node.outgoing_edges[w], 'fe'):
            fe = frame_node.outgoing_edges[w].fe
            fes = [fe]
            coretypes = [framenet.get_coretype(frame_node.name, fe)]

        # If there's a dependency relationship between the given word and
        # the target phrase of the frame, record its label and the POS tag
        # of the target word.
        w_node = instance.input_sents[s1].dparse.nodes[w]
        in_dep = False
        labels = []
        gov_tags = []
        for tgt_w in frame_node.tgt_idxs:
            if s0 != s1:
                break
            if tgt_w not in w_node.incoming_edges:
                continue

            in_dep = True
            dep_edge = w_node.incoming_edges[tgt_w]
            labels = cls.get_implied_labels(dep_edge.label, shallowest=4)
            gov_tags = cls.get_implied_tags(
                    instance.input_sents[s0].pos_tags[tgt_w])
            break

        occurrence_tuples = itertools.product([in_tgt, None],
                                              [in_lex, None],
                                              [in_dep, None],
                                              [frame_node.name, None],
                                              fes + [None],
                                              coretypes + [None],
                                              pos_tags + [None],
                                              gov_tags + [None],
                                              labels + [None],
                                              )

        feats = []
        for occurrence in occurrence_tuples:
            try:
                i = frame_fe_pos_dep_indicators[occurrence]
                feat_offset = ext_offset + (len(normalizers) * i)
                for k, norm in enumerate(normalizers):
                    feats.append((feat_offset + k,
                        1 if norm is None
                        else getattr(instance, norm)))
            except KeyError:
                pass
        return sorted(feats, key=itemgetter(0))
