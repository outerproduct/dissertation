#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
import itertools
from lexical import matcher


class SupportClusters(object):
    """A class to identify groups of similar words across sentences.
    """
    prefixes = set(('NN', 'VB', 'JJ', 'RB'))

    @classmethod
    def cluster_words(cls, input_sents):
        """Identify the content words which are repeated across sentences
        and note the number of occurrences.
        """
        #.TODO: extend to paraphrases?

        # clusters[token] = set(word_idxs)
        clusters = defaultdict(set)

        # merged[token] = [token variants]
        merged = {}

        for s0 in range(len(input_sents) - 1):
            sent0 = input_sents[s0]
            for t0, token0 in enumerate(sent0.tokens):
                if sent0.pos_tags[t0][:2] not in cls.prefixes:
                    continue

                for s1 in range(s0 + 1, len(input_sents)):
                    sent1 = input_sents[s1]
                    for t1, token1 in enumerate(sent1.tokens):
                        if sent1.pos_tags[t1][:2] not in cls.prefixes:
                            continue

                        if matcher.Matcher.check(t0, sent0, t1, sent1):
                            # Collect all implicitly involved tokens
                            token_set, idx_set = set(), set()

                            cls.pop_cluster(token0, clusters, merged,
                                            token_set, idx_set)
                            token_set.add(token0)
                            idx_set.add((s0,t0))
                            idx_set.add((s1,t1))

                            if token0 != token1:
                                cls.pop_cluster(token1, clusters, merged,
                                                token_set, idx_set)
                                token_set.add(token1)

                            # Add new cluster
                            token_list = sorted(token_set)
                            for token in token_list:
                                merged[token] = token_list
                            clusters['/'.join(token_list)] = idx_set

        return clusters

    @classmethod
    def pop_cluster(cls, key, clusters, merged_keys, output_keys, output_idxs):
        """For a given key (e.g., a token), remove its merged variants and
        mention indices from the current clustering and add them to the
        provided output sets.
        """
        if key in merged_keys:
            # Multiple keys point to the same cluster
            key_list = merged_keys[key]
            output_keys.update(key_list)

            keystring = '/'.join(sorted(key_list))
            output_idxs.update(clusters[keystring])
            del clusters[keystring]

        elif key in clusters:
            output_keys.add(key)
            output_idxs.update(clusters[key])
            del clusters[key]

    @classmethod
    def cluster_word_pairs(cls, input_sents, syntactic=False):
        """Identify pairs of words -- one of which must be a content word --
        which are repeated across sentences and note the number of occurrences.
        The pairs can take the form of bigrams or dependencies.
        """
        # clusters[token_pair] = set(pair_idxs)
        clusters = defaultdict(set)

        # merged[token_pair] = [token_pair variants]
        merged = {}

        for s0 in range(len(input_sents) - 1):
            sent0 = input_sents[s0]
            for t0, token0 in enumerate(sent0.tokens):

                u0 = cls.get_companion_idx(t0, sent0, syntactic=syntactic)
                if sent0.pos_tags[t0][:2] not in cls.prefixes and \
                        (u0 is None or \
                         sent0.pos_tags[u0][:2] not in cls.prefixes):
                    continue

                for s1 in range(s0 + 1, len(input_sents)):
                    sent1 = input_sents[s1]
                    for t1, token1 in enumerate(sent1.tokens):
                        # NOTE: we're ignoring END when syntactic=False
                        # because we expect it to usually be punctuation and
                        # therefore violate the prefix filter.
                        u1 = cls.get_companion_idx(t1, sent1,
                                                   syntactic=syntactic)
                        if sent1.pos_tags[t1][:2] not in cls.prefixes and \
                                (u1 is None or \
                                 sent1.pos_tags[u1][:2] not in cls.prefixes):
                            continue

                        if matcher.Matcher.check(t0, sent0, t1, sent1) and \
                                ((u0 is None and u1 is None) or
                                 (u0 is not None and u1 is not None and
                                 matcher.Matcher.check(u0, sent0, u1, sent1))):
                            # Collect all implicitly involved pairs
                            pair_set, idx_set = set(), set()

                            if u0 is None:
                                preceding0 = preceding1 = \
                                        'ROOT' if syntactic else 'START'
                            else:
                                preceding0 = sent0.tokens[u0]
                                preceding1 = sent1.tokens[u1]

                            pair0 = '_'.join((preceding0, token0))
                            pair1 = '_'.join((preceding1, token1))

                            cls.pop_cluster(pair0, clusters, merged,
                                            pair_set, idx_set)
                            pair_set.add(pair0)
                            idx_set.add((s0,u0,t0))
                            idx_set.add((s1,u1,t1))

                            if pair0 != pair1:
                                cls.pop_cluster(pair1, clusters, merged,
                                                pair_set, idx_set)
                                pair_set.add(pair1)

                            # Add new cluster
                            pair_list = sorted(pair_set)
                            for pair in pair_list:
                                merged[pair] = pair_list
                            clusters['/'.join(pair_list)] = idx_set

        return clusters

    @classmethod
    def get_companion_idx(cls, t, sent, syntactic=False):
        """Return the companion of the given token idx. If we're considering
        dependency pairs, the companion is its governor; otherwise, we're
        considering bigrams and the companion is the preceding token.
        """
        if syntactic:
            return sent.dparse.get_parent_idx(t) \
                    if not sent.dparse.is_root(t) else None
        else:
            return t - 1 if t != 0 else None

    @classmethod
    def record_support(cls, input_sents, support_clusters=None):
        """Record the support and bottom up support for each word, bigram
        and dependency.
        """
        if support_clusters is None:
            support_clusters = cls.cluster_words(input_sents)

        for sent in input_sents:
            # support[w] = degree_of_support
            sent.support = []

            # subtree_support[w][degree_of_support] = count
            sent.subtree_support = [defaultdict(int)
                                        for w in range(len(sent.tokens))]

            for t, token in enumerate(sent.tokens):
                # The degree of support is the number of sentences that
                # contain the term
                degree_of_support = len(set([ss for ss, tt in
                                             support_clusters[token]]))

                sent.support.append(degree_of_support)
#                sent.ancestral_support[w][degree_of_support] = 1
                sent.subtree_support[w][degree_of_support] += 1

                for idx in sent.dparse.get_ancestor_idxs(w):
                    sent.subtree_support[idx][degree_of_support] += 1

    @classmethod
    def get_expanded_support(cls, input_sents, input_maps,
            support_clusters=None, syntactic=False):
        """Generalize the tokens participating in clustered pairs to
        all tokens that they map to in the input setences. This permits,
        for instance, the support of a hypothetical dependency that
        spans input sentences to be recorded when it can be mapped to
        a supported dependency from a single input sentence.
        """
        if support_clusters is None:
            support_clusters = cls.cluster_word_pairs(input_sents,
                                                      syntactic=syntactic)

        # expanded_clusters[((s0, w0), (s1, w1))] = degree_of_support
        expanded_clusters = defaultdict(int)

        # For each cluster
        for pair_idxs in support_clusters.itervalues():
            # The degree of support is the number of input sentences that
            # contain the bigram or dependency
            degree_of_support = len(set([s for s, t0, t1 in pair_idxs]))

            # For each input instantiation of the bigram or dependency
            for s, t0, t1 in pair_idxs:
                t0_idxs, t1_idxs = set(), set()
                if t0 is None:
                    # Add an indicator that the first token is special
                    t0_idxs.add((None,None))
                else:
                    # Add this token and all mapped tokens
                    t0_idxs.add((s,t0))
                    if t0 in input_maps[s]:
                        for ss, tt_list in input_maps[s][t0].iteritems():
                            for tt in tt_list:
                                t0_idxs.add((ss, tt))

                # Add this token and all mapped tokens
                t1_idxs.add((s,t1))
                if t1 in input_maps[s]:
                    for ss, tt_list in input_maps[s][t1].iteritems():
                        for tt in tt_list:
                            t1_idxs.add((ss, tt))

                for t0_idx, t1_idx in itertools.product(t0_idxs, t1_idxs):
                    expanded_clusters[(t0_idx, t1_idx)] = degree_of_support

        return expanded_clusters

    @classmethod
    def record_pair_support(cls, input_sents, support_clusters=None,
            syntactic=False):
        """Record the support for each bigram or dependency in the
        input sentence.
        """
        if support_clusters is None:
            support_clusters = cls.cluster_word_pairs(input_sents,
                                                      syntactic=syntactic)

        name = 'dep_support' if syntactic else 'bigram_support'
        for s, sent in enumerate(input_sents):

            # bigram/dep_support[(u0,t0)] = degree_of_support
            setattr(sent, name, {})
            support = getattr(sent, name)

            for pair_idxs in support_clusters.itervalues():
                # The degree of support is the number of sentences that
                # contain the bigram or dependency
                degree_of_support = len(set([ss for ss, uu, tt in pair_idxs]))
                support.update(dict(((uu, tt), degree_of_support)
                                    for ss, uu, tt, in pair_idxs
                                    if ss == s))

    @classmethod
    def cluster_words_old(cls, input_sents):
        """DEPRECATED: see new version
        """
        #.TODO: extend to paraphrases?

        # clusters[token] = set(word_idxs)
        clusters = defaultdict(set)

        for s0 in range(len(input_sents) - 1):
            sent0 = input_sents[s0]
            for t0, token0 in enumerate(sent0.tokens):
                if sent0.pos_tags[t0][:2] not in cls.prefixes:
                    continue
                for s1 in range(s0 + 1, len(input_sents)):
                    sent1 = input_sents[s1]
                    for t1, token1 in enumerate(sent1.tokens):
                        if sent1.pos_tags[t1][:2] not in cls.prefixes:
                            continue
                        if matcher.Matcher.check(t0, sent0, t1, sent1):
                            token0 = token0.lower()
                            token1 = token1.lower()

                            clusters[token0].add((s0,t0))
                            clusters[token0].add((s1,t1))
                            if token0 != token1:
                                # We assume transitivity in the matcher,
                                # which is probably not a great idea
                                #self.support[token1].add((s0,t0))
                                #self.support[token1].add((s1,t1))
                                clusters[token1].update(clusters[token0])
                                clusters[token0].update(clusters[token1])
        return clusters
