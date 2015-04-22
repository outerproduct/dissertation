#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import numpy as np


class TokenVar(object):
    """A variable representing a single token in the input.
    """
    # No internal dict, can't be pickled, does not support weak references,
    # among other things.
    __slots__ = ['idx', 'pos', 'feat_vector', 'lagrange_mult', 'score']
    def __init__(self, idx, pos, instance, feats):
        """Initialize features.
        """
        self.idx = idx  # Index in the sentence (s, w)
        self.pos = pos  # Position in the list of token variables
        self.feat_vector = feats.get_feat_vector(instance, (idx,), 'word')
        self.lagrange_mult = 0
        self.score = 0

    def get_combined_score(self):
        """Return the score with the Lagrange multiplier.
        """
        return self.score + self.lagrange_mult


class TokenDP(object):
    """A class to decode the best tokens within a compression rate by sorting.
    """
    def __init__(self, instance, feats, var_conf, constraint_conf, **kwargs):
        """Initialize the token variables.
        """
        # Offsets for multiple input sentences so that we can map
        # word indices (s,w) to their Lagrange multiplier.
        self.lagrange_offsets = [0]
        for input_sent in instance.input_sents:
            self.lagrange_offsets.append(len(input_sent.tokens) +
                                         sum(self.lagrange_offsets))

        self.feats = feats

        self.token_vars = []
        for s, input_sent in enumerate(instance.input_sents):
            for w in range(len(input_sent.tokens)):
                self.token_vars.append(TokenVar((s, w), len(self.token_vars),
                                                instance, feats))

        self.num_tokens = sum(len(input_sent.tokens)
                                for input_sent in instance.input_sents)
        self.num_output_tokens = instance.get_compressed_len(constraint_conf)

    def update(self, weights, first_call=False):
        """Set scores for the variables with the current weights.
        """
        # Sanitize weights to support the use of 'fixed' features which are
        # not optimized over
        weights = self.feats.sanitize_weights(weights)

        for var in self.token_vars:
            # If this is the first call for this variable,
            # standardize the feature vector.
            if first_call and self.feats.standardize:
                var.feat_vector = self.feats.standardize_feat_vector(
                        var.feat_vector)
            score = self.feats.get_score(var.feat_vector, weights)
            if np.isnan(score):
                print 'w', weights
                print 'f', var.feat_vector
            var.score = score

    def update_lagrange(self, lagrange_mults):
        """Set coefficients for the word variables based on the
        Lagrange multipliers.
        """
        for token_var, lagrange_mult in zip(self.token_vars, lagrange_mults):
            token_var.lagrange_mult = lagrange_mult

    def solve(self, **kwargs):
        """Find the highest scoring token sequence.
        """
        sorted_vars = sorted(self.token_vars,
                             key=lambda x: x.get_combined_score(),
                             reverse=True)

        self.output_tokens = []
        self.score = 0
        self.raw_score = 0  # score without Lagrange multipliers
        for var in sorted_vars[:self.num_output_tokens]:
            self.output_tokens.append(var.pos)
            self.score += var.get_combined_score()
            self.raw_score += var.score

        assert len(self.output_tokens) == self.num_output_tokens

        return True

    def has_solution(self):
        """Return whether the decoder was run and has a stored solution.
        """
        return hasattr(self, 'output_tokens')

    def get_solution(self, instance):
        """Return the last solution as an ordered list of word indices.
        """
        return [self.token_vars[idx].idx for idx in self.output_tokens]

    def rescore(self, word_idxs):
        """Rescore a solution given its indices.
        """
        score = 0
        i = 0
        for token_var in self.token_vars:
            if i >= len(word_idxs):
                break
            if token_var.idx == word_idxs[i]:
                score += token_var.score
                i += 1
        return score

    def get_solution_feats(self):
        """Return the sum of the feature vectors from the solution variables.
        """
        feats_from_solution = [self.token_vars[idx].feat_vector
                                for idx in self.output_tokens]
        return self.feats.sum_feat_values(feats_from_solution)

    def get_solution_words(self):
        """Return the value of each word in the current solution.
        """
        words = np.zeros(len(self.token_vars))
        words[self.output_tokens] = 1
        return words
