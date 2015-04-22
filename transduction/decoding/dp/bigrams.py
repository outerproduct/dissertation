#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import numpy as np
from transduction.decoding.dp import mstwrapper
from transduction.model import metaidx


class TokenVar(object):
    """A variable representing a single token in the input.
    """
    # No internal dict, can't be pickled, does not support weak references,
    # among other things.
    __slots__ = ['idx', 'feat_vector', 'lagrange_mult', 'score', 'psi']
    def __init__(self, idx, instance, feats, psi=1):
        """Initialize features.
        """
        self.idx = idx
        self.feat_vector = feats.get_feat_vector(instance, (idx,), 'word')
        self.lagrange_mult = 0
        self.score = 0
        self.psi = psi


class BigramVar(object):
    """A pair of tokens in the input representing a bigram.
    """
    # No internal dict, can't be pickled, does not support weak references,
    # among other things.
    __slots__ = ['idx0', 'idx1', 'token0', 'token1', 'feat_vector', 'score']
    def __init__(self, t0_var, t1_var, instance, feats):
        """Initialize features for the bigram.
        """
        if t0_var in metaidx:
            self.idx0 = (None, t0_var)
        else:
            self.token0 = t0_var
            self.idx0 = t0_var.idx

        if t1_var in metaidx:
            self.idx1 = (None, t1_var)
        else:
            self.token1 = t1_var
            self.idx1 = t1_var.idx

        self.feat_vector = feats.get_feat_vector(
                instance,
                (self.idx0, self.idx1),
                'ngram',
                ngram_order=2)

        # Basic scores per iteration
        self.score = 0

    def get_combined_score(self, with_lagrange=True):
        """Return the score of this bigram variable added to the score of
        the terminal token in the bigram (assuming it's not END).
        """
        if self.idx1[1] == metaidx.END:
            return self.score
        else:
            return self.score + (self.token1.psi * self.token1.score) + \
                    (self.token1.lagrange_mult if with_lagrange else 0)


class BigramDP(object):
    """A class to decode bigram-based compressions following the dynamic
    program of McDonald (2006).
    """
    def __init__(self, instance, feats, var_conf, constraint_conf,
            ngram_order=2, sanity_check=False, psi=1, **kwargs):
        """Initialize the variables and the dynamic programming table.
        """
        assert ngram_order == 2

        # Offsets for multiple input sentences so that we can map
        # word indices (s,w) to their Lagrange multiplier.
        self.lagrange_offsets = [0]
        for input_sent in instance.input_sents:
            self.lagrange_offsets.append(len(input_sent.tokens) +
                                         sum(self.lagrange_offsets))

        self.feats = feats
        # TODO: is var_flags necessary?
        #self.var_flags = \
        #        variables.TransductionVariables.parse_var_conf(var_conf)

        self.init_variables(instance, feats, psi=psi)
        self.init_tables(instance, constraint_conf)

        # Initialize the MST wrapper
        if 'dep' in feats.categories:
            # NOTE: this is kept around for MSTwrapper
            self.instance = instance

        # XXX As a sanity check, also run the equivalent ILP
        self.sanity_check = sanity_check
        if self.sanity_check:
            from transduction.decoding import ilp
            self.ilp_decoder = ilp.TransductionILP(
                    instance, feats, var_conf, constraint_conf,
                    ngram_order=ngram_order, max_flow=100)

    def init_variables(self, instance, feats, psi=1):
        """Initialize the variables and generate features.
        """
        self.token_vars = [TokenVar((s, w), instance, feats, psi=psi)
                           for s, input_sent in enumerate(instance.input_sents)
                           for w in range(len(input_sent.tokens))]

        self.start_vars, self.end_vars, self.bigram_vars = [], [], []

        for j, token1_var in enumerate(self.token_vars):
            self.start_vars.append(
                    BigramVar(metaidx.START, token1_var, instance, feats))
            self.end_vars.append(
                    BigramVar(token1_var, metaidx.END, instance, feats))
            for i in range(j):
                token0_var = self.token_vars[i]
                self.bigram_vars.append(
                        BigramVar(token0_var, token1_var, instance, feats))

        # XXX A hacky dict for easy rescoring
        self.vardict = {}
        for var_type in ('start', 'end', 'bigram'):
            for var in getattr(self, var_type + '_vars'):
                self.vardict[(var.idx0, var.idx1)] = var
        for var in self.token_vars:
            self.vardict[var.idx] = var

    def init_tables(self, instance, constraint_conf):
        """Build the tables to score partial sequences and record backpointers.
        """
        self.num_tokens = sum(len(input_sent.tokens)
                                for input_sent in instance.input_sents)
        self.num_output_tokens = instance.get_compressed_len(constraint_conf)

        self.subseq_scores = -np.inf * np.ones([self.num_tokens,
                                        self.num_output_tokens])
        self.token_backptrs = np.zeros([self.num_tokens,
                                        self.num_output_tokens],
                                        dtype=np.uint8)
        self.bigram_backptrs = np.zeros([self.num_tokens,
                                        self.num_output_tokens],
                                        dtype=np.uint8)

    def update(self, weights, first_call=False):
        """Set scores for the variables with the current weights.
        """
        # Sanitize weights to support the use of 'fixed' features which are
        # not optimized over
        weights = self.feats.sanitize_weights(weights)

        for var_list_name in ('token_vars', 'start_vars', 'end_vars',
                'bigram_vars'):
            for var in getattr(self, var_list_name):
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
                if hasattr(var, 'lagrange_mult'):
                    var.lagrange_mult = 0

        if 'dep' in self.feats.categories:
            if hasattr(self, 'mst_wrapper'):
                print "MSTWrapper weights can't be updated; rebuilding"
            # Note that we use (1-psi) for token scoring in the dependency
            # tree sub-problem
            self.mst_wrapper = mstwrapper.MSTWrapper(self.instance,
                    self.feats, weights, self.token_vars,
                    get_idx=lambda x:x.idx,
                    get_score=lambda x:(1-x.psi)*x.score)

        # XXX
        if self.sanity_check:
            self.ilp_decoder.update(weights, first_call=first_call)

    def update_lagrange(self, lagrange_mults):
        """Set coefficients for the word variables based on the
        Lagrange multipliers.
        """
        for token_var, lagrange_mult in zip(self.token_vars, lagrange_mults):
            token_var.lagrange_mult = lagrange_mult

    def solve(self, **kwargs):
        """Solve the dynamic program.
        """
        self.subseq_scores[:,0] = [start_var.get_combined_score()
                                   for start_var in self.start_vars]

        # For efficiency
        bigram_scores = [var.get_combined_score() for var in self.bigram_vars]

        for r in range(1, self.num_output_tokens):
            b = 0  # Index into bigram variables

            for j in range(self.num_tokens):
                # Don't need to consider any word with index j < r
                # because it's impossible to reach it with r tokens
                if j < r:
                    b += j
                    continue

                # Score all partial sequences terminating in token j
                incoming_scores = self.subseq_scores[:j,r-1] + \
                                    bigram_scores[b:b+j]
#                for x in range(b,b+j):
#                    assert self.bigram_vars[x].idx1[1] == j
                if r == self.num_output_tokens - 1:
                    incoming_scores += self.end_vars[j].get_combined_score()

#                if len(incoming_scores) == 0:
#                    continue

                best_incoming_idx = incoming_scores.argmax()
                self.subseq_scores[j,r] = incoming_scores[best_incoming_idx]
                self.token_backptrs[j,r] = best_incoming_idx
                self.bigram_backptrs[j,r] = best_incoming_idx + b

                b += j

            # Check that all bigram variables were seen in each round
            assert b == len(self.bigram_vars)

        # Recover best token and bigram variable sequences by following
        # the backpointers
        self.output_tokens = [None for r in range(self.num_output_tokens)]
        self.output_bigrams = self.output_tokens[:-1]  # implicit list copy

        current_idx = self.subseq_scores[:,-1].argmax()
        self.score = self.subseq_scores[current_idx,-1]

        # The raw score is the score without Lagrange multipliers
        self.raw_score = 0

        for r in range(self.num_output_tokens - 1, -1, -1):
            self.output_tokens[r] = current_idx
            if r > 0:
                current_idx = self.token_backptrs[current_idx,r]
                bigram_idx = self.bigram_backptrs[current_idx,r]
                self.output_bigrams[r-1] = bigram_idx
                self.raw_score += \
                        self.bigram_vars[bigram_idx].get_combined_score(
                                                        with_lagrange=False)

        # XXX
        if self.sanity_check:
            self.ilp_decoder.solve(**kwargs)

        return True

    def has_solution(self):
        """Return whether the decoder was run and has a stored solution.
        """
        return hasattr(self, 'output_tokens')

    def get_solution(self, instance):
        """Return the last solution as an ordered list of word indices.
        """
        word_idxs = [self.token_vars[idx].idx for idx in self.output_tokens]

        # XXX
        if self.sanity_check:
            ilp_idxs = self.ilp_decoder.get_solution(instance)
            if word_idxs != ilp_idxs:
                assert self.ilp_decoder.get_score() == self.rescore(ilp_idxs)
                assert self.score == self.rescore(word_idxs)
                print "ILP", ilp_idxs, "score =", self.ilp_decoder.get_score()
                print "DP", word_idxs, "score =", self.score
                print

        return word_idxs

    def rescore(self, word_idxs):
        """Rescore a solution given its indices.
        """
        score = sum(self.vardict[(idx0, idx1)].get_combined_score(
                                                        with_lagrange=False)
                for idx0, idx1 in zip([(None, metaidx.START)] + word_idxs,
                    word_idxs + [(None, metaidx.END)]))
#        score = sum(self.vardict[(idx0,idx1)].score
#                for idx0, idx1 in zip(word_idxs[:-1],word_idxs[1:])) + \
#                self.vardict[((None, metaidx.START), word_idxs[0])].score + \
#                self.vardict[(word_idxs[-1], (None, metaidx.END))].score + \
#                sum(self.vardict[idx].score * self.vardict[idx].psi
#                        for idx in word_idxs)
        return score

    def get_solution_feats(self):
        """Return the sum of the feature vectors from the solution variables.
        """
        feats_from_solution = \
                [self.token_vars[idx].feat_vector
                        for idx in self.output_tokens] + \
                [self.bigram_vars[idx].feat_vector
                        for idx in self.output_bigrams] + \
                [self.start_vars[self.output_tokens[0]].feat_vector] + \
                [self.end_vars[self.output_tokens[-1]].feat_vector]

        return self.feats.sum_feat_values(feats_from_solution)

    def get_solution_words(self):
        """Return the value of each word in the current solution.
        """
        words = np.zeros(len(self.token_vars))
        words[self.output_tokens] = 1
        return words
