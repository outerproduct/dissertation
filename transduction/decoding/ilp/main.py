#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from interfaces import lpsolver
import numpy as np
from text import structure
import variables, constraints


class TransductionILP(object):
    """An integer linear program that can solve a joint transduction problem.
    """
    def __init__(self, instance, feats, var_conf, constraint_conf,
            ngram_order=2, max_flow=100):
        """Initialize LP variables and constraints.
        """
        self.lp = lpsolver.LinearProgram(maximization=True)
        self.feats = feats
        self.ngram_order = ngram_order

        self.var_flags = \
                variables.TransductionVariables.parse_var_conf(var_conf)
        if 'range' in feats.categories and not self.var_flags['directional']:
            print "WARNING: directionality required for range features"
            self.var_flags['directional'] = True

        # Add variables for words and structure
        variables.TransductionVariables.add_all(
                self.lp,
                feats,
                instance,
                ngram_order=ngram_order,
                **self.var_flags)

        # Add structural constraints
        constraints.TransductionConstraints.enforce_structure(
                self.lp,
                instance,
                feats.categories,
                ngram_order=ngram_order,
                max_flow=max_flow,
                **self.var_flags)

        # Add other constraints (NOTE: may include additional variables)
        for constraint_group in constraint_conf:
            constraints.TransductionConstraints.add(
                    constraint_group,
                    self.lp,
                    instance)

    def update(self, weights, first_call=False):
        """Set coefficients for the variables with the current weights.
        """
        # Sanitize weights to support the use of 'fixed' features which are
        # not optimized over
        orig_weights = weights
        weights = self.feats.sanitize_weights(weights)

        for feat_cat in self.feats.categories:
            var_type = feat_cat.upper()
            if var_type not in self.lp.variables:
                # FRAME and FE variables may be missing in some instances
                continue

            for var in self.lp.retrieve_all_variables(var_type):
                # If this is the first call for this variable,
                # standardize the feature vector.
                if first_call and self.feats.standardize:
                    var.metadata = self.feats.standardize_feat_vector(
                            var.metadata)
                score = self.feats.get_score(var.metadata, weights)
                if np.isnan(score):
                    print 'w', weights
                    print 'w_orig', orig_weights
                    print 'f', var.metadata
                self.lp.update_variable_coeff(score, var_type,
                                              **var.grounding)

    def solve(self, **kwargs):
        """Solve the LP and return whether a solution was found in the
        given timeframe.
        """
        self.lp.solve(rebuild=False,
                      **kwargs)
        return self.has_solution()

    def has_solution(self):
        """Return whether the last LP solving attempt was successful.
        """
        return self.lp.obj_value is not None

    def get_score(self):
        """Return the score of the solution.
        """
        return self.lp.obj_value

    def has_integral_solution(self, ndigits=3):
        """Return whether the last LP solving attempt was successful and
        produced an integral solution (up to some significance threshold).
        """
        if not self.has_solution():
            return False

        for feat_cat in self.feats.categories:
            var_type = feat_cat.upper()
            for var in self.lp.retrieve_all_variables(var_type):
                if not self.lp.get_value(var, ndigits=ndigits).is_integer():
                    return False

        return True

    def get_integrality(self, ndigits=3):
        """Return a dictionary that notes the number of non-integral variables
        of each type.
        """
        if not self.has_solution():
            return None

        integrality = {}
        for feat_cat in self.feats.categories:
            integrality[feat_cat] = [[], []]
            var_type = feat_cat.upper()

            for var in self.lp.retrieve_all_variables(var_type):
                value = self.lp.get_value(var, ndigits=ndigits)
                if value == 0:
                    continue
                integrality[feat_cat][int(value.is_integer())].append(
                        (var, value))

        return integrality

    def get_solution(self, instance):
        """Return the last solution as an ordered list of word indices.
        """
        if 'ngram' not in self.feats.categories:
            word_idxs = self.get_active_word_idxs(ordered=False)
        else:
            # TODO: as a sanity check, compare the words from the tree against
            # the words from the sentence
            word_idxs = self.get_active_word_idxs(ordered=True)

            if word_idxs is None:
                print "Failed ordered retrieval on instance", instance.idx
                for s, sent in enumerate(instance.input_sents):
                    print str(s) + ': ' + ' '.join(sent.tokens)

#                print "Active ngrams:"
#                ngram_vars = self.lp.retrieve_active_vars('NGRAM')
#                self.print_active_ngrams(ngram_vars, instance,
#                        show_words=False, show_adj=False)
#                print
#                self.print_active_ngrams(ngram_vars, instance,
#                        show_words=True, show_adj=True)
#                print

#                for ngram_idxs in self.get_active_ngram_idxs():
#                    ngram_tuple = []
#                    for s, w in ngram_idxs:
#                        if s is None:
#                            ngram_tuple.append(str(w))
#                        else:
#                            ngram_tuple.append(
#                                    instance.input_sents[s].tokens[w])
#                    print ngram_tuple

                print "Active flow:"
                adj_vars = self.lp.retrieve_active_vars('ADJ')
                self.print_adj(adj_vars, instance, show_words=False,
                        show_ngrams=False)
                print
                # Expanded form
#                self.print_adj(adj_vars, instance, show_words=True,
#                        show_ngrams=True)
#                print
#
#                print "Active words:"
#                word_vars = self.lp.retrieve_active_vars('WORD')
#                self.print_active_word(word_vars, instance, show_adj=False,
#                        show_ngrams=False)

                word_idxs = self.get_active_word_idxs(ordered=False)
                # Hack to avoid sentences that can't be parsed.
                if len(word_idxs) > 100:
                    word_idxs = word_idxs[:100]

        return word_idxs

    def get_solution_feats(self):
        """Return the sum of the feature vectors from the solution variables.
        """
#        feats_from_solution = [var.metadata
#                for feat_cat in self.feats.categories
#                for var in self.lp.retrieve_active_vars(feat_cat.upper())]

        feats_from_solution = []
        for feat_cat in self.feats.categories:
            var_type = feat_cat.upper()
            if var_type not in self.lp.variables:
                # FRAME and FE variables may be missing in some instances
                continue

            for var in self.lp.retrieve_active_vars(var_type):
                if self.lp.get_value(var) != 1:
                    feat_vector = self.feats.scale_feat_values(
                            var.metadata, self.lp.get_value(var))
                else:
                    feat_vector = var.metadata

                feats_from_solution.append(feat_vector)
#                print "##################", var.type
#                print self.feats.print_with_values(var.metadata)

        return self.feats.sum_feat_values(feats_from_solution)

##############################################################################
# Variable retrieval and debugging

    def get_active_word_idxs(self, ordered=True, use_adjs=False):
        """Retrieve a list of word indices representing the sentence from a
        solved LP.
        """
        if not ordered:
            # Use the ordering from the input.
            return [word_var.retrieve_grounding('s', 'w')
                    for word_var in self.lp.retrieve_active_vars('WORD')]

        if use_adjs:
            # Retrieve the words from ADJ variables, which are always bigrams.
            type_to_link = {'START': 'outgoing_adjs',
                            'WORD': 'outgoing_adjs',
                            'ADJ': 'tgt_word',
                            'END': 'nonexistent_link'}  # will terminate here
            leading_words_link = None
        else:
            # Retrieve the words from NGRAM vars.
            type_to_link = {'START': 'ngrams0',
                            'WORD': 'ngrams' + str(self.ngram_order - 2), #1',
                            'NGRAM': 'word' + str(self.ngram_order - 1),
                            'END': 'nonexistent_link'}  # will terminate here

            # This misses the first n-2 words so we add them in through
            # supplementary links
            leading_words_link = None# {'NGRAM': 'word1'}
            if self.ngram_order > 2:
                leading_words_link = {'NGRAM': ['word' + str(i)
                                    for i in range(1, self.ngram_order - 1)]}

        start_var = self.lp.retrieve_variable('START')
        chain = self.lp.retrieve_active_chain(type_to_link, start_var,
                supplementary_links=leading_words_link)
        if chain is None:
            return None
        else:
            return [tuple(var.retrieve_grounding('s', 'w'))
                    for var in chain if var.type == 'WORD']

    def get_active_ngram_idxs(self):
        """Return the indices of all active ngrams in the current solution.
        """
        ngram_idxs = []
        for ngram_var in self.lp.retrieve_active_vars('NGRAM'):
            ngram_idxs.append([ngram_var.retrieve_grounding('s' + str(i),
                                                            'w' + str(i))
                for i in range(self.ngram_order)])
        return ngram_idxs

    def print_adj(self, adj_vars, instance, show_words=False,
            show_ngrams=False):
        """Print adjacency graph for debugging.
        """
        for adj_var in adj_vars:
            if not self.lp.is_active(adj_var):
                continue

            s0, w0, s1, w1 = adj_var.retrieve_grounding('s0','w0','s1','w1')
            print w0 if s0 is None else instance.input_sents[s0].tokens[w0],
            print '->',
            print w1 if s1 is None else instance.input_sents[s1].tokens[w1],
            print ' <' + str(self.lp.get_value(adj_var)) + '> ',
            print adj_var.readable_grounding()

            if show_words:
                src_word_vars = self.lp.retrieve_links(adj_var, 'src_word')
                print "SRC:",
                self.print_active_word(src_word_vars, instance,
                        show_adj=False, show_ngrams=show_ngrams)

                tgt_word_vars = self.lp.retrieve_links(adj_var, 'tgt_word')
                print "TGT:",
                self.print_active_word(tgt_word_vars, instance,
                        show_adj=False, show_ngrams=show_ngrams)
                print

    def print_active_word(self, word_vars, instance, show_adj=False,
            show_ngrams=False):
        for word_var in word_vars:
            if word_var.type == 'START':
                print "START", self.lp.get_value(word_var)
            elif word_var.type == 'END':
                print "END", self.lp.get_value(word_var)
            else:
                if not self.lp.is_active(word_var):
                    continue
                else:
                    s, w = word_var.retrieve_grounding('s', 'w')
                    print instance.input_sents[s].tokens[w],
                    print ' <' + str(self.lp.get_value(word_var)) + '> ',
                    print word_var.readable_grounding()

            if show_ngrams:
                for n in reversed(range(self.ngram_order)):
                    ngram_vars = self.lp.retrieve_links(word_var,
                            'ngrams' + str(n))
                    print "ngrams" + str(n) + ": ",
                    self.print_active_ngrams(ngram_vars, instance,
                            show_adj=False, show_words=False)

            if show_adj:
                in_adj_vars = self.lp.retrieve_links(word_var,
                        'incoming_adjs')
                print "ADJ-IN:",
                self.print_adj(in_adj_vars, show_ngrams=False,
                        show_words=False)

                out_adj_vars = self.lp.retrieve_links(word_var,
                        'outgoing_adjs')
                print "ADJ-OUT:",
                self.print_adj(out_adj_vars, show_ngrams=False,
                        show_words=False)
                print

    def print_active_ngrams(self, ngram_vars, instance, show_adj=False,
            show_words=False):
        found_active = False
        for ngram_var in ngram_vars:
            if not self.lp.is_active(ngram_var):
                continue
            else:
                if found_active:
                    # More than one ngram is active, separate by newlines
                    print
                found_active = True

                word_idxs = [ngram_var.retrieve_grounding('s' + str(i),
                                                          'w' + str(i))
                             for i in range(self.ngram_order)]

                ngram_tuple = []
                for s, w in word_idxs:
                    if s is None:
                        ngram_tuple.append(str(w))
                    else:
                        ngram_tuple.append(
                                instance.input_sents[s].tokens[w])
                print ngram_tuple,
                print ' <' + str(self.lp.get_value(ngram_var)) + '> ',
                print ngram_var.readable_grounding()

                if show_adj:
                    for i in range(self.ngram_order-1):
                        adj_vars = self.lp.retrieve_links(ngram_var,
                                'adj' + str(i))
                        print "ADJ" + str(i) + "-" + str(i+1) + ": ",
                        self.print_adj(adj_vars, instance,
                                show_ngrams=False, show_words=show_words)

    def get_tree_solution(self, instance, word_idxs):
        """Retrieve the active dependency tree in a solved LP.
        """
        # For easy position lookup
        widx_position = dict((tuple(word_idx), position)
                             for position, word_idx in enumerate(word_idxs))

        # If we aren't enforcing a tree, use a graph.
        if self.var_flags['cyclic']:
            active_tree = structure.DependencyGraph(len(word_idxs))
        else:
            active_tree = structure.DependencyTree(len(word_idxs))

        for dep_var in self.lp.retrieve_active_vars('DEP'):
            s0, w0, s1, w1, direction = dep_var.retrieve_grounding('s0', 'w0',
                                                                   's1', 'w1',
                                                                   'direction')

            # Ignore ROOTs and LEAFs
            if s0 is None or s1 is None:
                continue

            src_pos = widx_position[(s0, w0)]
            tgt_pos = widx_position[(s1, w1)]

            # Confirm that the given direction actually matches the output
            # token positions
            if direction is not None and \
                    direction != np.sign(tgt_pos - src_pos):
                print "ERROR: inconsistent direction", direction,
                print "for", src_pos, "->", tgt_pos, "in:"
                print ' '.join(instance.input_sents[s].tokens[w]
                               for s, w in word_idxs)

            active_tree.add_edge(src_pos, tgt_pos, score=dep_var.coeff,
                                 direction=np.sign(tgt_pos - src_pos))

        if not self.var_flags['cyclic'] and not active_tree.is_well_formed():
            print "ERROR: active tree not well-formed for output sentence:"
            print ' '.join(instance.input_sents[s].tokens[w]
                           for s, w in word_idxs)
            for edge in active_tree.edges:
                print edge.to_text()

        return active_tree

    def get_frame_solution(self, instance, word_idxs):
        """Retrieve the active frame semantic structure in a solved LP.
        """
        # For easy position lookup
        widx_position = dict((tuple(word_idx), position)
                             for position, word_idx in enumerate(word_idxs))

        active_frames = structure.DependencyDag(len(word_idxs))

        if 'FRAME' not in self.lp.variables:
            return active_frames

        for frame_var in self.lp.retrieve_active_vars('FRAME'):

            s, f = frame_var.retrieve_grounding('s', 'f')
            input_frame = instance.input_sents[s].frames.nodes[f]
            output_frame = active_frames.add_aux_node(name=input_frame.name)

            # Add an edge connecting the frame to its target word span.
            # All the target words of the original frame should be preserved.
            tgt_idxs = [widx_position[(s, w)] for w in input_frame.tgt_idxs]
            tgt_edge = active_frames.add_edge(output_frame,
                                              tgt_idxs[-1],
                                              lex_idxs=tgt_idxs,
                                              target=True)
            output_frame.add_attributes(tgt_idxs=tgt_idxs, tgt_edge=tgt_edge)

            # Add active frame elements.
            for fe_var in self.lp.retrieve_links(frame_var, 'frame_elements'):
                if not self.lp.is_active(fe_var):
                    continue
                s1, w = fe_var.retrieve_grounding('s1', 'w')
                output_w = widx_position[(s1, w)]

                # Frame elements are necessarily unlabeled so we do not
                # create an 'fe_edges' attribute for the frame node or an
                # 'fe' attribute for the edge. We must also restrict the
                # lexical span of an FE to just one word.
                active_frames.update_edge(output_frame,
                                          output_w,
                                          lex_idxs=[output_w],
                                          fe='UNLABELED')

        return active_frames

    def verify_arity(self, instance, word_idxs, active_tree=None):
        """Ensure that the arity variables are accurate.
        """
        if active_tree is None:
            active_tree = self.get_tree_solution(instance, word_idxs)

        for position, word_idx in enumerate(word_idxs):
            s, w = word_idx
            arity_var = self.lp.retrieve_variable('ARITY', s=s, w=w)
            arity_value = self.lp.get_value(arity_var)

            num_active_children = len(active_tree.get_child_idxs(position))
            if self.var_flags['arity_abs']:
                assert arity_value == num_active_children
            else:
                num_orig_children = len(
                        instance.input_sents[s].dparse.get_child_idxs(w))
                assert arity_value == num_active_children - num_orig_children

    def verify_range(self, instance, word_idxs, active_tree=None):
        """Ensure that the range variables are accurate.
        """
        if active_tree is None:
            active_tree = self.get_tree_solution(instance, word_idxs)

        for edge in active_tree.edges:
            s0, w0 = word_idxs[edge.src_idx]
            s1, w1 = word_idxs[edge.tgt_idx]
            range_var = self.lp.retrieve_variable('RANGE',
                                                  s0=s0, w0=w0,
                                                  s1=s1, w1=w1,
                                                  direction=edge.direction)
            range_value = self.lp.get_value(range_var)

            assert range_value == abs(edge.src_idx - edge.tgt_idx)
