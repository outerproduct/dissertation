#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import numpy as np


class TransductionConstraints(object):
    """A collection of classmethods that specify linear constraints and
    groups of constraints for LPs which represent transduction instances.
    """
    @classmethod
    def enforce_structure(cls, lp, instance, var_types, ngram_order=3,
            max_flow=100, **kwargs):
        """Apply structural constraints specific to each variable type.
        """
        if 'ngram' in var_types:
            cls.ngram_seq(lp,
                    ngram_order=ngram_order,
                    max_flow=max_flow,
                    **kwargs)
        if 'dep' in var_types:
            cls.dep_tree(lp,
                    max_flow=max_flow,
                    **kwargs)
        if 'range' in var_types:
            cls.dep_range(lp,
                    max_flow=max_flow,
                    **kwargs)
        if 'arity' in var_types:
            cls.arity(lp, instance,
                    max_flow=max_flow,
                    **kwargs)
        if 'fe' in var_types:
            cls.frames(lp, instance,
                    **kwargs)

    @classmethod
    def add(cls, constraint_group, lp, instance):
        """Add a named group of constraints (defined below).
        """
        for constraint in getattr(cls, constraint_group)():
            getattr(cls, constraint[0])(lp, instance, *constraint[1:])

##############################################################################
# Constraints

    @classmethod
    def ngram_seq(cls, lp, ngram_order=None, max_flow=None, **kwargs):
        """Constrain active ngrams to form an acyclic path. Applied by
        default whenever ngram features are involved.
        """
        # Sanity constraints since the bounds don't always seem to work
        lp.add_constraint('START', 'is_exactly', 1)
        lp.add_constraint('END', 'is_exactly', 1)

        # Start and end words are covered by at least one boolean 'meta'
        # ADJ variable.
        lp.add_constraint('START', 'has_exactly', 1, 'ngrams0')
        lp.add_constraint('END', 'has_exactly', 1, 'ngrams' +
                str(ngram_order-1))

        # WORDs are only active when they consume flow from ADJ variables.
        # ADJs are only active when they have active target words.
        lp.add_constraint('ADJ', 'has_flow_over', 'ngrams', max_flow)
        lp.add_constraint('WORD', 'requires_flow_between',
                'incoming_adjs', 'outgoing_adjs')

        # NOTE: the adjacency flow between (final_word) and END will be 0
        # as END is not a WORD and therefore does not require incoming flow.
        # However, this doesn't cause problems because END is constrained
        # above to activate exactly one incoming n-gram.

        # Each word must be covered by exactly one NGRAM in each position.
        for i in range(ngram_order):
            lp.add_constraint('WORD', 'iff_exactly', 1, 'ngrams' + str(i))
            lp.add_constraint('NGRAM', 'implies', 'word' + str(i))

        # Note that only one of the three below should be strictly
        # necessary NEW NOTE: no longer necessary because of above
        # TODO: cubic constraints? Is this a bad idea? ANS: not cubic!
        # lp.add_constraint('NGRAM', 'implies', 'words')

        # ADJ variables should carry flow (up to F) when n-1 corresponding
        # NGRAM variables are active.
        # TODO: this seems a bit dodgy.
#            lp.add_constraint('ADJ', 'has_flow_over', 'ngrams', max_flow)

        # Each NGRAM variable should imply each of its ADJ variables
        # FIXME temporary; remove
#        for i in range(ngram_order - 1):
#            lp.add_constraint('NGRAM', 'implies', 'adj' + str(i))

        # ADJ variables should carry flow only if the corresponding
        # NGRAM variables are active. We expect that n-1 NGRAM variables
        # will be active for a given ADJ (except the ones connecting
        # START and END).
        #lp.add_constraint('ADJ', 'has_flow_over', 'ngrams',
        #                       max_flow/(ngram_order-1))

    @classmethod
    def dep_tree(cls, lp, directional=False, max_flow=None, cyclic=False,
            standalone=False, **kwargs):
        """Constrain active dependency arcs to form a tree. Applied by
        default whenever dependency features are involved.
        """
        # There is exactly one root verb present
        lp.add_constraint('ROOT', 'has_exactly', 1, 'outgoing_deps')

        # Ensure that each word has only one syntactic head
        lp.add_constraint('WORD', 'iff_exactly', 1, 'incoming_deps')

        if cyclic or standalone:
            # Ensure that dependencies have active governor words;
            # unnecessary if a tree is being enforced.
            lp.add_constraint('WORD', 'implied_by', 'outgoing_deps')

        if not cyclic:
            # Words are only active when they consume flow from ARC variables
#            lp.add_constraint('WORD', 'requires_flow_between',
#                    'incoming_arcs', 'outgoing_arcs')

            # The version above isn't applied when words don't have any
            # outgoing arcs, say for heavily-constrained tree spaces.
            for word_var in lp.retrieve_all_variables('WORD'):
                if word_var.has_link('outgoing_arcs'):
                    word_var.add_constraint('requires_flow_between',
                            'incoming_arcs', 'outgoing_arcs')
                else:
                    word_var.add_constraint('iff_exactly', 1, 'incoming_arcs')

            # ARC variables should carry flow only if the corresponding DEP
            # variables are active
            lp.add_constraint('ARC', 'has_flow_over', 'dep', max_flow)

        if directional:
            dep_vars = lp.retrieve_all_variables('DEP')
            for dep_var in dep_vars:
                if dep_var.grounding['direction'] is not None:
                    # The flow difference between the following word
                    # (smaller flow) and preceding word (larger flow)
                    # must be negative for active dependencies.
                    dep_var.add_constraint('general',
                            ['following_adjs','preceding_adjs','own_idx'],
                            [1, -1, max_flow],
                            '<=', max_flow)

    @classmethod
    def dep_range(cls, lp, max_flow=None, **kwargs):
        """Constrain range variables to actually record the distance for
        an active dependency.
        """
        # The range variables must capture only positive distances,
        # while the bilge variables can capture anything.
        lp.add_constraint('RANGE', 'lower_bound', 0)
        lp.add_constraint('BILGE', 'lower_bound', -max_flow)

        # The range and bilge variables together equal the positive flow
        # difference between the preceding word (larger flow) and the
        # following word (smaller flow).
        lp.add_constraint('RANGE', 'general',
                ['own_idx', 'bilge', 'following_adjs', 'preceding_adjs'],
                [1, 1, 1, -1],
                '=', 0)

        # When the dependency variable is inactive, the range variable
        # should be zero.
        lp.add_constraint('RANGE', 'general',
                ['own_idx', 'dep'],
                [1, -max_flow],
                '<=', 0)

        # When the dependency variable is active, the bilge variable should
        # be negative.
        lp.add_constraint('BILGE', 'general',
                ['own_idx', 'dep'],
                [1, max_flow],
                '<=', 0)

    @classmethod
    def arity(cls, lp, instance, arity_abs=False, max_flow=None, **kwargs):
        """Record arity difference of words from their arity in the
        input. Note that this will not work by default with paraphrases in
        the input.
        """
        if arity_abs:
            # Arity variables capture the actual arity of each active token
            lp.add_constraint('WORD', 'general',
                    ['arity','outgoing_deps'],
                    [1, -1],
                    '=', 0)
        else:
            # Allow arity to be negative to avoid infeasability
            lp.add_constraint('ARITY', 'lower_bound', -max_flow)

            for word_var in lp.retrieve_all_variables('WORD'):
                s, w = word_var.retrieve_grounding('s', 'w')
                sent = instance.input_sents[s]
                num_children = sent.dparse.get_num_children(w)

                word_var.add_constraint('general',
                        ['arity', 'own_idx', 'outgoing_deps'],
                        [1, num_children, -1],
                        '=', 0)

    @classmethod
    def frames(cls, lp, instance, **kwargs):
        """Ensure that frames and frame elements are consistent with
        the output.
        """
        # Ensure there are FRAME and FE variables to constrain
        if 'FRAME' not in lp.variables:
            return

        # An active frame implies ALL its target words (too strict? not really)
        lp.add_constraint('FRAME', 'implies', 'tgt_words')

        # An active frame has at least one frame element (not true at all)
#        lp.add_constraint('FRAME', 'implies_at_least', 1, 'frame_elements')

        # An active frame element implies both its frame and the head
        # word of its lexicalized span
        # TODO: try all words of its span for orig_fes=True?
        lp.add_constraint('FE', 'implies', 'frame')
        lp.add_constraint('FE', 'implies', 'word')

    @classmethod
    def comp_rate(cls, lp, instance, lb_frac, ub_frac=None):
        """Specify a minimum (and optionally maximum) number of words which
        must be in the output. The bounds are calculated using a
        fraction of the length of the first sentence,
        """
        lb_value = np.floor(lb_frac * instance.sent_lens[0])
        if lb_value == 0:
            lb_value = 1
        lp.add_constraint('WORD', 'sum_lower_bound', lb_value)

        if ub_frac is not None:
            assert lb_frac <= ub_frac
            ub_value = np.floor(ub_frac * instance.sent_lens[0])
            lp.add_constraint('WORD', 'sum_upper_bound', ub_value)

    @classmethod
    def gold_rate(cls, lp, instance, func=None):
        """Specify the exact number of words that should be in the output
        by looking at the gold output.
        """
        if func is not None:
            merged_value = func([len(gold_sent.tokens)
                    for gold_sent in instance.gold_sentences])
            lb_value = np.floor(merged_value)
            ub_value = np.ceil(merged_value)
        else:
            lb_value, ub_value = 9999, 0
            for gold_sent in instance.gold_sentences:
                gold_len = len(gold_sent.tokens)
                if gold_len < lb_value:
                    lb_value = gold_len
                if gold_len > ub_value:
                    ub_value = gold_len

        lp.add_constraint('WORD', 'sum_lower_bound', lb_value)
        lp.add_constraint('WORD', 'sum_upper_bound', ub_value)

    @classmethod
    def modifiers(cls, lp, instance):
        """Ensure that certain modifier words can only be present if their
        their syntactic heads are present.
        """
        # Stanford dependency labels used in Clarke-style modifier constraints
        # - non-clause modifiers (advmod, amod, infmod, nn, npadvmod,
        #   num, number, quantmod, rcmod, tmod)
        # - determiners (det)
#        mod0_labels = ('advmod', 'amod', 'infmod', 'npadvmod', 'nn', 'num',
#            'number', 'quantmod', 'rcmod', 'det')
        # RASP relation labels used in Clarke-style modifier constraints
        mod0_labels = ('gr_ncmod', 'gr_det')
        for mod_label in mod0_labels:
            lp.add_constraint('WORD', 'implies', mod_label, warnings=False)

        # Stanford dependency labels used in Clarke-style modifier constraints
        # - negations (neg)
        # - possessives (possessive, poss)
        mod1_labels = ('neg', 'poss', 'possessive')
        for mod_label in mod1_labels:
            lp.add_constraint('WORD', 'iff', mod_label, warnings=False)

    @classmethod
    def parenthetical(cls, lp, instance):
        """Ensure that parenthetical expressions are dropped.
        """
        lp.add_constraint('META', 'has_exactly', 0, 'parenthetical',
                warnings=False)

    @classmethod
    def argstructure(cls, lp, instance):
        """Ensure that syntactic structure is maintained in the output.
        """
        # Predicates and arguments must appear together
        lp.add_constraint('WORD', 'iff', 'arguments', warnings=False)

        # There must be at least one verb if verbs are in the input
        meta_var = lp.retrieve_variable('META')
        if len(lp.retrieve_links(meta_var, 'verbs')) > 0:
            lp.add_constraint('META', 'has_at_least', 1, 'verbs',
                warnings=False)

        # Prepositions and subordinating conjunctions must have at least one
        # word from the subordinated clause and, conversely, any such word
        # must be accompanied by the corresponding preposition or
        # subordinating conjunction.
        lp.add_constraint('WORD', 'implies_at_least', 1, 'subordinated',
                warnings=False)
        lp.add_constraint('WORD', 'implies', 'subordinator', warnings=False)

        # Coordinating conjunctions imply all conjuncts.
        lp.add_constraint('WORD', 'implies', 'coordinated', warnings=False)

        # Two or more active conjuncts imply the coordinating conjunction
        word_vars = lp.retrieve_all_variables('WORD')
        for word_var in word_vars:
            conj_vars = lp.retrieve_links(word_var, 'coordinated')
            if len(conj_vars) > 2:
                # word_var is a CC
                word_var.add_constraint('general',
                    ['coordinated', 'own_idx'], [1, -len(conj_vars)], '<=', 1,
                    warnings=False)

    @classmethod
    def discourse(cls, lp, instance):
        """Ensure that personal pronouns are preserved in the result.
        """
        meta_var = lp.retrieve_variable('META')
        for prp_var in lp.retrieve_links(meta_var, 'personal_pronouns'):
            prp_var.add_constraint('general', ['own_idx'], [1], '=', 1)

    @classmethod
    def redundancy(cls, lp, instance):
        """Ensure that redundant content words don't appear in the output.
        """
        meta_var = lp.retrieve_variable('META')
        seen_word_idxs = {}
        for word, word_idx_set in instance.support_clusters.iteritems():
            if len(word_idx_set) == 0:
                continue

            # Per sentence count for each word
            per_sent_count = [0 for i in range(len(instance.input_sents))]

            for word_idx in word_idx_set:
                if word_idx in seen_word_idxs:
                    # We've seen this set before. The token must have
                    # multiple variations, e.g., a capitalized version
                    dupe_word = seen_word_idxs[word_idx]
                    dupe_word_idx_set = instance.support_clusters[dupe_word]

                    # If the variation has the same support set, all is good;
                    # the constraint is just duplicated with no consequences.
                    # If not, we print a warning.
                    if word_idx_set != dupe_word_idx_set:
                        print "WARNING: found odd overlap among clusters for"
                        print "\'" + word + "\' -> ", word_idx_set, "and"
                        print "\'" + dupe_word + "\' -> ", dupe_word_idx_set
                        print "in sentences:"
                        for s, sent in enumerate(instance.input_sents):
                            print s, sent.untokenize()
                        print "from instance", instance.idx
                        print

                seen_word_idxs[word_idx] = word
                s, w = word_idx
                per_sent_count[s] += 1
                word_var = lp.retrieve_variable('WORD', s=s, w=w)
                meta_var.add_link(word, word_var)

            # Constrain to the maximum number of occurrences of the word
            # in any input sentence
            max_occurrences = max(per_sent_count)
            if max_occurrences < 1:
                print "ERROR: missing word", word, word_idx_set, "in all of"
                for sent in instance.input_sents:
                    print sent.untokenize()

            lp.add_constraint('META', 'has_at_most', max_occurrences, word)

            # Alternatively, constrain to at most one occurrence
            #lp.add_constraint('META', 'has_exactly', 1, word)
            #lp.add_constraint('META', 'has_at_most', 1, word)

##############################################################################
# Constraint groups

    @classmethod
    def goldcr(cls):
        """Compression rate based on gold data.
        """
        return [('gold_rate', None)]

    @classmethod
    def goldmincr(cls):
        """Compression rate from the minimum gold rate.
        """
        return [('gold_rate', min)]

    @classmethod
    def goldmedcr(cls):
        """Compression rate from the median gold rate.
        """
        return [('gold_rate', np.median)]

    @classmethod
    def goldmaxcr(cls):
        """Compression rate from the maximum gold rate.
        """
        return [('gold_rate', np.max)]

    @classmethod
    def goldavgcr(cls):
        """Compression rate from the average gold rate.
        """
        return [('gold_rate', np.average)]

    @classmethod
    def cr10(cls):
        """10 percent of input.
        """
        return [('comp_rate', 0.1, 0.1)]

    @classmethod
    def cr20(cls):
        """20 percent of input.
        """
        return [('comp_rate', 0.2, 0.2)]

    @classmethod
    def cr30(cls):
        """30 percent of input.
        """
        return [('comp_rate', 0.3, 0.3)]

    @classmethod
    def cr40(cls):
        """40 percent of input. Configuration of basic compression rate
        constraint in Clarke & Lapata (2008).
        """
        return [('comp_rate', 0.4, 0.4)]

    @classmethod
    def cr50(cls):
        """50 percent of input.
        """
        return [('comp_rate', 0.5, 0.5)]

    @classmethod
    def cr60(cls):
        """60 percent of input.
        """
        return [('comp_rate', 0.6, 0.6)]

    @classmethod
    def cr70(cls):
        """70 percent of input.
        """
        return [('comp_rate', 0.7, 0.7)]

    @classmethod
    def cr80(cls):
        """80 percent of input.
        """
        return [('comp_rate', 0.8, 0.8)]

    @classmethod
    def cr90(cls):
        """90 percent of input.
        """
        return [('comp_rate', 0.9, 0.9)]

    @classmethod
    def cr100(cls):
        """Sanity check: no compression permitted.
        """
        return [('comp_rate', 1)]

    @classmethod
    def clarke(cls):
        """Configuration of constraints to replicate Clarke & Lapata (2008).
        """
        return [('modifiers',),
                ('parenthetical',),
                ('argstructure',),
                ('discourse',),
                ]

    @classmethod
    def fusion(cls):
        """Constraints specific to sentence fusion.
        """
        return [('redundancy',)]
