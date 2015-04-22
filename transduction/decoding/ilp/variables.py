#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import sys
from transduction.model import metaidx


class TransductionVariables(object):
    """A collection of classmethods that add groups of variables to LPs
    which represent transduction instances.
    """
    @classmethod
    def add_all(cls, lp, feats, instance, ngram_order=3, **kwargs):
        """Add variables for words and structural indicators.
        """
        # Add variables for each word in the input
        cls.add_words(lp, feats, instance, **kwargs)

        # Add structural variables
        if 'ngram' in feats.categories:
            cls.add_fwd_ngrams(lp, feats, instance, ngram_order, **kwargs)
        if 'dep' in feats.categories:
            cls.add_dependencies(lp, feats, instance, **kwargs)
        if 'frame' in feats.categories or 'fe' in feats.categories:
            cls.add_frames_and_fes(lp, feats, instance, **kwargs)

    @classmethod
    def parse_var_conf(cls, var_conf):
        """Convert the variable configuration into a dictionary of flags.
        """
        var_flags = dict((flag_name, False) for flag_name in
                ('anc_deps',    # dependencies only from ancestors
                 'arity_abs',   # absolute arity values instead of deltas
                 'cubic',       # all trigrams instead of trigrams over pairs
                 'cyclic',      # don't enforce a tree in the output
                 'directional', # dependencies are direction-aware
                 'fixed_start', # output starts with an input start token
                 'fixed_end',   # output ends with an input end token
                 'fixed_root',  # output headed by an input root
                 'noninv_deps', # input dependencies can't be inverted
                 'orig_deps',   # dependencies only from original tree
                 'orig_fes',    # frame elements only from original annotation
                 'pos_deps',    # only dependencies that match POS tags
                 'projective',  # use model intended for projective trees
                 #'range_delta', # delta range values instead of absolute #TODO
                 #'reorder',     # permit full reordering of input words  #TODO
                 'supported',   # n-grams only pivot through matching words
                 'verb_root',   # output headed by an input head predicate
                 ))
        for flag_name in var_conf:
            try:
                var_flags[flag_name] = True
            except KeyError:
                sys.stderr.write("ERROR: unknown variable flag \'" +
                        flag_name + "\'\n")
                sys.stderr.write("Valid flags: " +
                        ', '.join(var_flags.iterkeys()))
                sys.exit()
        return var_flags

##############################################################################
# Variables

    @classmethod
    def add_word(cls, lp, feats, instance, word_idx):
        """Add a word variable.
        """
        s, w, = word_idx
        word_var = lp.create_boolean_variable('WORD', s=s, w=w)
        word_var.set_metadata(feats.get_feat_vector(instance,
                                                    (word_idx,),
                                                    'word'))

        if 'arity' in feats.categories:
            arity_var = lp.create_variable('ARITY', s=s, w=w)
            arity_var.set_metadata(feats.get_feat_vector(instance,
                                                         (word_idx,),
                                                         'arity'))

            # Link the word variable to its corresponding arity variable
            arity_var.add_link('word', word_var)
            word_var.add_link('arity', arity_var)

        return word_var

    @classmethod
    def add_ngram(cls, lp, feats, instance, word_idxs):
        """Add an n-gram variable and determine its coefficient in the
        objective function using the given features. Also add the
        underlying structural variables.
        """
        n = len(word_idxs)

        # Format keyword arguments to ground the variable, then create it
        kwargs = {}
        for i in range(n):
            s, w = word_idxs[i]
            kwargs['s' + str(i)] = s
            kwargs['w' + str(i)] = w
        ngram_var = lp.create_boolean_variable('NGRAM', **kwargs)
        ngram_var.set_metadata(feats.get_feat_vector(instance,
                                                     word_idxs,
                                                     'ngram',
                                                     ngram_order=n))

        # Link the ngram variable to its word variables
        for i, word_idx in enumerate(word_idxs):
            s, w = word_idx
            if w == metaidx.START:
                start_var = lp.retrieve_variable('START')
                start_var.add_link('ngrams0', ngram_var)
                ngram_var.add_link('word0', start_var)
            elif w == metaidx.END:
                end_var = lp.retrieve_variable('END')
                end_var.add_link('ngrams' + str(i), ngram_var)
                ngram_var.add_link('word' + str(i), end_var)
            else:
                word_var = lp.retrieve_variable('WORD', s=s, w=w)
                word_var.add_link('ngrams' + str(i), ngram_var)
                ngram_var.add_link('word' + str(i), word_var)

                # If this is a sentence-starting ngram, the non-START words
                # that it contains will not be covered by ngrams in certain
                # positions. For example, the first non-START word in a
                # sentence-starting trigram can only be preceded by START and
                # cannot terminate (be the final word of) any active trigram
                # since there are no words before START. We therefore assign
                # this ngram to those positions, so all those roles will be
                # filled if it's active.
                if word_idxs[0][1] == metaidx.START:
                    for j in range(i+1, n):
                        word_var.add_link('ngrams' + str(j), ngram_var)

                # Similarly, sentence-ending ngrams have non-END words that
                # can't be adequately covered by ngrams in a solution. For
                # example, the last non-END word in an ending trigram can't
                # start any trigram. As before, this ending ngram must cover
                # those positions for the word.
                if word_idxs[-1][1] == metaidx.END:
                    for j in range(0,i):
                        word_var.add_link('ngrams' + str(j), ngram_var)

        # Add adjacency variables for structural constraints
        for i in range(n-1):
            s0, w0 = word_idxs[i]
            s1, w1 = word_idxs[i+1]
            adj_var = cls.add_adj(lp, s0, w0, s1, w1)
            ngram_var.add_link('adj' + str(i), adj_var)
            adj_var.add_link('ngrams', ngram_var)

        return ngram_var

    @classmethod
    def add_adj(cls, lp, s0, w0, s1, w1):
        """Add a directed adjacency variable representing the integer-valued
        flow between two words from two (pseudo) sentences that should appear
        adjacently in the solution. Note that word 0 is assumed to appear
        before word 1.
        """
        try:
            adj_var = lp.retrieve_variable('ADJ', s0=s0, s1=s1, w0=w0, w1=w1)
        except KeyError:
            adj_var = lp.create_variable('ADJ', s0=s0, s1=s1, w0=w0, w1=w1)

            # Add links to and from the word variables
            if w0 == metaidx.START:
                w0_var = lp.retrieve_variable('START')
            else:
                w0_var = lp.retrieve_variable('WORD', s=s0, w=w0)

            if w1 == metaidx.END:
                w1_var = lp.retrieve_variable('END')
            else:
                w1_var = lp.retrieve_variable('WORD', s=s1, w=w1)
            adj_var.add_link('src_word', w0_var)
            adj_var.add_link('tgt_word', w1_var)
            w0_var.add_link('outgoing_adjs', adj_var)
            w1_var.add_link('incoming_adjs', adj_var)

        return adj_var

    @classmethod
    def add_dependency(cls, lp, feats, instance, word_idxs, direction=None):
        """Add a dependency variable and determine its coefficient in the
        objective function using the given features. Also add the
        underlying structural variables.
        """
        # Create the dependency variable
        assert len(word_idxs) == 2

        s0, w0 = word_idxs[0]
        s1, w1 = word_idxs[1]
        dep_var = lp.create_boolean_variable('DEP', s0=s0, w0=w0,
                s1=s1, w1=w1, direction=direction)
        dep_var.set_metadata(feats.get_feat_vector(instance,
                                                   word_idxs,
                                                   'dep',
                                                   direction=direction))

        if s0 is None and w0 == metaidx.ROOT:
            word0_var = lp.retrieve_variable('ROOT')
        else:
            word0_var = lp.retrieve_variable('WORD', s=s0, w=w0)

        if s1 is None and w1 == metaidx.LEAF:
            word1_var = lp.retrieve_variable('LEAF')
        else:
            word1_var = lp.retrieve_variable('WORD', s=s1, w=w1)

        # Link the dependency variable to its parent word variable
        dep_var.add_link('word0', word0_var)
        word0_var.add_link('outgoing_deps', dep_var)

        # Link the dependency variable to its child word variable
        dep_var.add_link('word1', word1_var)
        word1_var.add_link('incoming_deps', dep_var)

        # For a direction-specific dependency, we note two sets of
        # outgoing ADJ flow variables: one for the preceding word (larger
        # ADJ flow) and one for the following one (smaller ADJ flow)
        if direction is not None:
            # Head follows dependent: left attachment (word1 < word0)
            if direction == -1:
                prec_word_var = word1_var if w1 != metaidx.LEAF \
                                          else lp.retrieve_variable('START')
                foll_word_var = word0_var
            elif direction == 1:
                # Head precedes dependent: right attachment (word0 < word1)
                prec_word_var = word0_var if w0 != metaidx.ROOT \
                                          else lp.retrieve_variable('START')
                foll_word_var = word1_var

            preceding_adj_vars = lp.retrieve_links(
                    prec_word_var, 'outgoing_adjs')
            dep_var.add_links('preceding_adjs', preceding_adj_vars)

            following_adj_vars = lp.retrieve_links(
                    foll_word_var, 'outgoing_adjs')
            dep_var.add_links('following_adjs', following_adj_vars)

            if 'range' in feats.categories and \
                    w0 != metaidx.ROOT and \
                    w1 != metaidx.LEAF:
                cls.add_range(lp,
                              feats,
                              instance,
                              word_idxs,
                              direction,
                              dep_var,
                              preceding_adj_vars,
                              following_adj_vars)

        # Add arc variables for structural constraints
        arc_var = cls.add_arc(lp, s0, w0, s1, w1)
        dep_var.add_link('arc', arc_var)
        arc_var.add_link('dep', dep_var)

        return dep_var

    @classmethod
    def add_arc(cls, lp, s0, w0, s1, w1):
        """Add a directed arc variable which represents the integer-valued
        flow between two words from two (pseudo) sentences that may have a
        dependency relationship in the solution. Note that word 0 is assumed
        to be a parent of word 1.
        """
        try:
            arc_var = lp.retrieve_variable('ARC', s0=s0, s1=s1, w0=w0, w1=w1)
        except KeyError:
            arc_var = lp.create_variable('ARC', s0=s0, s1=s1, w0=w0, w1=w1)

            # Add links to and from the word variables
            if w0 == metaidx.ROOT:
                w0_var = lp.retrieve_variable('ROOT')
            else:
                w0_var = lp.retrieve_variable('WORD', s=s0, w=w0)

            if w1 == metaidx.LEAF:
                w1_var = lp.retrieve_variable('LEAF')
            else:
                w1_var = lp.retrieve_variable('WORD', s=s1, w=w1)

            arc_var.add_link('src_word', w0_var)
            arc_var.add_link('tgt_word', w1_var)
            w0_var.add_link('outgoing_arcs', arc_var)
            w1_var.add_link('incoming_arcs', arc_var)

        return arc_var

    @classmethod
    def add_range(cls, lp, feats, instance, word_idxs, direction, dep_var,
            preceding_adj_vars, following_adj_vars):
        """Add variables which represent the range of an active dependency
        in the solution, i.e., the distance between its tokens. This is done
        by using two additional variables for each potential dependency:
        'range' for distances related to active dependencies and 'bilge' for
        the rest.
        """
        try:
            range_var = lp.retrieve_variable('RANGE', **dep_var.grounding)
            bilge_var = lp.retrieve_variable('BILGE', **dep_var.grounding)
        except KeyError:
            range_var = lp.create_variable('RANGE', **dep_var.grounding)
            bilge_var = lp.create_variable('BILGE', **dep_var.grounding)

            range_var.set_metadata(feats.get_feat_vector(
                instance, word_idxs, 'range', direction=direction))

            range_var.add_links('preceding_adjs',
                    lp.retrieve_links(dep_var, 'preceding_adjs'))
            range_var.add_links('following_adjs',
                    lp.retrieve_links(dep_var, 'following_adjs'))

            range_var.add_link('dep', dep_var)
            range_var.add_link('bilge', bilge_var)

            bilge_var.add_link('dep', dep_var)
            bilge_var.add_link('range', range_var)

            dep_var.add_link('range', range_var)
            dep_var.add_link('bilge', bilge_var)

        return range_var, bilge_var

    @classmethod
    def add_frame(cls, lp, feats, instance, frame_idx):
        """Add a frame variable.
        """
        s, f, = frame_idx
        frame_var = lp.create_boolean_variable('FRAME', s=s, f=f)
        frame_var.set_metadata(feats.get_feat_vector(instance,
                                                    (frame_idx,),
                                                    'frame'))

        # Record all target words of the frame
        for w in instance.input_sents[s].frames.nodes[f].tgt_idxs:
            tgt_word_var = lp.retrieve_variable('WORD', s=s, w=w)
            frame_var.add_link('tgt_words', tgt_word_var)

        return frame_var

    @classmethod
    def add_fe(cls, lp, feats, instance, fe_idxs, direction=None):
        """Add a frame element variable  connecting frames to words.
        """
        assert len(fe_idxs) == 2

        s0, f = fe_idxs[0]
        s1, w = fe_idxs[1]

        fe_var = lp.create_boolean_variable('FE', s0=s0, f=f, s1=s1, w=w)
        fe_var.set_metadata(feats.get_feat_vector(instance,
                                                  fe_idxs,
                                                  'fe'))

        frame_var = lp.retrieve_variable('FRAME', s=s0, f=f)
        frame_var.add_link('frame_elements', fe_var)
        fe_var.add_link('frame', frame_var)

        word_var = lp.retrieve_variable('WORD', s=s1, w=w)
        word_var.add_link('frame_elements', fe_var)
        fe_var.add_link('word', word_var)

        return fe_var

##############################################################################
# Variable groups

    @classmethod
    def add_words(cls, lp, feats, instance, **kwargs):
        """Add indicator variables which note whether a word is present
        in the solution or not.
        """
        # Create an unused variable to track certain categories of words for
        # manual constraints
        meta_var = lp.create_boolean_variable('META')
        paren_nesting = 0
        open_parens = set(('(', '[', '{', '<'))
        close_parens = set((')', ']', '}', '>'))

        for s, sentence in enumerate(instance.input_sents):
            word_vars = []
            parenthetical_idxs = set()
            for w, pos in enumerate(sentence.pos_tags):
                word_idx = (s, w)
                word_var = cls.add_word(lp, feats, instance, word_idx)
                word_vars.append(word_var)

                # Note aspects from the POS tags of the sentence that are
                # useful in constraints from Clarke & Lapata (2008)
                token = sentence.tokens[w]
                if token in open_parens:
                    paren_nesting += 1
                if paren_nesting > 0:
                    meta_var.add_link('parenthetical', word_var)
                    parenthetical_idxs.add(w)
                if token in close_parens:
                    paren_nesting -= 1

                if pos.startswith('V'):
                    meta_var.add_link('verbs', word_var)
                elif pos == 'PRP':
                    meta_var.add_link('personal_pronouns', word_var)

            # Augment the word indicators with syntactic information to help
            # apply constraints from Clarke & Lapata (2008).
            parse = sentence.dparse
            pos_tags = sentence.pos_tags
            for edge in parse.edges:
                src_idx, tgt_idx = edge.src_idx, edge.tgt_idx
                if src_idx in parenthetical_idxs or \
                        tgt_idx in parenthetical_idxs:
                    continue

                src_pos, tgt_pos = pos_tags[src_idx], pos_tags[tgt_idx]
                src_var, tgt_var = word_vars[src_idx], word_vars[tgt_idx]

                # Record the dependency for every node
                tgt_var.add_link(edge.label, src_var)

                # Record arguments of verbs
                if src_pos.startswith('V'):
                    if edge.label in set(('nsubj', 'nsubjpass', 'dobj',
                            'acomp')):
                        src_var.add_link('arguments', tgt_var)

                # Record subordination
                if tgt_pos.startswith('W') or tgt_pos == 'IN' or \
                        sentence.tokens[tgt_idx] == 'that':
                    tgt_node = parse.nodes[tgt_idx]
                    for sub_node in parse.get_descendants(tgt_node):
                        sub_var = word_vars[sub_node.idx]
                        tgt_var.add_link('subordinated', sub_var)
                        sub_var.add_link('subordinator', tgt_var)

                # Record coordination
                if tgt_pos == 'CC':
                    tgt_node = parse.nodes[tgt_idx]
                    for child_idx, child_edge in \
                            tgt_node.outgoing_edges.iteritems():
                        if child_edge.label != 'conj':
                            continue
                        child_var = word_vars[child_idx]
                        tgt_var.add_link('coordinated', child_var)

            # More syntactic information from RASP.
            if hasattr(sentence, 'relgraph'):
                for edge in sentence.relgraph.edges:
                    src_idx, tgt_idx = edge.src_idx, edge.tgt_idx

                    src_pos, tgt_pos = pos_tags[src_idx], pos_tags[tgt_idx]
                    src_var, tgt_var = word_vars[src_idx], word_vars[tgt_idx]

                    # Record the grammatical relation for every node
                    tgt_var.add_link('gr_' + edge.label, src_var)

    @classmethod
    def add_fwd_ngrams(cls, lp, feats, instance, ngram_order, cubic=False,
            supported=False, **kwargs):
        """Add forward n-grams. This provides a common interface for bigrams
        and trigrams.
        """
        # Create start and end variables
        lp.create_boolean_variable('START')
        lp.create_boolean_variable('END')

        if ngram_order == 2:
            if hasattr(instance, 'support') and supported:
                cls.add_supported_fwd_bigrams(lp, feats, instance,
                                              ngram_order=ngram_order,
                                              **kwargs)
            else:
                cls.add_fwd_bigrams(lp, feats, instance,
                                    ngram_order=ngram_order,
                                    **kwargs)
        elif ngram_order == 3:
            if supported:
                print "\'supported\' flag not available for trigrams"

            if cubic and len(instance.input_sents) == 1:
                cls.add_fwd_cubic_trigrams(lp, feats, instance,
                                           ngram_order=ngram_order,
                                           **kwargs)
            else:
                cls.add_fwd_trigrams(lp, feats, instance,
                                     ngram_order=ngram_order,
                                     **kwargs)
        else:
            print "LM order", ngram_order, "not yet supported"
            raise Exception

    @classmethod
    def add_fwd_bigrams(cls, lp, feats, instance, fixed_start=False,
            fixed_end=False, **kwargs):
        """Add bigram variables for (a) all ordered word pairs within a
        sentence (b) all word pairs across two sentences.
        """
        for s0, sentence0 in enumerate(instance.input_sents):
            for w0 in range(len(sentence0.tokens)):
                # Only allow one of the existing start words to start the
                # solution unless otherwise specified
                if w0 == 0 or not fixed_start:
                    word_idxs = ((None, metaidx.START), (s0, w0))
                    cls.add_ngram(lp, feats, instance, word_idxs)

                # Only allow one of the existing end words to end the
                # solution unless otherwise specified
                if w0 == len(sentence0.tokens) - 1 or not fixed_end:
                    word_idxs = ((s0, w0), (None, metaidx.END))
                    cls.add_ngram(lp, feats, instance, word_idxs)

                for s1, sentence1 in enumerate(instance.input_sents):
                    # For bigrams within a sentence, only allow forward
                    # connections
                    start_w = 0
                    if s0 == s1:
                        start_w = w0 + 1
                    for w1 in range(start_w, len(sentence1.tokens)):
                        word_idxs = ((s0, w0), (s1, w1))
                        cls.add_ngram(lp, feats, instance, word_idxs)

    @classmethod
    def add_supported_fwd_bigrams(cls, lp, feats, instance, fixed_start=False,
            fixed_end=False, **kwargs):
        """Add bigram variables for (a) all ordered word pairs within a
        sentence (b) all supported word pairs across two sentences.
        """
        word_idxs_seen = set()
        for s0, sentence0 in enumerate(instance.input_sents):
            for w0 in range(len(sentence0.tokens)):
                # Only allow one of the existing start words to start the
                # solution unless otherwise specified
                if w0 == 0 or not fixed_start:
                    word_idxs = ((None, metaidx.START), (s0, w0))
                    cls.add_ngram(lp, feats, instance, word_idxs)

                # Only allow one of the existing end words to end the
                # solution unless otherwise specified
                if w0 == len(sentence0.tokens) - 1 or not fixed_end:
                    word_idxs = ((s0, w0), (None, metaidx.END))
                    cls.add_ngram(lp, feats, instance, word_idxs)

                # Supported source words can go to every word that follows a
                # supporting word in the corresponding sentences, otherwise
                # they are limited to following words in the same sentence
                token0 = sentence0.tokens[w0]
                if token0 in instance.support:
                    tgt_starts = [(s, w+1)
                                 for s, w in instance.support[token0]]
                else:
                    tgt_starts = [(s0, w0+1)]

                for s, w_start in tgt_starts:
                    for w in range(w_start,
                                   len(instance.input_sents[s].tokens)):
                        # Supported target words can be replaced by their
                        # supporting words.
                        token1 = instance.input_sents[s].tokens[w]
                        if token1 in instance.support:
                            tgts = instance.support[token1]
                        else:
                            tgts = [(s, w)]

                        for s1, w1 in tgts:
                            word_idxs = ((s0, w0), (s1, w1))
                            # Repetition is inevitable here
                            if word_idxs not in word_idxs_seen:
                                cls.add_ngram(lp, feats, instance, word_idxs)
                                word_idxs_seen.add(word_idxs)

    @classmethod
    def add_fwd_trigrams(cls, lp, feats, instance, fixed_start=False,
            fixed_end=False, **kwargs):
        """Add trigram variables covering (a) all ordered word pairs within a
        sentence, (b) all word pairs across two sentences.
        """
        for s0, sentence0 in enumerate(instance.input_sents):
            for w0 in range(len(sentence0.tokens)):

                # Permit single-token solutions.
                if (w0 == 0 or not fixed_start) and \
                        (w0 == len(sentence0.tokens) - 1 or not fixed_end):
                    word_idxs = ((None, metaidx.START),
                                 (s0, w0),
                                 (None, metaidx.END))
                    cls.add_ngram(lp, feats, instance, word_idxs)

                for s1, sentence1 in enumerate(instance.input_sents):
                    # For trigrams within a sentence, only allow forward
                    # connections
                    start_w = 0
                    if s0 == s1:
                        start_w = w0 + 1
                    else:
                        # Don't allow a sentence-terminating period to be
                        # followed by anything except </s>
                        if w0 == len(sentence0.tokens) - 1 and \
                                sentence0.tokens[w0] == '.':
                            continue
                    for w1 in range(start_w, len(sentence1.tokens)):
                        # Create a trigram variable using the word preceding
                        # w0. In order to prevent duplicate ngrams, we skip
                        # this option for certain trigrams within the same
                        # sentence.
                        if w0 > 0 and (w1 > w0 + 1 or s0 != s1):
                            word_idxs = ((s0, w0-1), (s0, w0), (s1, w1))
                            cls.add_ngram(lp, feats, instance, word_idxs)

                        # Create a trigram variable using the word following
                        # w1
                        if w1 < len(sentence1.tokens) - 1:
                            word_idxs = ((s0, w0), (s1, w1), (s1, w1+1))
                            cls.add_ngram(lp, feats, instance, word_idxs)

                        # Only allow one of the existing start words to start
                        # the solution unless otherwise specified
                        if w0 == 0 or not fixed_start:
                            word_idxs = ((None, metaidx.START),
                                         (s0, w0),
                                         (s1, w1))
                            cls.add_ngram(lp, feats, instance, word_idxs)

                        # Only allow one of the existing end words to end
                        # the solution unless otherwise specified
                        if w1 == len(sentence1.tokens) - 1 or not fixed_end:
                            word_idxs = ((s0, w0),
                                         (s1, w1),
                                         (None, metaidx.END))
                            cls.add_ngram(lp, feats, instance, word_idxs)

    @classmethod
    def add_fwd_cubic_trigrams(cls, lp, feats, instance, fixed_start=False,
            fixed_end=False, **kwargs):
        """Add trigram variables covering all possible ordered trigrams
        in a single sentence.
        """
        # This is currently only defined over single sentence inputs
        # TODO extend to splits across sentences
        # TODO: this assert doesn't seem to work for fusion instances ?
        assert len(instance.input_sents) == 1
        sent = instance.input_sents[0]

        for w0 in range(len(sent.tokens) - 1):

            # Permit single-token solutions.
            if (w0 == 0 or not fixed_start) and \
                    (w0 == len(sent.tokens) - 1 or not fixed_end):
                word_idxs = ((None, metaidx.START),
                             (0, w0),
                             (None, metaidx.END))
                cls.add_ngram(lp, feats, instance, word_idxs)

            for w1 in range(w0 + 1, len(sent.tokens)):
                # Only allow one of the existing start words to start
                # the solution unless otherwise specified
                if w0 == 0 or not fixed_start:
                    word_idxs = ((None, metaidx.START),
                                 (0, w0),
                                 (0, w1))
                    cls.add_ngram(lp, feats, instance, word_idxs)

                # Only allow one of the existing end words to end
                # the solution unless otherwise specified
                if w1 == len(sent.tokens) - 1 or not fixed_end:
                    word_idxs = ((0, w0),
                                 (0, w1),
                                 (None, metaidx.END))
                    cls.add_ngram(lp, feats, instance, word_idxs)

                for w2 in  range(w1 + 1, len(sent.tokens)):
                    word_idxs = ((0, w0), (0, w1), (0, w2))
                    cls.add_ngram(lp, feats, instance, word_idxs)

    @classmethod
    def add_dependencies(cls, lp, feats, instance,
            directional=False, fixed_root=False, verb_root=False,
            noninv_deps=False, orig_deps=False, anc_deps=False, pos_deps=False,
            **kwargs):
        """Add dependency variables for all plausible syntactic links.
        """
        lp.create_boolean_variable('ROOT')
#        lp.create_boolean_variable('LEAF')

        for s0, sentence0 in enumerate(instance.input_sents):
            for w0 in range(len(sentence0.tokens)):
                # Only allow words to be the root of the tree if they
                # were the root of an input tree, unless otherwise specified.
                # Only verbs are allowed to be roots.
                if sentence0.dparse.is_root(w0) or ((not fixed_root) and \
                        ((not verb_root) or
                            sentence0.pos_tags[w0].lower().startswith('vb'))):
                    word_idxs = ((None, metaidx.ROOT), (s0, w0))
                    # By convention, ROOT is a right attachment (direction = 1)
                    cls.add_dependency(lp, feats, instance, word_idxs,
                            direction=1 if directional else None)

#                word_idxs = ((s0, w0), (None, metaidx.LEAF))
#                # By convention, LEAF is a left attachment (direction = -1)
#                cls.add_dependency(lp, feats, instance, word_idxs)
#                        direction=-1 if directional else None)

                if anc_deps:
                    # Avoid all dependencies other than those expressed in
                    # the input sentences and their ancestral relationships.
                    plausible_idxs = set((s1, w1)
                        for d0 in sentence0.dparse.get_descendant_idxs(w0)
                        for s1, w1_list in
                            instance.input_maps[s0][d0].iteritems()
                        for w1 in w1_list)
                elif orig_deps:
                    # Avoid all dependencies other than those expressed in
                    # the input sentences.
                    plausible_idxs = set((s1, w1)
                        for c0 in sentence0.dparse.get_child_idxs(w0)
                        for s1, w1_list in
                            instance.input_maps[s0][c0].iteritems()
                        for w1 in w1_list)
                elif noninv_deps:
                    # For dependencies within a sentence, don't allow
                    # connections that flip existing dependencies or
                    # invert an ancestry relationship.
                    ancestor_idxs = instance.input_sents[s0].\
                                            dparse.get_ancestor_idxs(w0)

                # Can stack POS-matching constraint with original tree or
                # ancestral DAG constraint
                if pos_deps:
                    # Don't allow allow dependent words whose POS does not
                    # match the POS of one the the governor's children in the
                    # input.
                    plausible_tags = set(sentence0.pos_tags[c][:2]
                            for c in sentence0.dparse.get_child_idxs(w0))

                for s1, sentence1 in enumerate(instance.input_sents):
                    for w1 in range(len(sentence1.tokens)):
                        if s0 == s1 and w0 == w1:
                            continue
                        if (orig_deps or anc_deps) and (s1, w1) not in \
                                plausible_idxs:
                            continue
                        if noninv_deps and s0 == s1 and w1 in ancestor_idxs:
                            continue
                        if pos_deps and sentence1.pos_tags[w1][:2] not in \
                                plausible_tags:
                            continue

                        word_idxs = ((s0, w0), (s1, w1))
                        if directional:
                            cls.add_dependency(lp, feats, instance, word_idxs,
                                            direction=1)
                            cls.add_dependency(lp, feats, instance, word_idxs,
                                            direction=-1)
                        else:
                            cls.add_dependency(lp, feats, instance, word_idxs,
                                            direction=None)
#                        direction = None
#                        if directional and s0 == s1:
#                            # We define direction as -1 for left attachment
#                            # (head follows dependent) and +1 for right
#                            # attachment (head precedes attachment). In
#                            # other words,
#                            # direction = sign(dep_idx - head_idx)
#                            # TODO: what about directional arcs across
#                            # sentences? Shouldn't this create *two*
#                            # variables with opposite directions?
#                            # XXX need to fix!
#                            direction = np.sign(w1 - w0)
#                        cls.add_dependency(lp, feats, instance, word_idxs,
#                                           direction=direction)

#    @classmethod
#    def add_all_dependencies(cls, lp, feats, instance, directional=False,
#            fixed_root=False, verb_root=False, noninvdeps=False, **kwargs):
#        """Add dependency variables for all possible syntactic links,
#        although restricting the root to be either a verb or the current root.
#        """
#        lp.create_boolean_variable('ROOT')
##        lp.create_boolean_variable('LEAF')
#
#        for s0, sentence0 in enumerate(instance.input_sents):
#            for w0 in range(len(sentence0.tokens)):
#                # Only allow words to be the root of the tree if they
#                # were the root of an input tree, unless otherwise specified.
#                # Optionally, roots are restricted to verbs.
#                if sentence0.dparse.is_root(w0) or (not fixed_root and \
#                        (not verb_root or
#                            sentence0.pos_tags[w0].lower().startswith('vb'))):
#                    word_idxs = ((None, metaidx.ROOT), (s0, w0))
#                    # By convention, ROOT is a right attachment (direction = 1)
#                    cls.add_dependency(lp, feats, instance, word_idxs,
#                            direction=1 if directional else None)
#
##                word_idxs = ((s0, w0), (None, metaidx.LEAF))
##                # By convention, LEAF is a left attachment (direction = -1)
##                cls.add_dependency(lp, feats, instance, word_idxs,
##                        direction=-1 if directional else None)
#
#                # For dependencies within a sentence, optionally don't allow
#                # connections that flip existing dependencies or invert
#                # the ancestry.
#                pruned_idxs = set([w0])
#                if noninv_deps:
#                    pruned_idxs.update(
#                            sentence0.dparse.get_descendant_idxs(w0))
#
#                for s1, sentence1 in enumerate(instance.input_sents):
#                    for w1 in range(len(sentence1.tokens)):
#                        if s0 == s1 and w1 in pruned_idxs:
#                            continue
#
#                        # Note the inversion of indices below. We want to
#                        # prevent a potential w0 -> w1 dependency when w0
#                        # is currently a descendent of w1. To check for this,
#                        # we would need to enumerate each w1's subtree
#                        # for each w0. For efficiency, however, we choose to
#                        # generate subtrees for *w0* in the outer loop and so
#                        # we're now creating w1 -> w0 dependencies in the
#                        # inner loop.
#                        word_idxs = ((s1, w1), (s0, w0))
#                        direction = None
#                        if directional and s0 == s1:
#                            # We define direction as -1 for left attachment
#                            # (head follows dependent) and +1 for right
#                            # attachment (head precedes attachment). In
#                            # other words,
#                            # direction = sign(dep_idx - head_idx)
#                            direction = np.sign(w0 - w1)
#                        cls.add_dependency(lp, feats, instance, word_idxs,
#                                           direction=direction)

    @classmethod
    def add_frames_and_fes(cls, lp, feats, instance, orig_fes=False, **kwargs):
        """Add frame and frame element variables for frame-semantic parsing.
        """
        for s, sentence in enumerate(instance.input_sents):
            for f in sentence.frames.aux_idxs:
                frame_idx = (s, f)
                frame_node = cls.add_frame(lp, feats, instance, frame_idx)

                # Assume for now that frames are restricted to associations
                # with words in their originating sentence.
                if orig_fes:
                    # We ignore target edges here since they're tied
                    # to the frame in the constraints
                    w_list = [w for w, edge in frame_node.outgoing_edges
                                if hasattr(edge, 'fe')]
                else:
                    w_list = range(len(sentence.tokens))

                for w in w_list:
                    fe_idxs = ((s, f), (s, w))
                    cls.add_fe(lp, feats, instance, fe_idxs)
