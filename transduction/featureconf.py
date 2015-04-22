#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import itertools
from lexical.resources import framenet
from transduction import model
from model import pos_tags, chunk_tags, gr_types, label_hierarchy


class TransductionFeatureConfigs(model.TransductionFeatures):
    """A collection of classmethods that specify groups of features for use
    in transduction problems.
    """
    @classmethod
    def word(cls, normalizers=(None,), corpus_label_pos=(),
            corpus_pos_pairs=(), corpus_pos_triples=(), lex_words=(),
            verb_stems=(), fn_words=(), **kwargs):
        """Features over individual tokens.
        """
#        pos_threshold_tuples = \
#                list(itertools.product([None,'NN','VB','JJ','RB'],
#                                       [None, 2, 3, 4]))
        return [('word_fidelity', normalizers,),
                ('word_norm', normalizers,),
#                ('word_support', cls.to_dict(pos_threshold_tuples),
#                    normalizers,),
# Uninformative
#                ('word_depth', False),
#                ('word_depth', True),
#                ('word_tfidf',),
#                ('word_significance',),
                ('word_capitalization', normalizers,),
                ('word_capitalization_seq', False,),
                ('word_capitalization_seq', True,),
                ('word_in_parens',),
# Removed for fusion
                ('word_negation', normalizers,),
                ('word_lex', cls.to_dict(fn_words), [-1,0,1], False,
                    normalizers,),
#                ('word_lex', cls.to_dict(lex_words), [-1,0,1], False,
#                    normalizers,),
#                ('word_lex', cls.to_dict(verb_stems), [-1,0,1], True,
#                    normalizers,), # till here
                ('word_label_pos', cls.to_dict(corpus_label_pos), 1,
                    normalizers,),
                ('word_pos_seq', cls.to_dict([(pos,) for pos in pos_tags]),
                    [(i,) for i in -2,-1,0,1,2], normalizers,),
                ('word_pos_seq', cls.to_dict(corpus_pos_pairs),
                    [(i, i+1) for i in -2,-1,0,1], normalizers,),
# Removed for fusion
                ('word_pos_seq', cls.to_dict(corpus_pos_triples),
                    [(i, i+1, i+2) for i in -2,-1,0], normalizers,),
                ]

    @classmethod
    def wordf(cls, normalizers=(None,), corpus_label_pos=(),
            corpus_pos_pairs=(), corpus_pos_triples=(), lex_words=(),
            verb_stems=(), **kwargs):
        """Features over individual tokens.
        """
#        pos_threshold_tuples = \
#                list(itertools.product([None,'NN','VB','JJ','RB'],
#                                       [None, 2, 3, 4]))
        return [('word_fidelity', normalizers,),
                ('word_norm', normalizers,),
#                ('word_support', cls.to_dict(pos_threshold_tuples),
#                    normalizers,),
# Uninformative
#                ('word_depth', False),
#                ('word_depth', True),
#                ('word_tfidf',),
#                ('word_significance',),
                ('word_capitalization', normalizers,),
                ('word_capitalization_seq', False,),
                ('word_capitalization_seq', True,),
                ('word_in_parens',),
# Removed for fusion
#                ('word_negation', normalizers,),
#                ('word_lex', cls.to_dict(lex_words), [-1,0,1], False,
#                    normalizers,),
#                ('word_lex', cls.to_dict(verb_stems), [-1,0,1], True,
#                    normalizers,), # till here
                ('word_label_pos', cls.to_dict(corpus_label_pos), 1,
                    normalizers,),
                ('word_pos_seq', cls.to_dict([(pos,) for pos in pos_tags]),
                    [(i,) for i in -2,-1,0,1,2], normalizers,),
                ('word_pos_seq', cls.to_dict(corpus_pos_pairs),
                    [(i, i+1) for i in -2,-1,0,1], normalizers,),
# Removed for fusion
#                ('word_pos_seq', cls.to_dict(corpus_pos_triples),
#                    [(i, i+1, i+2) for i in -2,-1,0], normalizers,),
                ]

    @classmethod
    def ngram(cls, normalizers=(None,), fn_words=(), ngram_order=3, **kwargs):
        """Features over ngrams.
        """
        all_pos_pairs = \
                list(itertools.product(['START'], pos_tags)) + \
                list(itertools.product(pos_tags, repeat=2)) + \
                list(itertools.product(pos_tags, ['END']))
        all_labels = sorted(label_hierarchy.keys())
#        all_grels = sorted(gr_types)

        templates = [
#                ('ngram_norm', normalizers,),
                ('ngram_fidelity',),
                ('ngram_pos_seq', cls.to_dict(all_pos_pairs), normalizers,),
                ('ngram_label', cls.to_dict(all_labels), 1, False,
                    normalizers,),
#                ('ngram_label', cls.to_dict(all_grels), 1, True,
#                    normalizers,),
                ('ngram_lm_prob',),
#                ('ngram_lm_fixed',),
                ('ngram_lex', cls.to_dict(fn_words), False, 0, normalizers),
                ('ngram_lex', cls.to_dict(fn_words), False, 1, normalizers),
                ]

        if ngram_order >= 3:
            all_pos_triples = \
                    list(itertools.product(['START'], pos_tags, pos_tags)) + \
                    list(itertools.product(pos_tags, repeat=3)) + \
                    list(itertools.product(pos_tags, pos_tags, ['END']))
            templates += [
                ('ngram_pair_fidelity',),
                ('ngram_pos_seq', cls.to_dict(all_pos_triples), normalizers,),
                ('ngram_lex', cls.to_dict(fn_words), False, 2, normalizers),
                ]

        return templates

    @classmethod
    def dep(cls, fn_words=(), corpus_label_pos=(), corpus_pos_pairs=(),
            normalizers=(None,), **kwargs):
        """Features over dependency arcs.
        """
        all_labels = sorted(label_hierarchy.keys())
        fid_dir_pos_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT', None],
                                                    pos_tags + [None]))
#        anc_dir_pos_tuples = list(itertools.product([1, 2, 5, 10, 20],
#                                                    [None, -1, 1],
#                                                    pos_tags + ['ROOT'],
#                                                    [None]))
#        fid_dir_chk_tuples = list(itertools.product([None, 0, 1],
#                                                    [None, -1, 1],
#                                                    chunk_tags + [None],
#                                                    chunk_tags + [None, '=']))
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    fn_words + [None],
                                                    [None])) + \
                             list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    fn_words + [None]))
        fid_dir_span_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    pos_tags + [None],
                                                    pos_tags + [None]))

        return [#('dep_norm', normalizers,),
#                ('dep_fid_label', cls.to_dict(gr_types), None, True,
#                    normalizers,),
                ('dep_fid_label', cls.to_dict(all_labels), 1, False,
                    normalizers,),
#                ('dep_fid_label', cls.to_dict([None]), None, False,
#                    normalizers,),
#                ('dep_cond_prob', [False], 'stem',),
                ('dep_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
#                ('dep_anc_dir_pos', cls.to_dict(anc_dir_pos_tuples), True,
#                    normalizers,),
#                ('dep_fid_dir_chk', cls.to_dict(fid_dir_chk_tuples),
#                    normalizers,),
                ('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
                    normalizers,),
                ('dep_fid_dir_span', cls.to_dict(fid_dir_span_tuples),
                    normalizers,),
                ('dep_label_pos', cls.to_dict(corpus_label_pos), 1, 1,
                    normalizers,),
                ('dep_pos_seq', cls.to_dict(corpus_pos_pairs),
                    [(-1, 0), (0, 1), (-1, 1)], 1, normalizers,),
                ]

    @classmethod
    def dep2(cls, preps=(), normalizers=(None,), **kwargs):
        """Features over second-order dependency arcs.
        """
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         pos_tags + [None]))
        fid_dir_pos_dist_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 0, 1],
                                                         [None],
                                                         [None, 1],
                                                         pos_tags + [None])) \
                                + list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None, 0, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 1],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                 ('dep2_dir_pos_dist', cls.to_dict(fid_dir_pos_dist_tuples),
                    normalizers),
                ]

    @classmethod
    def frame(cls, normalizers=(None,), **kwargs):
        fe_coretypes = framenet.fe_coretypes + \
                        [(fe, None) for fe in framenet.fes] + \
                        [(None, coretype) for coretype in framenet.coretypes]

        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    None, fe, coretype,
                                    None, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for fe, coretype in fe_coretypes]
        return [('frame_name', cls.to_dict(framenet.frames), 2, normalizers),
                ('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def clarke(cls, **kwargs):
        return [('word_significance',),
                ('ngram_lm_fixed',),
                ]

    @classmethod
    def clarkewn(cls, **kwargs):
        return [('word_significance_fixed', 1.8),
                ('ngram_lm_fixed',),
                ]

    @classmethod
    def clarkebn(cls, **kwargs):
        return [('word_significance_fixed', 2.2),
                ('ngram_lm_fixed',),
                ]

    @classmethod
    def lm(cls, **kwargs):
        return [('ngram_lm',),
                ]

    @classmethod
    def lmn(cls, **kwargs):
        return [('ngram_lm_normed',),
                ]

    @classmethod
    def lmp(cls, **kwargs):
        return [('ngram_lm_prob',),
                ]

    @classmethod
    def lmf(cls, **kwargs):
        return [('ngram_lm_fixed',),
                ]

    # NO-OP for words + ngrams
    @classmethod
    def norm(cls, normalizers=(None,), **kwargs):
        return [('word_norm', normalizers,),
                ('ngram_norm', normalizers,),
                ]

    # NO-OP for words
    @classmethod
    def wnorm(cls, normalizers=(None,), **kwargs):
        return [('word_norm', normalizers,),
                ]

    @classmethod
    def sig(cls,**kwargs):
        return [('word_significance',),
                ]

    @classmethod
    def tf(cls, **kwargs):
        return [('word_tfidf',),
                ]

    @classmethod
    def df(cls, **kwargs):
        return [('word_depth', False),
                ]

    @classmethod
    def dt(cls, **kwargs):
        return [('word_depth', True),
                ]

    @classmethod
    def wsups(cls, **kwargs):
        return cls.wsup(**kwargs) + \
                cls.wgovsup(**kwargs) + \
                cls.wsubsup(**kwargs)

    @classmethod
    def wsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [None,2,3,4]))
        return [('word_support', cls.to_dict(pos_threshold_tuples),
                    normalizers),
                ]

    @classmethod
    def wgovsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [None,2,3,4]))
        return [('word_gov_support', cls.to_dict(pos_threshold_tuples),
                    normalizers),
                ]

    @classmethod
    def wsubsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [2,3,4]))
        return [('word_subtree_support', cls.to_dict(pos_threshold_tuples),
                    normalizers),
                ]

    @classmethod
    def wpos(cls, corpus_pos_pairs=(), corpus_pos_triples=(),
            normalizers=(None,), **kwargs):
        return [('word_pos_seq', cls.to_dict([(pos,) for pos in pos_tags]),
                    [(i,) for i in -2,-1,0,1,2], normalizers,),
                ('word_pos_seq', cls.to_dict(corpus_pos_pairs),
                    [(i, i+1) for i in -2,-1,0,1], normalizers,),
                ('word_pos_seq', cls.to_dict(corpus_pos_triples),
                    [(i, i+1, i+2) for i in -2,-1,0], normalizers,),
                ]

    @classmethod
    def wlab1(cls, corpus_label_pos=None, normalizers=(None,), **kwargs):
        return [('word_label_pos', cls.to_dict(corpus_label_pos), 1,
                normalizers,),
                ]
    @classmethod
    def wlab2(cls, corpus_label_pos=None, normalizers=(None,), **kwargs):
        return [('word_label_pos', cls.to_dict(corpus_label_pos), 2,
                normalizers,),
                ]
    @classmethod
    def wlab3(cls, corpus_label_pos=None, normalizers=(None,), **kwargs):
        return [('word_label_pos', cls.to_dict(corpus_label_pos), 3,
                normalizers,),
                ]
    @classmethod
    def wlab4(cls, corpus_label_pos=None, normalizers=(None,), **kwargs):
        return [('word_label_pos', cls.to_dict(corpus_label_pos), 4,
                normalizers,),
                ]

    @classmethod
    def wvrb(cls, verb_stems=(), normalizers=(None,), **kwargs):
        return [('word_lex', cls.to_dict(verb_stems), [-1,0,1], True,
                    normalizers,),
                ]

    @classmethod
    def wlex(cls, lex_words=(), normalizers=(None,), **kwargs):
        return [('word_lex', cls.to_dict(lex_words), [-1,0,1], False,
                    normalizers,),
                ]

    @classmethod
    def nnorm(cls, normalizers=(None,), **kwargs):
        return [('ngram_norm', normalizers,),
                ]

    @classmethod
    def nfid(cls, normalizers=(None,), **kwargs):
        return [('ngram_fidelity',),
                ]

    # NO-OP for bigrams
    @classmethod
    def npfid(cls, **kwargs):
        return [('ngram_pair_fidelity',),
                ]

    # Best shallowness parameter is 4 although 1 is similar
    @classmethod
    def nlab1(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('ngram_label', cls.to_dict(all_labels), 1, False,
                    normalizers,),
                ]
    @classmethod
    def nlab2(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('ngram_label', cls.to_dict(all_labels), 2, False,
                    normalizers,),
                ]
    @classmethod
    def nlab3(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('ngram_label', cls.to_dict(all_labels), 3, False,
                    normalizers,),
                ]
    @classmethod
    def nlab4(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('ngram_label', cls.to_dict(all_labels), 4, False,
                    normalizers,),
                ]
    @classmethod
    def ngrel(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(gr_types)
        return [('ngram_label', cls.to_dict(all_labels), 1, True,
                    normalizers,),
                ]

    @classmethod
    def npos2(cls, normalizers=(None,), **kwargs):
        all_pos_pairs = \
                list(itertools.product(['START'], pos_tags)) + \
                list(itertools.product(pos_tags, repeat=2)) + \
                list(itertools.product(pos_tags, ['END']))
        return [('ngram_pos_seq', cls.to_dict(all_pos_pairs), normalizers,),
                ]

    @classmethod
    def nlexfn(cls, fn_words=(), normalizers=(None,), **kwargs):
        return [('ngram_lex', cls.to_dict(fn_words), False, 0, normalizers),
                ('ngram_lex', cls.to_dict(fn_words), False, 1, normalizers),
                ]

    @classmethod
    def nsups(cls, **kwargs):
        return cls.nsup(**kwargs) + \
                cls.ntoksup(**kwargs) + \
                cls.ngovsup(**kwargs) + \
                cls.nsubsup(**kwargs)

    @classmethod
    def nsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = list(itertools.product(pos_tags + [None],
                                                      pos_tags + [None],
                                                      [None,2,3,4]))
        return [('ngram_pair_support', cls.to_dict(pos_threshold_tuples),
                    normalizers),
                ]

    @classmethod
    def ntoksup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [None,2,3,4]))
        return [('ngram_tok_support', cls.to_dict(pos_threshold_tuples), 0,
                    normalizers),
                ('ngram_tok_support', cls.to_dict(pos_threshold_tuples), 1,
                    normalizers),
                ]

    @classmethod
    def ngovsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [None,2,3,4]))
        return [('ngram_gov_support', cls.to_dict(pos_threshold_tuples), 0,
                    normalizers),
                ('ngram_gov_support', cls.to_dict(pos_threshold_tuples), 1,
                    normalizers),
                ]

    @classmethod
    def nsubsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [2,3,4]))
        return [('ngram_subtree_support', cls.to_dict(pos_threshold_tuples),
                    0, normalizers),
                ('ngram_subtree_support', cls.to_dict(pos_threshold_tuples),
                    1, normalizers),
                ]

    @classmethod
    def dlab1(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('dep_fid_label', cls.to_dict(all_labels), 1, False,
                    normalizers,),
                ]
    @classmethod
    def dlab2(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('dep_fid_label', cls.to_dict(all_labels), 2, False,
                    normalizers,),
                ]
    @classmethod
    def dlab3(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('dep_fid_label', cls.to_dict(all_labels), 3, False,
                    normalizers,),
                ]
    @classmethod
    def dlab4(cls, normalizers=(None,), **kwargs):
        all_labels = sorted(label_hierarchy.keys())
        return [('dep_fid_label', cls.to_dict(all_labels), 4, False,
                    normalizers,),
                ]
    @classmethod
    def dlabn(cls, normalizers=(None,), **kwargs):
        return [('dep_fid_label', cls.to_dict([None]), None, False,
                    normalizers,),
                ]
    @classmethod
    def dlabgr(cls, normalizers=(None,), **kwargs):
        return [('dep_fid_label', cls.to_dict(gr_types), None, True,
                    normalizers,),
                ]

    @classmethod
    def dlabposg(cls, corpus_label_pos=None, normalizers=(None,), **kwargs):
        return [('dep_label_pos', cls.to_dict(corpus_label_pos), 4, 0,
                    normalizers,),
                ]
    @classmethod
    def dlabposd(cls, corpus_label_pos=None, normalizers=(None,), **kwargs):
        return [('dep_label_pos', cls.to_dict(corpus_label_pos), 4, 1,
                    normalizers,),
                ]

    @classmethod
    def dposseqg(cls, corpus_pos_pairs=(), normalizers=(None,), **kwargs):
        return [('dep_pos_seq', cls.to_dict(corpus_pos_pairs),
                    [(-1, 0), (0, 1), (-1, 1)], 0, normalizers,),
                ]
    @classmethod
    def dposseqd(cls, corpus_pos_pairs=(), normalizers=(None,), **kwargs):
        return [('dep_pos_seq', cls.to_dict(corpus_pos_pairs),
                    [(-1, 0), (0, 1), (-1, 1)], 1, normalizers,),
                ]

    @classmethod
    def dprobf(cls, normalizers=(None,), **kwargs):
        return [('dep_cond_prob', [False], 'stem',),
                ]
    @classmethod
    def dprobt(cls, normalizers=(None,), **kwargs):
        return [('dep_cond_prob', [True], 'stem',),
                ]

    @classmethod
    def ddir(cls, normalizers=(None,), **kwargs):
        dir_tuples = itertools.product([None, -1, 0, 1], [-1, 1])
        return [('dep_dir', cls.to_dict(dir_tuples)),
                ]

    @classmethod
    def ddist(cls, normalizers=(None,), **kwargs):
        return [('dep_dist',),
                ]

    @classmethod
    def dposd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    pos_tags))
        return [('dep_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
                ]

    @classmethod
    def dposg(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT'],
                                                    [None]))
        return [('dep_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
                ]

    @classmethod
    def ddir2(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None], [None]))
        return [('dep_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
                ]

    @classmethod
    def dposdposg(cls, normalizers=(None,), **kwargs):
        fid_dir_pnp_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT', None],
                                                    pos_tags + [None]))
        return [('dep_fid_dir_pos', cls.to_dict(fid_dir_pnp_tuples),
                    normalizers,),
                ]

    @classmethod
    def dspand(cls, normalizers=(None,), **kwargs):
        fid_dir_pnp_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    pos_tags + [None],
                                                    pos_tags + [None]))
        return [('dep_fid_dir_span', cls.to_dict(fid_dir_pnp_tuples),
                    normalizers,),
                ]
    @classmethod
    def dspang(cls, normalizers=(None,), **kwargs):
        fid_dir_pnp_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT', None],
                                                    [None],
                                                    pos_tags + [None]))
        return [('dep_fid_dir_span', cls.to_dict(fid_dir_pnp_tuples),
                    normalizers,),
                ]
    @classmethod
    def dspandg(cls, normalizers=(None,), **kwargs):
        fid_dir_pnp_tuples = list(itertools.product([None],
                                                    [None],
                                                    pos_tags + ['ROOT', None],
                                                    pos_tags + [None],
                                                    pos_tags + [None]))
        return [('dep_fid_dir_span', cls.to_dict(fid_dir_pnp_tuples),
                    normalizers,),
                ]

    @classmethod
    def dlexd(cls, normalizers=(None,), preps=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    preps))
        return [('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
                    normalizers,),
                ]

    @classmethod
    def dlexg(cls, normalizers=(None,), verb_stems=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    verb_stems,
                                                    [None]))
        return [('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), True,
                    normalizers,),
                ]

    @classmethod
    def dlexdlexg(cls, normalizers=(None,), verb_stems=(), prep_stems=(),
            **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    verb_stems + [None],
                                                    prep_stems + [None]))
        return [('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), True,
                    normalizers,),
                ]

    @classmethod
    def dlexfn(cls, normalizers=(None,), fn_words=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    fn_words + [None],
                                                    fn_words + [None]))
        return [('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
                    normalizers,),
                ]

    @classmethod
    def dlexfnd(cls, normalizers=(None,), fn_words=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    fn_words + [None]))
        return [('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
                    normalizers,),
                ]

    @classmethod
    def dlexfng(cls, normalizers=(None,), fn_words=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    fn_words + [None],
                                                    [None]))
        return [('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
                    normalizers,),
                ]

    @classmethod
    def dlexfndg(cls, normalizers=(None,), fn_words=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    fn_words + [None])) + \
                             list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    fn_words + [None],
                                                    [None]))
        return [('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
                    normalizers,),
                ]


    @classmethod
    def dchk(cls, normalizers=(None,), **kwargs):
        fid_dir_chk_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    chunk_tags + [None],
                                                    chunk_tags + [None, '=']))
        return [('dep_fid_dir_chk', cls.to_dict(fid_dir_chk_tuples),
                    normalizers,),
                ]

    @classmethod
    def daposd(cls, normalizers=(None,), **kwargs):
        anc_dir_pos_tuples = list(itertools.product([1,2,5,10,20],
                                                    [None, -1, 1],
                                                    [None], pos_tags))
        return [('dep_anc_dir_pos', cls.to_dict(anc_dir_pos_tuples), True,
                    normalizers,),
                ]

    @classmethod
    def daposg(cls, normalizers=(None,), **kwargs):
        anc_dir_pos_tuples = list(itertools.product([1,2,5,10,20],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT'],
                                                    [None]))
        return [('dep_anc_dir_pos', cls.to_dict(anc_dir_pos_tuples), True,
                    normalizers,),
                ]

    @classmethod
    def daposdposg(cls, normalizers=(None,), **kwargs):
        anc_dir_pnp_tuples = list(itertools.product([1,2,5,10,20],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT', None],
                                                    pos_tags + [None]))
        return [('dep_anc_dir_pos', cls.to_dict(anc_dir_pnp_tuples), True,
                    normalizers,),
                ]

    @classmethod
    def dsups(cls, **kwargs):
        return cls.dsup(**kwargs) + \
                cls.dtoksup(**kwargs) + \
                cls.dgovsup(**kwargs) + \
                cls.dsubsup(**kwargs)

    @classmethod
    def dsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = list(itertools.product(pos_tags + [None],
                                                      pos_tags + [None],
                                                      [None,2,3,4]))
        return [('dep_support', cls.to_dict(pos_threshold_tuples),
                    normalizers),
                ]

    @classmethod
    def dtoksup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [None,2,3,4]))
        return [('dep_tok_support', cls.to_dict(pos_threshold_tuples), 0,
                    normalizers),
                ('dep_tok_support', cls.to_dict(pos_threshold_tuples), 1,
                    normalizers),
                ]

    @classmethod
    def dgovsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [None,2,3,4]))
        return [('dep_gov_support', cls.to_dict(pos_threshold_tuples), 0,
                    normalizers),
                ('dep_gov_support', cls.to_dict(pos_threshold_tuples), 1,
                    normalizers),
                ]

    @classmethod
    def dsubsup(cls, normalizers=(None,), **kwargs):
        pos_threshold_tuples = \
                list(itertools.product([None,'NN','VB','JJ','RB'],
                                       [2,3,4]))
        return [('dep_subtree_support', cls.to_dict(pos_threshold_tuples), 0,
                    normalizers),
                ('dep_subtree_support', cls.to_dict(pos_threshold_tuples), 1,
                    normalizers),
                ]

    @classmethod
    def rnorm(cls, normalizers=(None,), **kwargs):
        return [('range_norm', normalizers,),
                ]

    @classmethod
    def rposd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    pos_tags))
        return [('range_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
                ]

    @classmethod
    def rposg(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT'],
                                                    [None]))
        return [('range_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
                ]

    @classmethod
    def rdir(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None], [None]))
        return [('range_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
                ]

    @classmethod
    def rposposd(cls, normalizers=(None,), **kwargs):
        fid_dir_pnp_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT', None],
                                                    pos_tags + [None]))
        return [('range_fid_dir_pos', cls.to_dict(fid_dir_pnp_tuples),
                    normalizers,),
                ]

    @classmethod
    def rlexd(cls, normalizers=(None,), preps=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    [None],
                                                    preps))
        return [('range_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
                    normalizers,),
                ]

    @classmethod
    def rlexg(cls, normalizers=(None,), verb_stems=(), **kwargs):
        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    verb_stems,
                                                    [None]))
        return [('range_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), True,
                    normalizers,),
                ]

    @classmethod
    def rchk(cls, normalizers=(None,), **kwargs):
        fid_dir_chk_tuples = list(itertools.product([None, 0, 1],
                                                    [None, -1, 1],
                                                    chunk_tags + [None],
                                                    chunk_tags + [None, '=']))
        return [('range_fid_dir_chk', cls.to_dict(fid_dir_chk_tuples),
                    normalizers,),
                ]

    @classmethod
    def anorm(cls, normalizers=(None,), **kwargs):
        return [('arity_norm', normalizers,),
                ]

    @classmethod
    def anpos(cls, normalizers=(None,), **kwargs):
        arity_label_pos = list(itertools.product([None],
                                                 [None],
                                                 pos_tags))
        return [('arity_label_pos', cls.to_dict(arity_label_pos), 2,
                    normalizers,),
                ]

    @classmethod
    def anlab(cls, normalizers=(None,), **kwargs):
        arity_label_pos = list(itertools.product([None],
                                                 label_hierarchy.keys(),
                                                 [None]))
        return [('arity_label_pos', cls.to_dict(arity_label_pos), 2,
                    normalizers,),
                ]

    @classmethod
    def anlabpos(cls, normalizers=(None,), **kwargs):
        arity_label_pos = list(itertools.product([None],
                                                 label_hierarchy.keys(),
                                                 pos_tags))
        return [('arity_label_pos', cls.to_dict(arity_label_pos), 2,
                    normalizers,),
                ]

    @classmethod
    def a6pos(cls, normalizers=(None,), **kwargs):
        arity_label_pos = list(itertools.product(range(6),
                                                 [None],
                                                 pos_tags))
        return [('arity_label_pos', cls.to_dict(arity_label_pos), 2,
                    normalizers,),
                ]

    @classmethod
    def a6lab(cls, normalizers=(None,), **kwargs):
        arity_label_pos = list(itertools.product(range(6),
                                                 label_hierarchy.keys(),
                                                 [None]))
        return [('arity_label_pos', cls.to_dict(arity_label_pos), 2,
                    normalizers,),
                ]

    @classmethod
    def a6labpos(cls, normalizers=(None,), **kwargs):
        arity_label_pos = list(itertools.product(range(6),
                                                 label_hierarchy.keys(),
                                                 pos_tags))
        return [('arity_label_pos', cls.to_dict(arity_label_pos), 2,
                    normalizers,),
                ]

    @classmethod
    def d2poss(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2posy(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None],
                                                         [None],
                                                         pos_tags + [None],
                                                         [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2posgd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         [None],
                                                         [None],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2possd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2posyd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None],
                                                         [None],
                                                         pos_tags + [None],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2possy(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         pos_tags + [None],
                                                         [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2posxy(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         pos_tags + [None],
                                                         [None],
                                                         pos_tags + [None],
                                                         [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2possyd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         pos_tags + [None],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2posgyd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         [None],
                                                         pos_tags + [None],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2posgsd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_span_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_span', cls.to_dict(fid_dir_pos_span_tuples),
                    normalizers),
                ]

    @classmethod
    def d2distgd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_dist_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 0, 1],
                                                         [None],
                                                         [None, 1],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_dist', cls.to_dict(fid_dir_pos_dist_tuples),
                    normalizers),
                ]

    @classmethod
    def d2distsd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_dist_tuples = list(itertools.product([None, -1, 1],
                                                         [None],
                                                         [None, 0, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 1],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_dist', cls.to_dict(fid_dir_pos_dist_tuples),
                    normalizers),
                ]

    @classmethod
    def d2distgs(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_dist_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 0, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 1],
                                                         [None]))
        return  [('dep2_dir_pos_dist', cls.to_dict(fid_dir_pos_dist_tuples),
                    normalizers),
                ]

    @classmethod
    def d2distgsd(cls, normalizers=(None,), **kwargs):
        fid_dir_pos_dist_tuples = list(itertools.product([None, -1, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 0, 1],
                                                         pos_tags + ['ROOT',
                                                                     None],
                                                         [None, 1, 2],
                                                         pos_tags + [None]))
        return  [('dep2_dir_pos_dist', cls.to_dict(fid_dir_pos_dist_tuples),
                    normalizers),
                ]

    @classmethod
    def wordx(cls, normalizers=(None,), corpus_label_pos=(),
            corpus_pos_pairs=(), corpus_pos_triples=(), lex_words=(),
            verb_stems=(), **kwargs):
        """Features over individual tokens.
        """
#        pos_threshold_tuples = \
#                list(itertools.product([None,'NN','VB','JJ','RB'],
#                                       [None, 2, 3, 4]))
        return [('word_fidelity', normalizers,),
                ('word_norm', normalizers,),
                ('word_capitalization', normalizers,),
                ('word_capitalization_seq', False,),
                ('word_capitalization_seq', True,),
                ('word_in_parens',),
# Removed for fusion
                ('word_negation', normalizers,),
                ('word_lex', cls.to_dict(lex_words), [-1,0,1], False,
                    normalizers,),
                ('word_lex', cls.to_dict(verb_stems), [-1,0,1], True,
                    normalizers,), # till here
                ('word_pos_seq', cls.to_dict([(pos,) for pos in pos_tags]),
                    [(i,) for i in -2,-1,0,1,2], normalizers,),
                ('word_pos_seq', cls.to_dict(corpus_pos_pairs),
                    [(i, i+1) for i in -2,-1,0,1], normalizers,),
# Removed for fusion
                ('word_pos_seq', cls.to_dict(corpus_pos_triples),
                    [(i, i+1, i+2) for i in -2,-1,0], normalizers,),
                ]

    @classmethod
    def ngramx(cls, normalizers=(None,), ngram_order=2, **kwargs):
        """Features over ngrams.
        """
        all_pos_pairs = \
                list(itertools.product(['START'], pos_tags)) + \
                list(itertools.product(pos_tags, repeat=2)) + \
                list(itertools.product(pos_tags, ['END']))

        templates = [
                ('ngram_norm', normalizers,),
                ('ngram_fidelity',),
                ('ngram_pos_seq', cls.to_dict(all_pos_pairs), normalizers,),
#                ('ngram_lm_fixed',),
                ]

        if ngram_order >= 3:
            all_pos_triples = \
                    list(itertools.product(['START'], pos_tags, pos_tags)) + \
                    list(itertools.product(pos_tags, repeat=3)) + \
                    list(itertools.product(pos_tags, pos_tags, ['END']))
            templates += [
                ('ngram_pair_fidelity',),
                ('ngram_pos_seq', cls.to_dict(all_pos_triples), normalizers,),
                ]

        return templates

    @classmethod
    def depx(cls, preps=(), normalizers=(None,), **kwargs):
        fid_dir_pos_tuples = list(itertools.product([None],
                                                    [None, -1, 1],
                                                    pos_tags + ['ROOT'],
                                                    [None])) + \
                             list(itertools.product([None],
                                                    [None, -1, 1],
                                                    [None],
                                                    pos_tags))
#        anc_dir_pos_tuples = list(itertools.product([1, 2, 5, 10, 20],
#                                                    [None, -1, 1],
#                                                    pos_tags + ['ROOT'],
#                                                    [None]))
        fid_dir_chk_tuples = list(itertools.product([None],
                                                    [None, -1, 1],
                                                    chunk_tags + [None],
                                                    chunk_tags + [None, '=']))
#        fid_dir_lex_tuples = list(itertools.product([None, 0, 1],
#                                                    [None, -1, 1],
#                                                    preps,
#                                                    [None]))

        return [#('dep_norm', normalizers,),
                ('dep_cond_prob', [False], 'stem',),
                ('dep_fid_dir_pos', cls.to_dict(fid_dir_pos_tuples),
                    normalizers,),
#                ('dep_anc_dir_pos', cls.to_dict(anc_dir_pos_tuples), True,
#                    normalizers,),
                ('dep_fid_dir_chk', cls.to_dict(fid_dir_chk_tuples),
                    normalizers,),
#                ('dep_fid_dir_lex', cls.to_dict(fid_dir_lex_tuples), False,
#                    normalizers,),
                ]

    # The best value for ancestor_limit appears to be 1 or 2
    @classmethod
    def frname0(cls, normalizers=(None,), **kwargs):
        return [('frame_name', cls.to_dict(framenet.frames), 0, normalizers),
                ]

    @classmethod
    def frname1(cls, normalizers=(None,), **kwargs):
        return [('frame_name', cls.to_dict(framenet.frames), 1, normalizers),
                ]

    @classmethod
    def frname2(cls, normalizers=(None,), **kwargs):
        return [('frame_name', cls.to_dict(framenet.frames), 2, normalizers),
                ]

    @classmethod
    def frname3(cls, normalizers=(None,), **kwargs):
        return [('frame_name', cls.to_dict(framenet.frames), 3, normalizers),
                ]

    # POS tags seem very useful but FE labels don't help.
    @classmethod
    def fetld(cls, normalizers=(None,), **kwargs):
        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    None, None, None,
                                    None, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldfra(cls, normalizers=(None,), **kwargs):
        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    frame, None, None,
                                    None, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for frame in framenet.frames + [None]]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldfrapos(cls, normalizers=(None,), **kwargs):
        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    frame, None, None,
                                    pos_tag, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for frame in framenet.frames + [None]
                                   for pos_tag in pos_tags + [None]]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldfracore(cls, normalizers=(None,), **kwargs):
        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    frame, None, coretype,
                                    None, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for frame in framenet.frames + [None]
                                   for coretype in framenet.coretypes
                                    + [None]]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldcore(cls, normalizers=(None,), **kwargs):
        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    None, None, coretype,
                                    None, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for coretype in framenet.coretypes
                                    + [None]]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldfecore(cls, normalizers=(None,), **kwargs):
        fe_coretypes = framenet.fe_coretypes + \
                        [(fe, None) for fe in framenet.fes] + \
                        [(None, coretype) for coretype in framenet.coretypes]

        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    None, fe, coretype,
                                    None, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for fe, coretype in fe_coretypes]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldfecorepos(cls, normalizers=(None,), **kwargs):
        fe_coretypes = framenet.fe_coretypes + \
                        [(fe, None) for fe in framenet.fes] + \
                        [(None, coretype) for coretype in framenet.coretypes]

        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    None, fe, coretype,
                                    pos_tag, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for fe, coretype in fe_coretypes
                                   for pos_tag in pos_tags + [None]]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldfecorelab(cls, normalizers=(None,), **kwargs):
        fe_coretypes = framenet.fe_coretypes + \
                        [(fe, None) for fe in framenet.fes] + \
                        [(None, coretype) for coretype in framenet.coretypes]

        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    None, fe, coretype,
                                    None, None, label)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for fe, coretype in fe_coretypes
                                   for label in sorted(label_hierarchy.keys())
                                    + [None]]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]

    @classmethod
    def fetldfrafecore(cls, normalizers=(None,), **kwargs):
        fra_fe_coretypes = framenet.frame_fe_coretypes + \
                    [(frame, None, coretype)
                            for frame in framenet.frames + [None]
                            for coretype in framenet.coretypes + [None]] + \
                    [(frame, fe, None)
                            for frame, fe in framenet.frame_fes] + \
                    [(None, fe, coretype)
                            for fe, coretype in framenet.fe_coretypes]

        frame_fe_pos_dep_tuples = [(in_tgt, in_lex, in_dep,
                                    frame, fe, coretype,
                                    None, None, None)
                                   for in_tgt in [True, False]
                                   for in_lex in [True, False]
                                   for in_dep in [True, False]
                                   for frame, fe, coretype in
                                    fra_fe_coretypes]
        return [('fe_frame_pos_dep', cls.to_dict(frame_fe_pos_dep_tuples),
                    normalizers),
                ]
