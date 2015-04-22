#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
import numpy as np
import sys
import text
from text import annotations
from transduction import instance as instancemod
from transduction.decoding.ilp import variables
from utils import avg, evaluation, jsonrpc, timer


class TransductionCorpus(text.Corpus):
    """A corpus of TransductionInstances.
    """
    def __init__(self, name, restore=True, path=None,
                 instancecls=instancemod.TransductionInstance):
        """Initialize the corpus metadata.
        """
        text.Corpus.__init__(self, name, restore, path, instancecls)

    def add_instance(self, sentences, idx=None, **kwargs):
        """Add a group of sentences representing an instance of a transduction
        problem.
        """
        new_instance = self.instancecls(sentences,
                idx=len(self.instances) if idx is None else idx,
                **kwargs)
        self.instances.append(new_instance)

    def get_instances(self, partition=None, debug_idxs=None, skip_idxs=()):
        """Return instances from the specific partition, overriding with
        a list of debugging instances and skipping particular indices.
        """
        return [instance
                for instance in self.retrieve_slice(name=partition,
                                                    idxs=debug_idxs)
                if instance.idx not in skip_idxs]

    def decode_instances(self, learner, partition=None, debug_idxs=None,
            skip_idxs=(), **kwargs):
        """Decode all instances within a named partition of the corpus,
        e.g., 'train', 'test', 'dev', and None for all instances.
        """
        tgt_instances = self.get_instances(partition=partition,
                                           debug_idxs=debug_idxs,
                                           skip_idxs=skip_idxs)
        learner.run(tgt_instances, **kwargs)
        return tgt_instances

    def get_transductions(self, learner, **kwargs):
        """Produce and return transductions for all instances in the given
        corpus partition.
        """
        tgt_instances = self.decode_instances(learner, **kwargs)
        num_instances = len(tgt_instances)

        transductions = []
        with timer.AvgTimer(num_instances):
            for i, instance in enumerate(learner):
                sys.stdout.write("Retrieving " + str(num_instances) +
                        " transductions: " + str(i+1) + '\r')
                if hasattr(instance, 'output_sent'):
                    transductions.append(instance.output_sent)
                else:
                    transductions.append(None)
        return transductions


class GoldTransductionCorpus(TransductionCorpus):
    """A corpus of GoldTransductionInstances that can be used to train a new
    model for transduction.
    """
    def __init__(self, name, restore=True, path=None):
        """Initialize the corpus metadata.
        """
        TransductionCorpus.__init__(self, name, restore, path,
                instancemod.GoldTransductionInstance)

    def add_instance(self, sentences, gold_sentences, idx=None, **kwargs):
        """Add a group of sentences representing an instance of a transduction
        problem along with corresponding gold output sentences.
        """
        new_instance = self.instancecls(sentences, gold_sentences,
                idx=len(self.instances) if idx is None else idx,
                **kwargs)
        self.instances.append(new_instance)

    def evaluate(self, learner,
            partition='test', debug_idxs=None, skip_idxs=(), decoder='ilp',
            n_eval=(1,2,3,4), streaming=True, overwritten_params=(),
            eval_path=None, output_path=None, lm_proxy=None, **kwargs):
        """Run the transduction model on designated test instances and report
        performance metrics.
        """
        # When evaluating multiple iterations of the same model over a fixed
        # partition, decoding should ensure that initialization isn't
        # unnecessarily repeated.
        if learner is not None:
            eval_instances = self.decode_instances(learner,
                                                   partition=partition,
                                                   debug_idxs=debug_idxs,
                                                   skip_idxs=skip_idxs,
                                                   decoder=decoder,
                                                   streaming=streaming,
                                                   overwritten_params=\
                                                           overwritten_params,
                                                   **kwargs)
            system_name = learner.name
        else:
            eval_instances = self.get_instances(partition=partition,
                                                debug_idxs=debug_idxs,
                                                skip_idxs=skip_idxs)
            system_name = 'baseline'
        num_instances = len(eval_instances)

        # Record overwritten parameters in the filenames
        overwriting_str = None
        if len(overwritten_params) > 0:
            overwriting_str = '_OW-'
            i = 0
            for param_name, value in overwritten_params.iteritems():
                if isinstance(value, list) or isinstance(value, tuple):
                    overwriting_str += '+'.join(str(v) for v in sorted(value))
                else:
                    overwriting_str += str(value)
                i += 1
                if i < len(overwritten_params):
                    overwriting_str += '-'

        if output_path is not None:
            output_filename = ''.join((output_path, '/',
                    '_'.join((partition, 'under', system_name)),
                    overwriting_str if overwriting_str is not None else '',
                    '_', decoder, '.out'))
            outf = open(output_filename, 'wb')

        # Determine the evaluations to run by looking at a representative
        # instance
        i = 0
        while i < len(eval_instances) and \
                not hasattr(eval_instances[i], 'output_sent'):
            i += 1
        if i == len(eval_instances):
            print "WARNING: all instances failed; skipping evaluation"
            sys.exit()
        some_instance = eval_instances[i]
        has_labels = hasattr(some_instance, 'label_sentences')
        has_rasp = hasattr(some_instance.gold_sentences[0], 'relgraph')
        has_outtrees = hasattr(some_instance.output_sent, 'outtree')
        has_outframes = hasattr(some_instance.output_sent, 'outframes')

        # FIXME TEMPORARY! MUST MAKE "False" FOR TEST!
        skip_failed = False

        # Initialize the evaluations
        eval_obj = evaluation.Evaluation(title='TRANSDUCTION_EVAL')
        output_sents = []
        with timer.AvgTimer(num_instances):
            for i, instance in enumerate(eval_instances):
                sys.stdout.write("Evaluating " + str(num_instances) +
                        (" " + partition if partition is not None else "") +
                        " instances: " + str(i+1) + '\r')

                # Duration and failure status
                eval_obj.include(
                        system=system_name,
                        corpus='other',
                        decode_time=instance.decode_times[-1],
                        solution_time=instance.solution_times[-1] \
                                if len(instance.solution_times) > 0 else 0,
                        inputs=len(instance.input_sents),
                        _failed=int(not hasattr(instance, 'output_sent')),
                        )

                if skip_failed and not hasattr(instance, 'output_sent'):
                    print "WARNING: Skipping failed instance", instance.idx
                    continue

                # POS tag recall
                for use_labels in set([False]) | set([has_labels]):
                    for prefix in ('NN', 'VB', 'JJ', 'RB'):
                        p, r, f = instance.score_content_words(
                                use_labels=use_labels, prefixes=(prefix,))
                        eval_obj.add_metrics(
                                precision=p,
                                recall=r,
                                system=system_name,
                                corpus=('LBLs ' + prefix) if use_labels \
                                        else ('GOLD ' + prefix),
                                )

                try:
                    if lm_proxy is not None:
                        output_tokens = instance.output_sent.tokens \
                                if hasattr(instance, 'output_sent') else []
                        eval_obj.include(
                                system=system_name,
                                corpus='other',
                                lm=lm_proxy.score_sent(output_tokens)
                                )
                except jsonrpc.RPCTransportError:
                    print "ERROR: JSON-RPC hiccups; skipping LM scoring"
                    pass

                if decoder.startswith('dp+'):
                        # Record convergence of dual decomposition or
                        # bisection. Will be 0 if neither are used.
                        eval_obj.include(
                                system=system_name,
                                corpus='other',
                                convergence_=int(instance.converged),
                                iterations=instance.num_iterations,
                                )

                if len(instance.sentences) == 1:
                    # Paraphrasing or compression-specific metrics
                    eval_obj.include(
                            system=system_name,
                            corpus='STATS gold',
                            comp_=instance.get_gold_compression_rate(),
                            length=instance.avg_gold_len,
                            proj_=avg(int(gold_sent.dparse.is_projective())
                                for gold_sent in instance.gold_sentences),
                            overlap_=avg(instance.get_overlap(gold_sent)
                                for gold_sent in instance.gold_sentences),
                            )
                    eval_obj.include(
                            system=system_name,
                            corpus='STATS input',
                            comp_=1.0,
                            length=instance.avg_len,
                            proj_=int(
                                instance.sentences[0].dparse.is_projective()),
                            overlap_=instance.get_overlap(
                                instance.sentences[0])
                            )
                    eval_obj.include(
                            system=system_name,
                            corpus='STATS output',
                            comp_=instance.get_compression_rate(),
                            length=len(instance.output_sent.tokens)
                                    if hasattr(instance, 'output_sent') else 0,
                            )
                    if hasattr(instance, 'output_sent') and has_outtrees:
                        eval_obj.include(
                                system=system_name,
                                corpus='STATS output',
                                proj_=int(instance.output_sent.\
                                          outtree.is_projective())
                                      if hasattr(instance.output_sent.outtree,\
                                                 'is_projective')
                                      else 0,
                                overlap_=instance.get_overlap(
                                    instance.output_sent,
                                    parse_type='outtree')
                                )

#                    print "INSTANCE ", instance.idx
#                    crossing_edges = \
#                        instance.output_sent.outtree.get_crossing_edges()
#                    print "\n\nINPUT:",
#                    self.dump_parse(instance.sentences[0])
#
#                    for gs, gold_sent in enumerate(
#                            instance.gold_sentences):
#                        # get output indices for gold
#                        gold_idxs = []
#                        i = 0
#                        for token in gold_sent.tokens:
#                            while instance.sentences[0].tokens[i] != token:
#                                i += 1
#                            gold_idxs.append((0,i))
#
#                        print "\nGOLD:", gs,
#                        self.dump_parse(gold_sent,
#                            idx_mapper=gold_idxs)
#
#                    print "\n\nOUTPUT:",
#                    self.dump_parse(instance.output_sent,
#                            parse_type='outtree',
#                            crossing_edges=crossing_edges,
#                            idx_mapper=instance.output_idxs)

                # n-gram precision and recall
                for use_labels in set([False]) | set([has_labels]):
                    for n in n_eval:
                        p, r, f = instance.score_ngrams(n=n,
                                use_labels=use_labels)
                        eval_obj.add_metrics(
                                precision=p,
                                recall=r,
                                system=system_name,
                                corpus='LBLs n='+str(n) if use_labels else
                                       'GOLD n='+str(n),
                                )
                if hasattr(instance, 'output_sent') and has_outframes:
                    # Precision and recall for frames
                    p, r, f = instance.score_frames(fes=False,
                                                    frames_type='outframes',
                                                    use_labels=use_labels)
                    eval_obj.add_metrics(
                            precision=p,
                            recall=r,
                            system=system_name,
                            corpus="GOLD frames",
                            )

                    # Precision and recall for frame elements
                    p, r, f = instance.score_frames(fes=True,
                                                    frames_type='outframes',
                                                    use_labels=use_labels)
                    eval_obj.add_metrics(
                            precision=p,
                            recall=r,
                            system=system_name,
                            corpus="GOLD fes",
                            )

                # Parse output sentences for syntactic evaluation. The
                # 100 token limit is intended for the Stanford parser.
                if hasattr(instance, 'output_sent') and \
                        len(instance.output_sent.tokens) <= 100:
                    output_sents.append(instance.output_sent)

                # Write the output to a file
                if output_path is not None:
                    outf.write(instance.get_display_string())
#            print
            if output_path is not None:
                outf.close()

            # Parse-based evaluations
            try:
                parse_types = ['dparse']
                if has_outtrees:
                    parse_types.append('outtree')

                # Get annotations. Only run RASP if the inputs have RASP
                # annotations since it's slow
                annotations.annotate(output_sents, 'Stanford')
                if has_rasp:
                    annotations.annotate(output_sents, 'Rasp')
                    parse_types.append('relgraph')

                # Add dependency results to evaluations
                for i, instance in enumerate(eval_instances):
                    if skip_failed and not hasattr(instance, 'output_sent'):
                        print "WARNING: Skipping failed instance",
                        print instance.idx, "again"
                        continue

                    for parse_type in parse_types:
                        for use_labels in set([False]) | set([has_labels]):
                            name = ('LBLs ' if use_labels else 'GOLD ') + \
                                parse_type
                            p, r, f = instance.score_dependencies(
                                    parse_type=parse_type,
                                    use_labels=use_labels)
                            eval_obj.add_metrics(
                                    precision=p,
                                    recall=r,
                                    system=system_name,
                                    corpus=name,
                                    _failed=int(not instance.has_output_parses(
                                            parse_type=parse_type)))
            except OSError:
                print "Skipping parser evaluations"

        print eval_obj.title
        print eval_obj.table(skip_single_keys=True)
        if eval_path is not None and debug_idxs is None:
            eval_filename = ''.join((eval_path, '/',
                    '_'.join((partition, 'under', system_name)),
                    overwriting_str if overwriting_str is not None else '',
                    '_', decoder,
                    '.eval'))
            eval_obj.save(eval_filename, append=False)

    def set_label_baseline(self, partition='test', debug_idxs=None,
            skip_idxs=()):
        """Generate a fake baseline using the label which shares the maximum
        overlap with the gold.
        """
        tgt_instances = self.get_instances(partition=partition,
                                           debug_idxs=debug_idxs,
                                           skip_idxs=skip_idxs)
        for instance in tgt_instances:
            gold_stems = set(stem for gold_sent in instance.gold_sentences
                    for stem in gold_sent.stems)

            max_overlap = 0
            max_overlap_idx = -1
            for l, label_sent in enumerate(instance.label_sentences):
                overlap = len(gold_stems.intersection(label_sent.stems))
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_idx = l

            instance.output_sent = instance.label_sentences[max_overlap_idx]
            instance.output_tokens = instance.output_sent.tokens
            instance.decode_times = [0]

    def check_dep_coverage(self, partition='train', debug_idxs=None,
            skip_idxs=(), var_conf=None):
        """Record the fraction of potential arcs that are present in gold
        trees.
        """
        var_flags = variables.TransductionVariables.parse_var_conf(var_conf)
        tgt_instances = self.get_instances(partition=partition,
                                           debug_idxs=debug_idxs,
                                           skip_idxs=skip_idxs)
        prev_average_overlap = None
        print "ancestor_limit\tavg_overlap_rate\tavg_reachability"
        for ancestor_limit in range(30):
            ancestor_limit = None if ancestor_limit == 0 \
                                  else ancestor_limit
            instance_overlaps = []
            instance_reachability = []

            for instance in tgt_instances:
                # TODO: merge with instance.get_overlap()
                sent_dep_tuples = instance.get_constrained_dep_tuples(
                                instance.sentences[0],
                                original_tree=var_flags['orig_deps'],
                                ancestor_dags=var_flags['anc_deps'],
                                pos_matching=var_flags['pos_deps'],
                                noninverted_deps=var_flags['noninv_deps'],
                                fixed_root=var_flags['fixed_root'],
                                verb_root=var_flags['verb_root'],
                                ancestor_limit=ancestor_limit)
                gold_overlaps = []
                for gold_sent in instance.gold_sentences:
                    gold_dep_tuples = instance.get_dep_tuples(
                                gold_sent,
                                parse_type='dparse')
                    gold_dep_tuple_set = set(gold_dep_tuples)
                    overlap = gold_dep_tuple_set.intersection(
                                sent_dep_tuples)
                    gold_overlaps.append(len(overlap) / \
                                         len(gold_dep_tuple_set))

#                    if len(overlap) < len(gold_dep_tuple_set):
#                        print instance.get_display_string()
#                        print gold_dep_tuple_set - overlap

                instance_overlaps.append(avg(gold_overlaps))
                instance_reachability.append(int(min(gold_overlaps) == 1))

            average_overlap = avg(instance_overlaps)
            average_reachability = avg(instance_reachability)
            if average_overlap == prev_average_overlap:
                continue
            prev_average_overlap = average_overlap

            print ancestor_limit, '\t\t', average_overlap,
            print '\t\t', average_reachability
            print '\t\t\t', sum(instance_overlaps),
            print '\t\t', sum(instance_reachability)
            print '\t\t\t', len(instance_overlaps),
            print '\t\t', len(instance_reachability)
            if not var_flags['anc_deps'] or \
                    (ancestor_limit > 0 and average_overlap == 1.0):
                break

    def check_frame_coverage(self, partition='train', debug_idxs=None,
            skip_idxs=()):
        """Record the fraction of potential frames and FEs that are present
        in gold compressions.
        """
        tgt_instances = self.get_instances(partition=partition,
                                           debug_idxs=debug_idxs,
                                           skip_idxs=skip_idxs)

        print "items\tavg_overlap_rate\tavg_reachability\tnum_frameless"
        for item in ['frame', 'fe']:
            instance_overlaps = []
            instance_reachability = []
            num_frameless = 0

            for instance in tgt_instances:
                sent_frame_tuples = getattr(instance,
                        'get_' + item + '_tuples')(instance.sentences[0])
                gold_overlaps = []
                for gold_sent in instance.gold_sentences:
                    gold_frame_tuples = getattr(instance,
                            'get_' + item + '_tuples')(gold_sent)
                    if len(gold_frame_tuples) == 0:
                        gold_overlaps.append(1)     # always reachable
                        num_frameless += 1
                        break
                    gold_frame_tuple_set = set(gold_frame_tuples)
                    overlap = gold_frame_tuple_set.intersection(
                                sent_frame_tuples)
                    gold_overlaps.append(len(overlap) / \
                                            len(gold_frame_tuple_set))

                instance_overlaps.append(avg(gold_overlaps))
                instance_reachability.append(int(min(gold_overlaps) == 1))

            print item, '\t', avg(instance_overlaps),
            print '\t\t', avg(instance_reachability),
            print '\t\t', num_frameless

        # We also need to check which frames are present and how many FEs
        # they have
        num_fes = defaultdict(int)
        in_tgts = 0
        for instance in tgt_instances:
            for sentence in instance.gold_sentences: # + instance.input_sents:
                for frame in sentence.frames.nodes:
                    key = sum(int(hasattr(edge, 'fe'))
                              for edge in frame.outgoing_edges.itervalues())
                    num_fes[key] += 1
                    in_tgts += sum(int(hasattr(edge, 'target') and
                                       hasattr(edge, 'fe'))
                              for edge in frame.outgoing_edges.itervalues())

        print "Histogram of FEs per frame:", dict(num_fes)
        print "Number of FEs which are also targets:", in_tgts

    def test_tightness(self, learner, partition='train', debug_idxs=None,
            skip_idxs=(), decoder='ilp', streaming=False,
            overwritten_params=(),**kwargs):
        """Note the proportion of integral solutions to LPs.
        """
        eval_instances = self.decode_instances(learner,
                                               partition=partition,
                                               debug_idxs=debug_idxs,
                                               skip_idxs=skip_idxs,
                                               decoder='ilp',
                                               relax=True,
                                               streaming=False,  # keep the LP
                                               overwritten_params=\
                                                       overwritten_params,
                                               **kwargs)

        print "idx\tsize\twords\toptwrds\tequiv?\tdeps\toptdeps"
        num_failed, num_tight, num_loose = 0, 0, 0
        token_tightness, dep_tightness = [], []
        for i, instance in enumerate(eval_instances):
            if not instance.decoder.has_solution():
                num_failed += 1
            elif instance.decoder.has_integral_solution(ndigits=3):
                num_tight += 1
                token_tightness.append(1)
                dep_tightness.append(1)
            else:
                num_loose += 1

                # Print some additional statistics for the loose ones
                relaxed = instance.decoder.get_integrality()
                instance.decoder.solve(relax=False)
                optimal = instance.decoder.get_integrality()

                # Ensure that the optimal result is integral for sanity
                for var_type in optimal:
                    assert len(optimal[var_type][0]) == 0

                # Check whether the relaxed word solution is the same as the
                # optimal solution, even if it's non-integral
                is_equiv = sorted([var_tuple[0].idx() for var_tuple in
                                  relaxed['word'][0] + relaxed['word'][1]]) \
                    == sorted([var_tuple[0].idx() for var_tuple in
                              optimal['word'][1]])

                print "%d:\t%d\t%d/%d\t%d\t%s\t%d/%d\t%d" % \
                        (i,
                         sum(sent.length for sent in instance.input_sents),
                         len(relaxed['word'][0]),
                         len(relaxed['word'][0]) + len(relaxed['word'][1]),
                         len(optimal['word'][1]),
                         '' if is_equiv else '!',
                         len(relaxed['dep'][0]),
                         len(relaxed['dep'][0]) + len(relaxed['dep'][1]),
                         len(optimal['dep'][1]),
                         )

                token_tightness.append(len(relaxed['word'][1]) / \
                        (len(relaxed['word'][0]) + len(relaxed['word'][1])))
                dep_tightness.append(len(relaxed['dep'][1]) / \
                        (len(relaxed['dep'][0]) + len(relaxed['dep'][1])))

                # If restricted to a few instances, print the details
                if len(eval_instances) < 3:
                    for feat_cat in ('word', 'dep'):
                        for integrality in (0,1):
                            for rel_tuple in relaxed[feat_cat][integrality]:
                                var, relaxed_value = rel_tuple
                                optimal_value = None
                                for opt_tuple in optimal[feat_cat][1]:
                                    if opt_tuple[0].idx() == var.idx():
                                        optimal_value = opt_tuple[1]
                                        break
                                print "%s\t%.3f\t%s" % \
                                        (var.readable_grounding(),
                                         relaxed_value, optimal_value)

        print "%d/%d (%.1f%%) integral solutions%s" % \
                (num_tight,
                 num_tight + num_loose,
                 (num_tight * 100) / float(num_tight + num_loose),
                 "; %d failed" % (num_failed,) if num_failed > 0 else "")
        print "token integrality rate: %.1f%%" % (avg(token_tightness) * 100,)
        print "dep integrality rate: %.1f%%" % (avg(dep_tightness) * 100,)

    def test_optimality(self, learner, partition='train', debug_idxs=None,
            skip_idxs=(), decoder='lp+mst', streaming=False,
            overwritten_params=(),**kwargs):
        """Note the proportion of optimal solutions when approximating.
        """
        eval_instances = self.decode_instances(learner,
                                               partition=partition,
                                               debug_idxs=debug_idxs,
                                               skip_idxs=skip_idxs,
                                               decoder=decoder,
                                               streaming=False,  # keep the LP
                                               overwritten_params=\
                                                       overwritten_params,
                                               **kwargs)

        approx_token_solns, approx_dep_solns = [], []
        for instance in eval_instances:
            if instance.decoder.has_solution():
                approx_token_solns.append([tuple(idx)
                                        for idx in instance.output_idxs])
                approx_dep_solns.append(instance.get_dep_tuples(
                                        instance.output_sent,
                                        parse_type='outtree'))
            else:
                approx_token_solns.append([])
                approx_dep_solns.append([])
            del instance.decoder

        eval_instances = self.decode_instances(learner,
                                               partition=partition,
                                               debug_idxs=debug_idxs,
                                               skip_idxs=skip_idxs,
                                               decoder='ilp',
                                               streaming=False,  # keep the LP
                                               overwritten_params=\
                                                       overwritten_params,
                                               **kwargs)

        exact_token_solns, exact_dep_solns = [], []
        for instance in eval_instances:
            if instance.decoder.has_solution():
                exact_token_solns.append([tuple(idx)
                                        for idx in instance.output_idxs])
                exact_dep_solns.append(instance.get_dep_tuples(
                                        instance.output_sent,
                                        parse_type='outtree'))
            else:
                exact_token_solns.append([])
                exact_dep_solns.append([])

        token_optimality, dep_optimality = [], []
        num_correct_tokens, num_total_tokens = [], []
        num_correct_deps, num_total_deps = [], []
        num_failed_approx, num_failed_exact, num_succeeded = 0, 0, 0
        for approx_tokens, approx_deps, exact_tokens, exact_deps in zip(
                approx_token_solns, approx_dep_solns,
                exact_token_solns, exact_dep_solns):
            if len(approx_tokens) == 0:
                num_failed_approx += 1
            if len(exact_tokens) == 0:
                num_failed_exact += 1
            if len(approx_tokens) == 0 or len(exact_tokens) == 0:
                continue
            else:
                num_succeeded += 1

            assert len(approx_tokens) == len(exact_tokens)

            token_overlap = set(approx_tokens).intersection(exact_tokens)
            token_optimality.append(
                    int(len(token_overlap) == len(exact_tokens)))
            num_correct_tokens.append(len(token_overlap))
            num_total_tokens.append(len(exact_tokens))

            dep_overlap = set(approx_deps).intersection(exact_deps)
            dep_optimality.append(int(len(dep_overlap) == len(exact_deps)))
            num_correct_deps.append(len(dep_overlap))
            num_total_deps.append(len(exact_deps))

        print "%d/%d (%.1f%%) optimal token solutions%s" % \
                (sum(token_optimality),
                 num_succeeded,
                 avg(token_optimality) * 100,
                 "; %d approx failed, %d exact failed" % \
                         (num_failed_approx, num_failed_exact)
                         if num_succeeded < len(eval_instances) else "")
        print "token optimality rate: %.1f%% over %d instances, " \
                                     "%.1f%% over %d tokens" % \
                (avg(correct/total * 100
                    for correct, total in zip(num_correct_tokens,
                                              num_total_tokens)),
                 num_succeeded,
                 sum(num_correct_tokens)/sum(num_total_tokens) * 100,
                 sum(num_total_tokens))
        print
        print "%d/%d (%.1f%%) optimal dep solutions%s" % \
                (sum(dep_optimality),
                 num_succeeded,
                 avg(dep_optimality) * 100,
                 "; %d approx failed, %d exact failed" % \
                         (num_failed_approx, num_failed_exact)
                         if num_succeeded < len(eval_instances) else "")
        print "dep optimality rate: %.1f%% over %d instances, " \
                                     "%.1f%% over %d deps" % \
                (avg(correct/total * 100
                    for correct, total in zip(num_correct_deps,
                                              num_total_deps)),
                 num_succeeded,
                 sum(num_correct_deps)/sum(num_total_deps) * 100,
                 sum(num_total_deps))

    @staticmethod
    def dump_parse(sent, parse_type='dparse', crossing_edges=None,
            idx_mapper=None):
        """Print a sentence and its parse for analysis.
        """
        marked_edges = set()
        if crossing_edges is not None:
            for edge1, edge2 in crossing_edges:
                marked_edges.add(edge1)
                marked_edges.add(edge2)

        print ' '.join(sent.tokens)
        for edge in getattr(sent, parse_type).edges:
            s, t = edge.src_idx, edge.tgt_idx
            print "%d,%d %s -> %s\t%s" % \
                    (s if idx_mapper is None else idx_mapper[s][1],
                     t if idx_mapper is None else idx_mapper[t][1],
                     sent.tokens[s], sent.tokens[t],
                            '**' if edge in marked_edges else '')

    def dump_scores(self, learner, partition='train', debug_idxs=None,
            skip_idxs=(), **kwargs):
        """Dump the edge scores from a particular set of instances for
        external comparison. Only works with compression for now.
        """
        eval_instances = self.decode_instances(learner,
                                               partition=partition,
                                               debug_idxs=debug_idxs,
                                               skip_idxs=skip_idxs,
                                               decoder='ilp',
                                               relax=True,
                                               streaming=False,  # keep the LP
                                               **kwargs)
        num_instances = len(eval_instances)

        with timer.AvgTimer(num_instances):
            for instance in eval_instances:
                print instance.get_display_string()
                print
                print "## TOKENS"
                num_tokens = 0
                for token_var in instance.decoder.lp.retrieve_all_variables(
                        'WORD'):
                    w = token_var.retrieve_grounding('w')
                    print w, '\t', token_var.coeff
                    num_tokens += 1
                print
                print "## BIGRAMS"
                for bigram_var in instance.decoder.lp.retrieve_all_variables(
                        'NGRAM'):
                    w0, w1 = bigram_var.retrieve_grounding('w0', 'w1')
                    w0 = w0 if isinstance(w0, int) else -1
                    w1 = w1 if isinstance(w1, int) else num_tokens
                    print str(w0) + ',' + str(w1), '\t', bigram_var.coeff
                print
                print "## DEPS"
                for dep_var in instance.decoder.lp.retrieve_all_variables(
                        'DEP'):
                    w0, w1 = dep_var.retrieve_grounding('w0', 'w1')
                    w0 = w0 if isinstance(w0, int) else -1
                    print str(w0) + ',' + str(w1), '\t', dep_var.coeff
                print
                print

    def gold_hists(self, partition='train', **kwargs):
        """Produce histograms of statistics of the length of gold outputs
        with respect to the input sentences.
        """
        in_lens, min_lens, med_lens, max_lens, avg_lens = [], [], [], [], []
        min_rates, med_rates, max_rates, avg_rates = [], [], [], []

        min_hist = defaultdict(int)
        med_hist = defaultdict(int)
        max_hist = defaultdict(int)
        min_to_med_hist = defaultdict(int)
        max_to_med_hist = defaultdict(int)

        for instance in self.get_instances(partition=partition):
            if len(instance.input_sents) > 1:
                print "ERROR: not set up for multiple input sentences"
                break

            input_len = instance.sum_len
            in_lens.append(input_len)

            gold_lens = [len(gold_sent.tokens)
                         for gold_sent in instance.gold_sentences]
            avg_len = np.average(gold_lens)
            avg_lens.append(avg_len)
            avg_rates.append(avg_len / input_len)

            med_len = np.median(gold_lens)
            med_lens.append(med_len)
            med_rates.append(med_len / input_len)
            med_hist[input_len - med_len] += 1

            if len(gold_lens) == 1:
                continue

            min_len = min(gold_lens)
            min_lens.append(min_len)
            min_rates.append(min_len / input_len)
            min_hist[input_len - min_len] += 1
            min_to_med_hist[med_len - min_len] += 1

            max_len = max(gold_lens)
            max_lens.append(max_len)
            max_rates.append(max_len / input_len)
            max_hist[input_len - max_len] += 1
            max_to_med_hist[max_len - med_len] += 1

        # Print histograms
        for name in ('min', 'med', 'max', 'min_to_med', 'max_to_med'):
            hist = locals()[name + '_hist']
            if len(hist) == 0:
                continue

            print "Histogram of token differences:", name
            total, num_instances = 0, 0
            for len_diff in sorted(hist.iterkeys()):
                freq = hist[len_diff]
                print "(" + str(len_diff) + ", " + str(freq) + ")"
                total += freq * len_diff
                num_instances += freq
            print "Average:", total / num_instances
            print

        # Print correlations
        import scipy.stats as stats
        def print_row(name, lens, rates):
            r, p = stats.pearsonr(lens, rates)
            print "%s\t%-.4f\t%-.4f\t%-.4f\t%-.4f" % (name, np.average(lens),
                                                      np.average(rates), r, p)

        print "CORRELATIONS: sentence length vs compression rate"
        print "\tlength\trate\tr\tp"
        print_row("Input", in_lens, avg_rates)
        print_row("Output", avg_lens, avg_rates)
        if len(max_lens) > 1:
            # Multiple outputs
            print_row("Min", min_lens, min_rates)
            print_row("Median", med_lens, med_rates)
            print_row("Max", max_lens, max_rates)

    def check_agreement(self, features, **kwargs):
        """Check agreement for the broadcastnews corpus.
        """
        tgt_instances = self.get_instances(**kwargs)
        num_instances = len(tgt_instances)

        tags = {'*': [0,0,0,0,0], 'root':[0,0,0,0,0]}
        for instance in tgt_instances:
            assert len(instance.gold_sentences) == 3

            instance.initialize(features, **kwargs)

            occurrences = [0] * len(instance.input_sents[0].tokens)

            for gs in range(3):
                for gw in range(len(instance.gold_maps[gs])):
                    assert len(instance.gold_maps[gs][gw]) == 1
                    w = instance.gold_maps[gs][gw][0][0]
                    occurrences[w] += 1

            for w, occurrence in enumerate(occurrences):
                tag = instance.input_sents[0].pos_tags[w]
                if tag[:2] in ('NN', 'VB', 'JJ', 'RB'):
                    tag = tag[:2]
                if tag not in tags:
                    tags[tag] = [0,0,0,0,0]
                if occurrence > 3:
                    print "WARNING:", tag, occurrence
                    occurrence = 3
                tags[tag][occurrence] += 1
                tags[tag][4] += 1
                tags['*'][occurrence] += 1
                tags['*'][4] += 1

                if tag.startswith('VB') and w in \
                        instance.input_sents[0].dparse.root_idxs:
                    tags['root'][occurrence] += 1
                    tags['root'][4] += 1

#            root_counts = defaultdict(int)
#            for gs in range(3):
#                gold_sent = instance.gold_sentences[gs]
#                for root_idx in gold_sent.dparse.root_idxs:
#                    root_word = gold_sent.tokens[root_idx]
#                    root_counts[root_word] += 1
#
#            sorted_roots = sorted(root_counts.iteritems(), reverse=True,
#                                  key=lambda x: x[1])
#            tags['root'][sorted_roots[0][1]] += 1
#            tags['root'][4] += 1

        for tag in tags:
            print tag, '\t', '\t'.join("%.2f" % (t/tags[tag][4]*100)
                                       for t in tags[tag])
