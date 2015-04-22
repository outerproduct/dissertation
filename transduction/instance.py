#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from interfaces import srilm
import itertools
from lexical import mapping, support
import numpy as np
import psutil
import text
import time
from transduction.decoding import dp, ilp
from transduction.model import metaidx


class TransductionInstance(text.Instance):
    """A class representing a text transduction problem.
    """
    def __init__(self, sentences, idx=None):
        """Store multiple input MultiSentences and maintain a collection
        of individual sentences as well.
        """
        text.Instance.__init__(self, sentences)
        self.idx = idx

        # Create an index for all input sentences
        # TODO: is this a good idea? Can we instead use the MultiSentences?
        self.input_sents = []
        self.sent_lens = []
        for multisentence in self.sentences:
            if hasattr(multisentence, 'sentences'):
                # This is a MultiSentence; return its individual sentences
                self.input_sents.extend(multisentence.sentences)
                self.sent_lens.extend(len(sent.tokens)
                        for sent in multisentence.sentences)
            else:
                # This is actually a single sentence
                self.input_sents.append(multisentence)
                self.sent_lens.append(len(multisentence.tokens))

        # Note some additional length statistics; these can be used for
        # normalizing features.
        self.num_sents = len(self.input_sents)
        self.sum_len = sum(self.sent_lens)
        self.avg_len = self.sum_len / len(self.sent_lens)

        # For normalizing features, we keep the reciprocals of instance size
        # handy
        self.num_sents_reciprocal = 1 / self.num_sents
        self.sum_len_reciprocal = 1 / self.sum_len
        self.avg_len_reciprocal = 1 / self.avg_len

        # Note which sentences cover the original input (since we will likely
        # add new sentences for expansions)
        self.original_sent_idxs = range(len(self.input_sents))

    def initialize(self, feats, var_conf=(), constraint_conf=(), ngram_order=3,
            max_flow=100, decoder='ilp', **kwargs):
        """Initialize LP and get scores for all variables.
        """
        if hasattr(self, 'decoder'):
            return

        # Expand chunk annotations for each sentence
        self.input_chunks = self.expand_chunks(self.input_sents)

        # Map each input token to identical tokens in the input sentences
        self.input_maps = mapping.TokenMapping.map_all_sents(
                            self.input_sents,
                            self.input_sents,
                            monotonic=len(self.input_sents) == 1,
                            syntax=True,
                            context=len(self.input_sents) == 1,
                            display=len(self.input_sents) == 1,
                            )

        # In a fusion context, identify which content words, bigrams and
        # dependencies are repeated across sentences and note the number
        # of occurrences.
        # TODO: change to sentences instead of input sents?
        if len(self.input_sents) > 1:
            # Record the per-token support clusters here for redundancy
            # constraints.
            self.support_clusters = support.SupportClusters.cluster_words(
                                        self.input_sents)
            support.SupportClusters.record_support(self.input_sents,
                                        support_clusters=self.support_clusters)

            # Bigram and dependency support are identified and then expanded
            # to cross-sentence support using the input maps
            self.bigram_support = support.SupportClusters.get_expanded_support(
                            self.input_sents, self.input_maps, syntactic=False)
            self.dep_support = support.SupportClusters.get_expanded_support(
                            self.input_sents, self.input_maps, syntactic=True)

        # Build and store the decoder. Options for the --decoder flag
        # 'ilp': exact ILP; works with most --var_flags except 'projective'
        # 'dp': exact DP for McDonald-style bigrams only
        # ... (other options removed for code release)
        if decoder == 'ilp':
            assert 'dep2' not in feats.categories
            self.decoder = ilp.TransductionILP(
                    self, feats, var_conf, constraint_conf,
                    ngram_order=ngram_order, max_flow=max_flow)

        elif decoder == 'dp':
            assert sorted(feats.categories) == ['ngram', 'word']
            assert ngram_order == 2
            self.decoder = dp.BigramDP(
                    self, feats, var_conf, constraint_conf,
                    ngram_order=ngram_order, psi=1)
        else:
            print "ERROR: unusable decoder", decoder

        self.decode_times = []
        self.solution_times = []
        self.failed_count = 0

    def decode(self, weights, feats, relax=False, display_output=False,
            **kwargs):
        """Decode a transduction for this instance.
        """
        if not hasattr(self, 'decoder'):
            # Looks like we've given up on this one
            print "wut?"
            return

        # Update weights and solve the LP. Don't save the internal LP
        # if we don't have much memory left. TODO: handle large LPs gracefully
        self.decoder.update(weights, first_call=(len(self.decode_times)==0))
        start_moment = time.time()
        self.decoder.solve(save=psutil.virtual_memory()[2] < 90,
                      relax=relax,
                      **kwargs)
        self.decode_times.append(time.time() - start_moment)

        if self.decoder.has_solution():
            if not relax:
                # Recover the current sentence and optionally display it.
                # Note that output tokens are stored separately for evaluation.
                # TODO: added timing here as well
                start_moment2 = time.time()
                self.output_idxs = self.decoder.get_solution(self)
                self.solution_times.append(time.time() - start_moment2)

                self.output_tokens = [self.input_sents[s].tokens[w]
                                      for s, w in self.output_idxs]
                if len(self.output_tokens) > 0:
                    self.output_sent = text.Sentence(self.output_tokens)
                    if 'dep' in feats.categories:
                        # Record inferred parse
                        self.output_sent.outtree = \
                                self.decoder.get_tree_solution(
                                        self, self.output_idxs)
                    if 'arity' in feats.categories:
                        self.decoder.verify_arity(self, self.output_idxs,
                                active_tree=self.output_sent.outtree)
                    if 'range' in feats.categories:
                        self.decoder.verify_range(self, self.output_idxs,
                                active_tree=self.output_sent.outtree)
                    if 'frame' in feats.categories or 'fe' in feats.categories:
                        self.output_sent.outframes = \
                                self.decoder.get_frame_solution(
                                        self, self.output_idxs)

                    if display_output:
                        print self.get_display_string()

                        print self.output_sent.outtree.to_text(
                                ext_nodes=self.output_tokens,
                                attribute='score')
                        print

            # HACK: Before returning the final feature vector, we should
            # nullify the fixed feature values so that scores will be
            # displayed properly by the learner. For this, we replace
            # them with the corresponding gold feature values, if available.
            feat_vector = self.decoder.get_solution_feats()
            return feats.sanitize_feat_vector(feat_vector, self)
        else:
            print "Failed on instance", self.idx
            self.failed_count += 1
            for s, sent in enumerate(self.input_sents):
                print str(s) + ':', ' '.join(sent.tokens)
                for edge in sent.dparse.edges:
                    print sent.tokens[edge.src_idx],
                    print '--[' + edge.label + ']->',
                    print sent.tokens[edge.tgt_idx]

            # If this rarely succeeds, don't bother with this instance in
            # the future
            if self.failed_count >= 2:
                delattr(self, 'decoder')
            return None

    def get_display_string(self):
        """Display the instance along with current output and gold target
        if available.
        """
        display_str = "\nInstance %d" % (self.idx,)
        display_str += "  [%.3gs]\n" % (self.decode_times[-1],) \
                if hasattr(self, 'decode_times') else "\n"
        for s, sent in enumerate(self.input_sents):
            display_str += "INPUT_" + str(s) + ": " + \
                    ' '.join(sent.tokens) + "\n"
        if hasattr(self, 'gold_sentences'):
            for gs, gold_sent in enumerate(self.gold_sentences):
                display_str += "GOLD_" + str(gs) + ": " + \
                        ' '.join(gold_sent.tokens) + "\n"
        if hasattr(self, 'output_sent'):
            display_str += "OUTPUT: " + \
                    ' '.join(self.output_sent.tokens) + "\n"
        return display_str

    def cleanup(self):
        """Remove resource-heavy members that aren't needed in the future.
        """
        try:
            delattr(self, 'decoder')
        except AttributeError:
            print "WARNING: no decoder to clean for instance", self.idx

    def expand_chunks(self, sentences):
        """Expand the chunks of the given sentences, i.e., convert them
        to a per-token annotation for easy reference by feature functions.
        """
        per_sentence_expansion = []
        for s, sent in enumerate(sentences):
            # Use a tuple by default so that chunk[0] gives either the label
            # or None
            expanded_chunks = [(None,) for w in sent.tokens]
            for span, chunk_label in sent.chunks.iteritems():
                for w in range(span[0], span[1]+1):
                    expanded_chunks[w] = (chunk_label, s, span[0], span[1])
            per_sentence_expansion.append(expanded_chunks)
        return per_sentence_expansion

    def get_compression_rate(self):
        """Return the ratio of output tokens to input tokens.
        """
        if self.sum_len == 0:
            print "WARNING: empty input in instance", self.idx
            for s, sent in enumerate(self.input_sents):
                print sent
            return 0

        if not hasattr(self, 'output_sent'):
            print "WARNING: no result for instance", self.idx
            return 0

        return len(self.output_sent.tokens) / self.sum_len


class GoldTransductionInstance(TransductionInstance):
    """A transduction instance featuring gold-standard sentence transductions.
    """
    def __init__(self, sentences, gold_sentences, idx=None,
            label_sentences=None):
        """Create an instance with multiple input sentences and output
        sentences.
        """
        TransductionInstance.__init__(self, sentences, idx=idx)

        # Add each gold sentence as a MultiSentence (for annotation purposes)
        self.gold_sentences = [text.MultiSentence(
                                [text.Sentence(gold_sentence)])
                                for gold_sentence in gold_sentences]

        self.sum_gold_len = sum(len(gold_sent.tokens)
                                for gold_sent in self.gold_sentences)
        self.avg_gold_len = self.sum_gold_len / len(self.gold_sentences)

        # If provided, add additional sentences as well.
        if label_sentences is not None:
            self.label_sentences = [text.MultiSentence(
                                    [text.Sentence(label_sentence)])
                                    for label_sentence in label_sentences]

    def initialize(self, feats, **kwargs):
        """Initialize gold feature vectors.
        """
        TransductionInstance.initialize(self, feats, **kwargs)

        # Map gold tokens to input sentences
        self.gold_maps = mapping.TokenMapping.map_all_sents(
                            self.gold_sentences,
                            self.input_sents,
                            monotonic=len(self.input_sents) == 1,
                            syntax=True,
                            context=len(self.input_sents) == 1,
                            display=len(self.input_sents) == 1,
                            )

        # Map gold frames to input sentences
        if 'fe' in feats.categories:
            self.gold_frame_maps = mapping.FrameMapping.map_all_sents(
                                self.gold_sentences,
                                self.input_sents,
                                disambiguate=True)

        self.gold_feat_vectors = [self.score_gold_features(gs, gold_sentence,
                                                           feats, **kwargs)
                for gs, gold_sentence in enumerate(self.gold_sentences)]

        # The final gold feature vector is an average of feature vectors
        # over each gold sentence
        if len(self.gold_sentences) == 1:
            self.gold_feat_vector = self.gold_feat_vectors[0]
        else:
            self.gold_feat_vector = feats.sum_feat_values(
                    self.gold_feat_vectors, average=True)

    def score_gold_features(self, gs, gold_sentence, feats, ngram_order=3,
            directional=False, **kwargs):
        """Get feature values for the gold sentence.
        """
        word_idxs = [(gs, w) for w in range(len(gold_sentence.tokens))]

        # Add indices for individual words
        gold_idxs = {}
        gold_idxs['word'] = [[word_idx] for word_idx in word_idxs]

        # Add indices for arity variables, if present
        if 'arity' in feats.categories:
            gold_idxs['arity'] = gold_idxs['word']

        # Add indices for all ngrams, including start and end symbols
        if 'ngram' in feats.categories:
            gold_idxs['ngram'] = []
            # Copy the words and add start and end nodes
            words = [(None, metaidx.START)] + \
                    word_idxs + \
                    [(None, metaidx.END)]
            gold_idxs['ngram'] = [words[i:i+ngram_order]
                    for i in range(len(words) - ngram_order + 1)]

        # Add indices for all dependencies, including the root symbol
        if 'dep' in feats.categories:
            gold_idxs['dep'] = []

            gold_tree = gold_sentence.dparse
            for root_idx in gold_tree.root_idxs:
                if root_idx >= len(gold_sentence.tokens):
                    # The occasional bad parse may create extra root nodes
                    continue
                gold_idxs['dep'].append([(None, metaidx.ROOT),
                                         (gs, root_idx)])

            for edge in gold_tree.edges:
                gold_idxs['dep'].append([(gs, edge.src_idx),
                                         (gs, edge.tgt_idx)])

        # Add indices for second-order dependencies, including the root symbol
        if 'dep2' in feats.categories:
            gold_idxs['dep2'] = []

            gold_tree = gold_sentence.dparse
            for root_idx in gold_tree.root_idxs:
                if root_idx >= len(gold_sentence.tokens):
                    # The occasional bad parse may create extra root nodes
                    continue
                gold_idxs['dep2'].append([(None, metaidx.ROOT),
                                          (gs, root_idx),
                                          (None, metaidx.ROOT),
                                          ])

            for edge in gold_tree.edges:
                gold_idxs['dep2'].append([(gs, edge.src_idx),
                                          (gs, edge.tgt_idx),
                                          (gs, gold_tree.get_elder_sibling_idx(
                                              edge.tgt_idx)),
                                          ])

        # Add indices for range variables, if present
        if 'range' in feats.categories:
            gold_idxs['range'] = [word_idxs
                                  for word_idxs in gold_idxs['dep']
                                  if (word_idxs[0][1] != metaidx.ROOT and
                                      word_idxs[1][1] != metaidx.LEAF)]

        # Add indices for frame variables, if present
        if 'frame' in feats.categories:
            gold_frames = gold_sentence.frames
            gold_idxs['frame'] = [[(gs, frame_idx)]
                                  for frame_idx in gold_frames.aux_idxs]

        # Add indices for FE variables excluding the 'targets' of frames
        # which are implicitly addressed in the constraints.
        if 'fe' in feats.categories:
            gold_frames = gold_sentence.frames
            gold_idxs['fe'] = [[(gs, frame_idx), (gs, word_idx)]
                               for frame in gold_frames.get_aux_nodes()
                               for word_idx, edge in
                                    frame.outgoing_edges.iteritems()
                               if hasattr(edge, 'fe')]

        gold_feat_vectors = []
        for feat_cat in feats.categories:
            for word_idxs in gold_idxs[feat_cat]:

                # Check for directional dependencies
                direction = None
                if feat_cat == 'range' or (feat_cat == 'dep' and directional):
                    if  word_idxs[0][1] == metaidx.ROOT:
                        direction = 1  # by convention, right attachment
                    elif word_idxs[1][1] == metaidx.LEAF:
                        direction = -1  # by convention, left attachment
                    else:
                        direction = np.sign(word_idxs[1][1] - word_idxs[0][1])

                # For gold features, we will first retrieve mappings to
                # input words for each substructure and then compute an
                # average feature vector for the substructure
                mapped_idxs_list = self.get_mapped_idxs(feat_cat, word_idxs)

                mapped_feat_vectors = []
                for mapped_idxs in mapped_idxs_list:
                    mapped_feat_vectors.append(
                            feats.get_feat_vector(self, mapped_idxs,
                                feat_cat, ngram_order=ngram_order,
                                direction=direction, **kwargs))

                gold_feat_vector = feats.sum_feat_values(mapped_feat_vectors,
                                                         average=True)

                # Scale the range variable features by the actual distance
                # in the gold sentence
                if feat_cat == 'range':
                    distance = abs(word_idxs[0][1] - word_idxs[1][1])
                    gold_feat_vector = feats.scale_feat_values(
                            gold_feat_vector, distance)

                gold_feat_vectors.append(gold_feat_vector)

        # Return the sum of all substructure-specific feature vectors
        # TODO: what happens if we average?
        return feats.sum_feat_values(gold_feat_vectors, average=False)

    def get_mapped_idxs(self, feat_cat, word_idxs):
        """Retrieve mappings to input words or frames for each
        gold substructure. This is used to compute an average feature
        vector for the substructure.
        """
        if feat_cat in ('frame', 'fe'):
            # NOTE: if the frame is not mapped successfully, an empty list
            # will be returned and the frame will thus not contribute to the
            # gold features.

            # The first idx is a frame in the gold sentence
            gs, gf = word_idxs[0]
            substituted_idxs = [[(s,f) for s, f_list in
                                self.gold_frame_maps[gs][gf].iteritems()
                                for f in f_list]]

            if feat_cat == 'fe':
                # The second idx is an actual token in a gold sentence,
                # likely the same
                gs, gw = word_idxs[1]
                substituted_idxs.append([(s,w) for s, w_list in
                                self.gold_maps[gs][gw].iteritems()
                                for w in w_list])
        else:
            # All idxs are tokens or pseudo-tokens in the gold sentence
            substituted_idxs = [[(s,w) for s, w_list in
                                self.gold_maps[gs][gw].iteritems()
                                for w in w_list]
                            if gs is not None else [(gs, gw)]
                        for gs, gw in word_idxs]

        mapped_idxs_list = list(itertools.product(*substituted_idxs))
        return mapped_idxs_list

    def get_sentences(self):
        """Retrieve all Sentence-like objects for annotation, including the
        target gold sentence.
        """
        if hasattr(self, 'label_sentences'):
            return self.sentences + self.gold_sentences + self.label_sentences
        else:
            return self.sentences + self.gold_sentences

###############################################################################
# Scoring

    def score_ngrams(self, n=2, use_labels=False):
        """Score by ngram precision and recall. The default measure is bigrams.
        """
        if not hasattr(self, 'output_tokens'):
            return 0.0, 0.0, 0.0

        # By supplying tuples, this returns a list of ngram tuples
        # that can be hashed and compared easily
        output_ngrams = srilm.LangModel.get_ngrams(n,
                tuple(self.output_tokens))

        gold_sentences = self.label_sentences if use_labels \
                else self.gold_sentences

        precisions, recalls = [], []
        for gold_sentence in gold_sentences:
            gold_ngrams = srilm.LangModel.get_ngrams(n,
                    tuple(gold_sentence.tokens))
            correct = 0
            gold_ngram_set = set(gold_ngrams)
            for ngram in output_ngrams:
                if ngram in gold_ngram_set:
                    correct += 1
                    # Don't allow ngrams to be counted repeatedly
                    # WHY NOT just use sets? Because recall can be > 100!
                    gold_ngram_set.remove(ngram)

            # Calculate precision and recall
            precision = 0.0
            recall = 0.0
            if len(output_ngrams) > 0:
                precision = correct / len(output_ngrams)
            if len(gold_ngrams) > 0:
                recall = correct / len(gold_ngrams)
            precisions.append(precision)
            recalls.append(recall)

        # Average over the number of gold sentences available
        avg_precision = np.average(precisions)
        avg_recall = np.average(recalls)

        f = 0.0
        if avg_precision + avg_recall > 0:
            f = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

        return avg_precision, avg_recall, f

    def score_content_words(self, use_labels=False, prefixes=('NN','VB')):
        """Score by precision and recall of nouns and verbs.
        """
        if not hasattr(self, 'output_idxs'):
            return 0.0, 0.0, 0.0

        output_content_words = [self.input_sents[s].tokens[w]
                for s, w in self.output_idxs
                if self.input_sents[s].pos_tags[w][:2] in prefixes]

        gold_sentences = self.label_sentences if use_labels \
                else self.gold_sentences

        precisions, recalls = [], []
        for gold_sentence in gold_sentences:
            gold_content_words = [gold_sentence.tokens[i]
                    for i, tag in enumerate(gold_sentence.pos_tags)
                    if tag[:2] in prefixes]

            correct = 0
            gold_set = set(gold_content_words)
            for word in output_content_words:
                if word in gold_set:
                    correct += 1
                    # Don't allow words to be counted repeatedly
                    # WHY NOT just use sets? Because recall can be > 100!
                    gold_set.remove(word)

            # Calculate precision and recall
            precision = 0.0
            recall = 0.0
            if len(output_content_words) > 0:
                precision = correct / len(output_content_words)
            if len(gold_content_words) > 0:
                recall = correct / len(gold_content_words)
            precisions.append(precision)
            recalls.append(recall)

        # Average over the number of gold sentences available
        avg_precision = np.average(precisions)
        avg_recall = np.average(recalls)

        f = 0.0
        if avg_precision + avg_recall > 0:
            f = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

        return avg_precision, avg_recall, f

    def score_dependencies(self, parse_type='dparse', use_labels=False,
            **kwargs):
        """Score by precision and recall of dependency arcs or grammatical
        relations.
        """
        if not self.has_output_parses(parse_type=parse_type):
            return 0.0, 0.0, 0.0

        output_deps = set(self.get_dep_tuples(self.output_sent,
                                              parse_type=parse_type,
                                              **kwargs))
        if len(output_deps) == 0:
            return 0.0, 0.0, 0.0

        gold_sentences = self.label_sentences if use_labels \
                else self.gold_sentences

        precisions, recalls = [], []
        for gold_sent in gold_sentences:
            gold_deps = set(self.get_dep_tuples(gold_sent,
                parse_type='dparse' if parse_type == 'outtree' else parse_type,
                **kwargs))
            if len(gold_deps) == 0:
                continue

            correct = len(output_deps.intersection(gold_deps))
            precision = correct / len(output_deps)
            recall = correct / len(gold_deps)

            precisions.append(precision)
            recalls.append(recall)

#            if parse_type == 'outtree':
#                print "\n\n"
#                print correct, len(output_deps), len(gold_deps)
#                print precision, recall

        # Average over the number of gold sentences available
        avg_precision = np.average(precisions)
        avg_recall = np.average(recalls)

        f = 0.0
        if avg_precision + avg_recall > 0:
            f = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

        return avg_precision, avg_recall, f

    def has_output_parses(self, parse_type='dparse'):
        """Return whether the output sentence has an accompanying parse.
        """
        return hasattr(self, 'output_sent') and \
                hasattr(self.output_sent, parse_type)

    def get_dep_tuples(self, sent, stem=False, parse_type='dparse', **kwargs):
        """Return a list of tuples for each edge in the given sentence, where
        each tuple contains the governor and dependent token, optionally
        stemmed.
        """
        tokens = sent.stems[:] if stem else sent.tokens[:]
        parse = getattr(sent, parse_type)

        deps = [(tokens[edge.src_idx], tokens[edge.tgt_idx])
                for edge in parse.edges]

        # RASP parses and outputs with "--vars cyclic" are dependency graphs
        # and don't have roots
        if hasattr(parse, 'root_idxs'):
            deps.extend(('ROOT', tokens[root_idx])
                    for root_idx in parse.root_idxs)

        return deps

    def get_constrained_dep_tuples(self, sent, stem=False,
            original_tree=False, ancestor_dags=False, pos_matching=False,
            noninverted_deps=False, fixed_root=False, verb_root=False,
            ancestor_limit=None):
        """Return a list of tuples for each edge in the given sentence as
        well as ancestral edges, implicitly constructing a DAG.
        """
        tokens = sent.stems[:] if stem else sent.tokens[:]
        if original_tree:
            ancestor_limit = 1
        elif not ancestor_dags or ancestor_limit == 0:
            ancestor_limit = None

        tuples = []
        for node in sent.dparse.nodes:
            if original_tree or ancestor_dags:
                # Only ancestors can be governor
                gov_idxs = sent.dparse.get_ancestor_idxs(node.idx,
                        limit=ancestor_limit)
            else:
                # All nodes can be governor
                gov_idxs = set(range(len(sent.tokens))) - set([node.idx])
                if noninverted_deps:
                    gov_idxs -= set(sent.dparse.get_descendant_idxs(
                                        node.idx, limit=None))

            for gov_idx in gov_idxs:
                if pos_matching:
                    # Only consider dependency tuples when the POS of the
                    # potential child is among the POS of the existing
                    # children of the parent
                    gov_child_tags = [sent.pos_tags[c][:2]
                            for c in sent.dparse.get_child_idxs(gov_idx)]
                    if sent.pos_tags[node.idx][:2] not in gov_child_tags:
                        continue
                tuples.append((tokens[gov_idx], tokens[node.idx]))

            if ancestor_limit is None or node.depth < ancestor_limit:
                if sent.dparse.is_root(node) or ((not fixed_root) and
                        ((not verb_root) or
                         sent.pos_tags[node.idx].lower().startswith('vb'))):
#                if (not fixed_root or sent.dparse.is_root(node)) and \
#                        (not verb_root or
#                         sent.pos_tags[node.idx].lower().startswith('vb')):
                    tuples.append(('ROOT', tokens[node.idx]))

        return tuples

    def get_overlap(self, tgt_sent, ref_sent=None, parse_type='dparse'):
        """Return the number of arcs from a target sentence that overlap with
        a reference sentence.
        """
        if ref_sent is None:
            if len(self.sentences) != 1:
                print "WARNING: overlap not defined for >1 sentence"
                return 0
            else:
                ref_sent = self.sentences[0]

        ref_dep_tuples = set(self.get_dep_tuples(ref_sent))
        tgt_dep_tuples = set(self.get_dep_tuples(tgt_sent,
                                                 parse_type=parse_type))

        overlap = ref_dep_tuples.intersection(tgt_dep_tuples)
        return len(overlap) / len(ref_dep_tuples)

    def score_frames(self, frames_type='frames', use_labels=False, fes=False,
            **kwargs):
        """Score by precision and recall of frames or frame elements.
        """
        if not hasattr(self, 'output_sent') or \
                not hasattr(self.output_sent, frames_type):
            return 0.0, 0.0, 0.0

        if fes:
            output_tuples = set(self.get_fe_tuples(self.output_sent,
                                                   frames_type=frames_type,
                                                   fe_roles=False, **kwargs))
        else:
            output_tuples = set(self.get_frame_tuples(self.output_sent,
                                                      frames_type=frames_type,
                                                      tgts=True, **kwargs))


        gold_sentences = self.label_sentences if use_labels \
                                              else self.gold_sentences

        precisions, recalls = [], []
        for gold_sent in gold_sentences:
            if fes:
                gold_tuples = set(self.get_fe_tuples(gold_sent,
                    frames_type='frames' if frames_type=='outframes'
                                         else frames_type,
                    fe_roles=False, **kwargs))
            else:
                gold_tuples = set(self.get_frame_tuples(gold_sent,
                    frames_type='frames' if frames_type=='outframes'
                                         else frames_type,
                    tgts=True, **kwargs))

            # #out     0       >0     <- #gold
            #   0    1 / 1    0 / 0
            #  >0    0 / 1    p / r
            if len(gold_tuples) == 0:
                precisions.append(float(len(output_tuples) == 0))
                recalls.append(1.0)
                continue

            if len(output_tuples) == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            correct = len(output_tuples.intersection(gold_tuples))
            precision = correct / len(output_tuples)
            recall = correct / len(gold_tuples)

            precisions.append(precision)
            recalls.append(recall)

        # Average over the gold sentences available
        avg_precision = np.average(precisions)
        avg_recall = np.average(recalls)

        f = 0.0
        if avg_precision + avg_recall > 0:
            f = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

        return avg_precision, avg_recall, f

    def get_frame_tuples(self, sent, stem=False, frames_type='frames',
            tgts=True, **kwargs):
        """Return tuples of frames from the given sentence with optional
        target tokens.
        """
        frames = getattr(sent, frames_type)

        if tgts:
            tokens = sent.stems[:] if stem else sent.tokens[:]
            return [(frame.name, tuple([tokens[w] for w in frame.tgt_idxs]))
                    for frame in frames.get_aux_nodes()]
#            return [(frame.name, tokens[frame.tgt_edge.tgt_idx])
#                    for frame in frames.get_aux_nodes()]
        else:
            return [(frame.name,) for frame in frames.get_aux_nodes()]

    def get_fe_tuples(self, sent, stem=False, frames_type='frames',
            fe_roles=False, **kwargs):
        """Return tuples of frame elements and their lexical units from the
        given sentence with optional FE labels.
        """
        tokens = sent.stems[:] if stem else sent.tokens[:]
        frames = getattr(sent, frames_type)

        if fe_roles:
            # (frame, fe_role, coretype)
            return [(frame.name, edge.fe, tokens[w])
                    for frame in frames.get_aux_nodes()
                    for w, edge in frame.outgoing_edges.iteritems()
                    if hasattr(edge, 'fe')]
        else:
            # (frame, coretype)
            return [(frame.name, tokens[w])
                    for frame in frames.get_aux_nodes()
                    for w, edge in frame.outgoing_edges.iteritems()
                    if hasattr(edge, 'fe')]

    def get_frame_overlap(self, tgt_sent, ref_sent=None, frames_type='frames',
            fes=False, **kwargs):
        """Return the number of frames or frame elements from a target
        sentence that overlap with a reference sentence.
        """
        if ref_sent is None:
            if len(self.sentences) != 1:
                print "WARNING: overlap not defined for >1 sentence"
                return 0
            else:
                ref_sent = self.sentences[0]

        if fes:
            ref_fe_tuples = set(self.get_fe_tuples(ref_sent), **kwargs)
            tgt_fe_tuples = set(self.get_fe_tuples(tgt_sent),
                                                   frames_type=frames_type,
                                                   **kwargs)
            overlap = ref_fe_tuples.intersection(tgt_fe_tuples)
            return len(overlap) / len(ref_fe_tuples)

        else:
            ref_frame_tuples = set(self.get_frame_tuples(ref_sent), **kwargs)
            tgt_frame_tuples = set(self.get_frame_tuples(tgt_sent),
                                                    frames_type=frames_type,
                                                    **kwargs)
            overlap = ref_frame_tuples.intersection(tgt_frame_tuples)
            return len(overlap) / len(ref_frame_tuples)

    def get_compressed_len(self, constraint_conf):
        """Extract the number of output tokens specified by a constraint
        configuration. Used by non-ILP decoders.
        """
        # Find the first compression rate indicator and use that
        for constraint_name in constraint_conf:
            if constraint_name.startswith('cr'):
                rate_percent = int(constraint_name[2:])
                return np.floor(rate_percent * 0.01 * self.sent_lens[0])
            elif constraint_name.endswith('cr'):
                gold_lengths = [len(gold_sent.tokens)
                                for gold_sent in self.gold_sentences]
                if constraint_name == 'goldmedcr':
                    return int(np.median(gold_lengths))
                elif 'goldmincr' in constraint_conf:
                    return min(gold_lengths)
                elif 'goldmaxcr' in constraint_conf:
                    return max(gold_lengths)
                elif 'goldcr' in constraint_conf:
                    # Return a tuple specifying the lower and upper bounds
                    return min(gold_lengths), max(gold_lengths)

        print "WARNING: no usable compression rate supplied",
        print "in constraint configuration", constraint_conf
        return None

    def get_gold_compression_rate(self):
        """Return the averaged ratio of gold tokens to input tokens.
        """
        if self.sum_len == 0:
            print "WARNING: empty input in instance", self.idx
            for s, sent in enumerate(self.input_sents):
                print sent
            return 0

        return self.avg_gold_len / self.sum_len
