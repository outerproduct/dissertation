#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict


class TokenMapping(object):
    """Generate a token mapping between two sets of sentences.
    Can be constrained to a heuristic token alignment.
    """
    @classmethod
    def map_all_sents(cls, src_sents, tgt_sents, **kwargs):
        """Map the tokens in the source sentences to tokens in the target
        sentences.
        """
        # sentence_maps[src_s][src_w][tgt_s] = [tgt_ws]
        sentence_maps = []

        for src_s, src_sent in enumerate(src_sents):
            src_token_map = [{} for src_token in src_sent.tokens]

            for tgt_s, tgt_sent in enumerate(tgt_sents):
                token_map = cls.map_sent_pair(src_sent, tgt_sent, **kwargs)

                for src_w, tgt_w_list in token_map.iteritems():
                    src_token_map[src_w][tgt_s] = tgt_w_list

            sentence_maps.append(src_token_map)

        return sentence_maps

    @classmethod
    def map_sent_pair(cls, src_sent, tgt_sent, monotonic=False, syntax=False,
            context=False, display=True):
        """Map the tokens in the source sentence to tokens in the target
        sentence, optionally disambiguating mappings within a sentence
        using heuristics.
        """
        # token_map[src_w] = [tgt_ws]
        token_map = defaultdict(list)
        ambiguous = []

        for src_w, src_token in enumerate(src_sent.tokens):
            for tgt_w, tgt_token in enumerate(tgt_sent.tokens):
                if src_token == tgt_token:
                    token_map[src_w].append(tgt_w)
            # Note ambiguous mappings
            if len(token_map[src_w]) > 1:
                ambiguous.append(src_w)

        if len(ambiguous) > 0 and monotonic:
#            print "MAPPING AMBIGUITY before monotonicity"
#            cls.display_ambiguity(src_sent, tgt_sent, token_map)
            ambiguous = cls.disambiguate_monotonic(src_sent, tgt_sent,
                                                token_map, ambiguous)

        if len(ambiguous) > 0 and syntax:
#            print "MAPPING AMBIGUITY before syntax"
#            cls.display_ambiguity(src_sent, tgt_sent, token_map)
            ambiguous = cls.disambiguate_syntax(src_sent, tgt_sent,
                                                token_map, ambiguous)

        if len(ambiguous) > 0 and context:
#            print "MAPPING AMBIGUITY before context"
#            cls.display_ambiguity(src_sent, tgt_sent, token_map)
            ambiguous = cls.disambiguate_context(src_sent, tgt_sent,
                                                token_map, ambiguous)
        if len(ambiguous) > 0 and display:
            print "WORD MAPPING AMBIGUITY"
            cls.display_ambiguity(src_sent, tgt_sent, token_map)

        return token_map

    @classmethod
    def display_ambiguity(cls, src_sent, tgt_sent, token_map):
        """Show the ambiguous mappings in a pair of sentences.
        """
        print "SRC: ", ' '.join(src_sent.tokens)
        print "TGT: ", ' '.join(tgt_sent.tokens)

        for src_w, tgt_w_list in token_map.iteritems():
            if len(tgt_w_list) > 1:
                print src_w, "[" + src_sent.tokens[src_w] + "] ->", tgt_w_list
        print

    @classmethod
    def disambiguate_syntax(cls, src_sent, tgt_sent, token_map, ambiguous,
            keep_only_aligned=False):
        """Try to disambiguate the ambiguous mappings using their
        governors in a dependency parse.
        """
        if not hasattr(src_sent, 'dparse') or \
                not hasattr(tgt_sent, 'dparse'):
            # Can't do much here
            print "WARNING: no parses for syntactic disambiguation"
            return ambiguous

        remaining_ambiguous = []
        resolved_tgts = []

        for src_w in ambiguous:
            src_p = src_sent.dparse.get_parent_idx(src_w)
            src_p_to_tgt_list = token_map[src_p] \
                                if src_p is not None else [None]

            tgt_w_list = token_map[src_w]
            tgt_p_list = [tgt_sent.dparse.get_parent_idx(tgt_w)
                          for tgt_w in tgt_w_list]

            # If the parents of a source and target word align, we
            # prefer the mapping
            overlap = set(tgt_p_list).intersection(src_p_to_tgt_list)

            if len(overlap) == 0:
                # No overlap; all still ambiguous
                if not keep_only_aligned:
                    # Drop these cases if we only care about mappings which
                    # are syntactically supported (unused)
                    remaining_ambiguous.append(src_w)
                    continue

            # Otherwise, we drop the mappings not supported by syntax.
            reduced_tgt_w_list = [tgt_w for tgt_w in tgt_w_list
                                  if tgt_sent.dparse.get_parent_idx(tgt_w)
                                  in overlap]
            token_map[src_w] = reduced_tgt_w_list

            if len(reduced_tgt_w_list) > 1:
                # Still too ambiguous
                remaining_ambiguous.append(src_w)
            elif len(reduced_tgt_w_list) == 1:
                resolved_tgts.append(reduced_tgt_w_list[0])

        # Remove resolved target tokens from the mappings of remaining
        # ambiguous alignments
        cls.trim_resolved(token_map, remaining_ambiguous, resolved_tgts)

        return remaining_ambiguous

    @classmethod
    def disambiguate_monotonic(cls, src_sent, tgt_sent, token_map, ambiguous):
        """Try to disambiguate the ambiguous mappings by assuming that
        the mappings must be monotonic, i.e., no word reordering.
        """
        remaining_ambiguous = []
        resolved_tgts = []

        prev_tgt_w = -1
        for src_w in range(len(src_sent.tokens)):
            tgt_w_list = token_map[src_w]
            if len(tgt_w_list) == 1:
                assert tgt_w_list[0] > prev_tgt_w
                prev_tgt_w = tgt_w_list[0]

            elif len(tgt_w_list) > 1:
                assert src_w in ambiguous
                reduced_tgt_w_list = [tgt_w for tgt_w in tgt_w_list
                                        if tgt_w > prev_tgt_w]
                prev_tgt_w = min(reduced_tgt_w_list)

                if len(reduced_tgt_w_list) < len(tgt_w_list):
                    # Reduced ambiguity
                    token_map[src_w] = reduced_tgt_w_list

                if len(reduced_tgt_w_list) > 1:
                    # Still too ambiguous
                    remaining_ambiguous.append(src_w)
                elif len(reduced_tgt_w_list) == 1:
                    resolved_tgts.append(reduced_tgt_w_list[0])

        # Remove resolved target tokens from the mappings of remaining
        # ambiguous alignments
        cls.trim_resolved(token_map, remaining_ambiguous, resolved_tgts)

        return remaining_ambiguous

    @classmethod
    def trim_resolved(cls, token_map, ambiguous, resolved_tgts):
        """Remove the newly-resolved target tokens from ambiguous mappings.
        """
        while len(resolved_tgts) > 0:
            tgt_w = resolved_tgts.pop()
            for src_w in ambiguous:
                if tgt_w in token_map[src_w]:
                    token_map[src_w].remove(tgt_w)

                    if len(token_map[src_w]) == 1:
                        ambiguous.remove(src_w)
                        resolved_tgts.append(token_map[src_w][0])

    @classmethod
    def disambiguate_context(cls, src_sent, tgt_sent, token_map, ambiguous,
            max_gap=15):
        """Try to disambiguate the ambiguous mappings using the
        distance to their nearest unambiguous neighbors.
        """
        remaining_ambiguous = []
        resolved_tgts = []

        for src_w in ambiguous:
            tgt_w_list = token_map[src_w]

            prev_src_mappings = []
            next_src_mappings = []
            gap = 0
            while len(prev_src_mappings) == 0 or \
                    len(next_src_mappings) == 0:
                gap += 1
                prev_src_mappings = cls.get_prev_src_mappings(
                        token_map,
                        src_w,
                        gap=gap)
                next_src_mappings = cls.get_next_src_mappings(
                        token_map,
                        src_w,
                        len(src_sent.tokens),
                        len(tgt_sent.tokens),
                        gap=gap)

            closest_tgt_w = cls.argmin_distance(
                    tgt_w_list,
                    prev_src_mappings + next_src_mappings,
                    len(tgt_sent.tokens))

            while len(closest_tgt_w) != 1 and gap < max_gap:
                # Keep increasing the gap and trying to find a unique
                # solution
                gap += 1
                prev_src_mappings = cls.get_prev_src_mappings(
                        token_map,
                        src_w,
                        gap=gap)
                next_src_mappings = cls.get_next_src_mappings(
                        token_map,
                        src_w,
                        len(src_sent.tokens),
                        len(tgt_sent.tokens),
                        gap=gap)
                closest_tgt_w = cls.argmin_distance(
                        tgt_w_list,
                        prev_src_mappings + next_src_mappings,
                        len(tgt_sent.tokens))

            token_map[src_w] = closest_tgt_w
            if len(closest_tgt_w) > 1:
                remaining_ambiguous.append(src_w)
            elif len(closest_tgt_w) == 1:
                resolved_tgts.append(closest_tgt_w[0])

        # Remove resolved target tokens from the mappings of remaining
        # ambiguous alignments
        cls.trim_resolved(token_map, remaining_ambiguous, resolved_tgts)

#        for src_w in ambiguous:
#            print src_w, '->', token_map[src_w]

        return remaining_ambiguous

    @classmethod
    def get_prev_src_mappings(cls, token_map, src_w, gap=1):
        """Get the mappings of the nearest unambiguously-mapped tokens
        which appear before (and in the neighborhood of) the given source
        token.
        """
        prev_src_mappings = []
        if src_w - gap < 0:
            # Sentence boundary
            prev_src_mappings.append(-1)

        for prev_src_w in range(src_w, max(src_w-gap-1, -1), -1):
            if len(token_map[prev_src_w]) != 1:
                continue
            prev_src_mappings.append(token_map[prev_src_w][0])

        return prev_src_mappings

    @classmethod
    def get_next_src_mappings(cls, token_map, src_w, src_len, tgt_len, gap=1):
        """Get the mapping of the nearest unambiguously-mapped tokens
        which appear after (and in the neighborhood of) the given source
        token.
        """
        next_src_mappings = []
        if src_w + gap >= src_len:
            # Sentence boundary
            next_src_mappings.append(tgt_len)

        for next_src_w in range(src_w, min(src_w+gap+1, src_len)):
            if len(token_map[next_src_w]) != 1:
                continue
            next_src_mappings.append(token_map[next_src_w][0])

        return next_src_mappings

    @classmethod
    def argmin_distance(cls, w_list, anchors, max_dist):
        """Find the point(s) which minimize absolute distance to the
        anchors.
        """
        closest_w = []
        min_dist = max_dist * len(anchors)

        for w in w_list:
            dist = sum(abs(anchor - w) for anchor in anchors)
            if dist < min_dist:
                closest_w = [w]
                min_dist = dist
            elif dist == min_dist:
                closest_w.append(w)

        return closest_w


class FrameMapping(object):
    """Generate a frame mapping between two sets of sentences.
    """
    @classmethod
    def map_all_sents(cls, src_sents, tgt_sents, disambiguate=False):
        """Map the frames in the source sentences to frames in the target
        sentences.
        """
        # sentence_maps[src_s][src_f][tgt_s] = [tgt_fs]
        sentence_maps = []

        for src_s, src_sent in enumerate(src_sents):
            # We must initialize a vector for all nodes for correct indexing
            # even though the non-auxiliary token nodes will remain unmapped.
            src_frame_map = [{} for node in src_sent.frames.nodes]

            for tgt_s, tgt_sent in enumerate(tgt_sents):
                frame_map = cls.map_sent_pair(src_sent, tgt_sent,
                                                disambiguate=disambiguate)

                for src_f, tgt_f_list in frame_map.iteritems():
                    src_frame_map[src_f][tgt_s] = tgt_f_list

            sentence_maps.append(src_frame_map)

        return sentence_maps

    @classmethod
    def map_sent_pair(cls, src_sent, tgt_sent, disambiguate=False,
            drop_ambiguous=True):
        """Map the frames in the source sentence to frames in the target
        sentence.
        """
        # frame_map[src_f] = [tgt_fs]
        frame_map = defaultdict(list)
        ambiguous = []

        for src_f in src_sent.frames.aux_idxs:
            src_frame = src_sent.frames.nodes[src_f]
            for tgt_f in tgt_sent.frames.aux_idxs:
                tgt_frame = tgt_sent.frames.nodes[tgt_f]

                if src_frame.name == tgt_frame.name:
                    frame_map[src_f].append(tgt_f)

            # Note ambiguous mappings
            if len(frame_map[src_f]) > 1:
                ambiguous.append(src_f)

        if disambiguate:
            if len(ambiguous) > 0:
#                print "MAPPING AMBIGUITY before syntax"
#                cls.display_ambiguity(src_sent, tgt_sent, token_map)

                ambiguous = cls.disambiguate_tgt(src_sent, tgt_sent,
                                                 frame_map, ambiguous)

        if drop_ambiguous:
            # Drop mappings that are not supported by aligned targets
            for src_f in ambiguous:
                frame_map[src_f] = []
        elif len(ambiguous) > 0:
            print "FRAME MAPPING AMBIGUITY"
            cls.display_ambiguity(src_sent, tgt_sent, frame_map)

        return frame_map

    @classmethod
    def disambiguate_tgt(cls, src_sent, tgt_sent, frame_map, ambiguous):
        """Try to disambiguate the ambiguous frame mappings using their
        'target' words (i.e., lexicalizations, not words from the target
        sentence) and a disambiguated token mapping between the sentences.
        """
        token_map = TokenMapping.map_sent_pair(src_sent, tgt_sent,
                                               monotonic=True,
                                               syntax=True,
                                               context=True)
        remaining_ambiguous = []
        for src_f in ambiguous:
            src_ws = src_sent.frames.nodes[src_f].tgt_idxs
            src_w_to_tgt_ws = set(token_map[src_w][0] for src_w in src_ws)

            max_overlap = -1
            best_tgt_fs = []
            for tgt_f in frame_map[src_f]:
                tgt_ws = tgt_sent.frames.nodes[tgt_f].tgt_idxs
                overlap = src_w_to_tgt_ws.intersection(tgt_ws)
                if len(overlap) > max_overlap:
                    max_overlap = len(overlap)
                    best_tgt_fs = [tgt_f]
                elif len(overlap) == max_overlap:
                    best_tgt_fs.append(tgt_f)

            frame_map[src_f] = best_tgt_fs

            if len(best_tgt_fs) > 1:
                remaining_ambiguous.append(src_f)

        return remaining_ambiguous

    @classmethod
    def display_ambiguity(cls, src_sent, tgt_sent, frame_map):
        """Show the ambiguous mappings in a pair of sentences.
        """
        print "SRC: ", ' '.join(src_sent.tokens)
        print "TGT: ", ' '.join(tgt_sent.tokens)

        for src_f, tgt_f_list in frame_map.iteritems():
            if len(tgt_f_list) <= 1:
                continue

            src_node = src_sent.frames.nodes[src_f]
            src_lex = ' '.join(src_sent.tokens[w]
                               for w in src_node.tgt_idxs)
            src_str = src_node.name + " [" + src_lex + "]"
            print src_str, "->",

            for tgt_f in tgt_f_list:
                tgt_node = tgt_sent.frames.nodes[tgt_f]
                tgt_lex = ' '.join(tgt_sent.tokens[w]
                                   for w in tgt_node.tgt_idxs)
                tgt_str = "[" + tgt_lex + "]"
                print tgt_str,
            print
        print
