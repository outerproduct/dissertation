#!/usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import instance
from nltk.corpus import wordnet
from operator import attrgetter
import re
#import sys
import string
from text.tokens import splitter
from xml.etree import ElementTree


class Pyramid(object):
    """A class to describe a group of summaries with SCU annotations.
    """
    def __init__(self, filepath, year):
        """Initialize the contents.
        """
        self.filepath = filepath
        self.year = year
        self.name = filepath[filepath.rfind('/') + 1:-4]
        self.docid = self.name[:self.name.rfind('-')] \
                if '-' in self.name else self.name

        self.lines = []
        self.scus = {}
        self.parse_from_xml()

    def parse_from_xml(self):
        """Read in the document contents from an XML file.
        """
        etree = ElementTree.parse(self.filepath)
        pyramid = etree.getroot()

        all_lines = []
        self.char_to_idx = {}

        start = 0
        for line in pyramid.find('text'):
            if line.text is None:
                # Account for the implicit newline between lines
                all_lines.append('')
                start += 1
            else:
                all_lines.append(line.text)

                sublines = splitter.split(line.text)

                sub_start = start
                for s, subline in enumerate(sublines):
                    self.lines.append(subline)

                    # Count the length of the line plus 1 for the newline that
                    # separates full lines or 2 for the spaces that separate
                    # sublines.
                    if s < len(sublines) - 1:
                        end_lb = sub_start + len(subline)
                        end = start + line.text.find(sublines[s+1],
                                                     end_lb - start) - 1
                    else:
                        end = start + len(line.text)

                    for i in range(sub_start, end + 1):
                        if i in self.char_to_idx:
                            print "ERROR: Duplicate line assignment"
                            print line.text
                            print sublines
                        assert i not in self.char_to_idx
                        self.char_to_idx[i] = len(self.lines) - 1

                    # Next subline
                    sub_start = end + 1

                # Next line
                start += len(line.text) + 1

        self.text = '\n'.join(all_lines)
        assert len(self.text) == start - 1

        offsets = set()
        for scu in pyramid.findall('scu'):
            pyr_scu = PyramidSCU(scu, self.lines, self.text, offsets,
                    self.char_to_idx)
            if pyr_scu.uid in self.scus:
                print "WARNING: Ignoring duplicate uid", pyr_scu.uid,
                print "for SCU in", self.filepath
                print '#' + pyr_scu.uid, self.scus[pyr_scu.uid].label
                print '#' + pyr_scu.uid, pyr_scu.label

            self.scus[pyr_scu.uid] = pyr_scu

    def get_fusions(self, **kwargs):
        """Return a list of fusion instances from SCUs.
        """
        fusions = []
        for scu in self.scus.itervalues():
            if scu.is_fusion_candidate(**kwargs):
                fusions.append(instance.FusionInstance(
                    input_sents=['\n'.join(contrib.lines)
                        for contrib in scu.get_filtered_contribs(**kwargs)],
                    output_sent=scu.label,
                    filepaths=(self.filepath,),
                    year=self.year,
                    labels=[contrib.label
                        for contrib in scu.get_filtered_contribs(**kwargs)]
                    ))
        return fusions

    def get_xref_fusions(self, **kwargs):
        """Get fusion instances from the source doc sentences to summary
        sentences.
        """
        fusions = []
        for idx, summ_line in enumerate(self.summ_lines):
            if summ_line.is_fusion_candidate(self.scus, **kwargs):
                fusions.append(instance.FusionInstance(
                    input_sents=summ_line.get_unique_mentions(self.scus),
                    output_sent=summ_line.line,
                    filepaths=summ_line.get_doc_filepaths(self.scus) +
                              [self.filepath],
                    year=self.year))
        return fusions

    def map_source_docs(self, source_docs):
        """Map the annotated SCUs to source documents for this pyramid.
        """
        self.summ_lines = [SummaryLine(idx, line)
                for idx, line in enumerate(self.lines)]

        for source_doc in source_docs:
            for uid, scu in source_doc.scus.iteritems():
                pyr_scu = self.scus[uid]
                if pyr_scu.label != scu.label:
                    print "WARNING: Mismatched SCU labels for #" + uid
                    print "DOC:", source_doc.filepath
                    print '#' + uid, scu.label
                    print "PYR:", self.filepath
                    print '#' + uid, pyr_scu.label
                    print
                pyr_scu.doc_mentions = scu.get_mentions()

                if hasattr(pyr_scu, 'doc_filepaths'):
                    pyr_scu.doc_filepaths.append(source_doc.filepath)
                else:
                    pyr_scu.doc_filepaths = [source_doc.filepath]

                for contributor in pyr_scu.contributors:
                    for line_idx in contributor.line_idxs:
                        self.summ_lines[line_idx].add_scu(uid)


class PyramidSCU(object):
    """A class to describe an SCU within a pyramid.
    """
    prefix = r"(\([A-Z]\) )?"
    suffix = r"(\s*\(\d+\.\d+(?:, ?\d+\.\d+)*\)|\s*\(NONE\))?"
    label_re = re.compile(prefix + r"(.*?)" + suffix + r"\s*$")

    def __init__(self, scu_etree, *args):
        """Initialize the SCU.
        """
        self.uid = scu_etree.get('uid')

        # Postprocess the SCU label to remove format artifacts.
        match = re.match(self.label_re, scu_etree.get('label'))
        if match is None:
            print "ERROR: Strange SCU label"
            print scu_etree.get('label')
        self.prefix, self.label, self.suffix = match.groups()

        self.contributors = [PyramidContributor(contrib_etree, *args)
                for contrib_etree in scu_etree]
        self.weight = len(self.contributors)

    @staticmethod
    def split_to_words(sentence):
        """Split up a sentence into lowercased words ignoring punctuation.
        """
        lowered = sentence.lower().replace('-', ' ').replace('...', ' ')
        if isinstance(lowered, unicode):
            return re.sub(ur"\p{P}+", '', lowered).strip().split()
        else:
            return lowered.translate(string.maketrans('',''),
                    string.punctuation).strip().split()

    def is_fusion_candidate(self, min_inputs=2, min_words=5, max_words=100,
            min_scu_line_overlap=1, min_scu_part_ratio=0.5,
            min_part_line_ratio=0.5, drop_verbless=True, **kwargs):
        """Return whether this SCU has multiple contributors that each
        only account for a single sentence. Also drop SCUs in which
        the label is not fully covered by the sources.
        """
        if len(self.get_filtered_contribs(**kwargs)) < min_inputs:
            return False

        # Drop SCUs that don't contain verbs past the first word. We exclude
        # the first word since many SCUs are sentence fragments.
        scu_words = self.split_to_words(self.label)
        found_verb = False
        for w in range(1, len(scu_words)):
            if wordnet.morphy(scu_words[w], 'v') is not None:
                found_verb = True
                break
        if not found_verb and drop_verbless:
            return False

        # Drop instances in which the SCU label is too small to be a plausible
        # sentence or too big to be parsed
#        if len(scu_words) >=3 and len(scu_words) < min_words:
#            print self.label
        if len(scu_words) < min_words or len(scu_words) > max_words:
            return False

        # Drop instances in which any source sentence is too small to be a
        # plausible sentence or too big to be parsed
        contrib_line_words = [self.split_to_words(' '.join(contrib.lines))
                for contrib in self.get_filtered_contribs(**kwargs)]
        min_line_len = min(len(words) for words in contrib_line_words)
        max_line_len = max(len(words) for words in contrib_line_words)
        if min_line_len < min_words or max_line_len > max_words:
            return False

        # Drop instances in which the SCU label is much smaller than the
        # contributor labels, as this indicates that the SCU is a small
        # fragment and not a valid sentence.
        contrib_label_words = [self.split_to_words(contrib.label)
                for contrib in self.get_filtered_contribs(**kwargs)]
        min_label_len = min(len(words) for words in contrib_label_words)
        if len(scu_words) / min_label_len < min_scu_part_ratio:
            return False

        # Drop instances in which no contributor label is a substantial
        # portion of its corresponding source line since it indicates that
        # this SCU is not a major part of the text.
        found_significant = False
        for label_words, line_words in zip(contrib_label_words,
                contrib_line_words):
            if len(label_words) / len(line_words) >= min_part_line_ratio:
                found_significant = True
                break
        if not found_significant:
            return False

        # Drop instances in which the SCU label is not adequately covered
        # by the contributor source lines, as this will make it harder to
        # recover for fusion.
        contrib_line_words_set = set(word
                for line_words in contrib_line_words for word in line_words)
        overlap = contrib_line_words_set.intersection(scu_words)
        if len(overlap) / len(scu_words) < min_scu_line_overlap:
            return False

        return True

    def get_filtered_contribs(self, single_sent_input=True,
            skip_exact_lines=True, skip_exact_labels=False,
            max_inputs=4, **kwargs):
        """Filter down the contributors into ones which will make good fusions.
        """
        if hasattr(self, 'filtered_contributors'):
            return self.filtered_contributors

        self.filtered_contributors = []
        for contributor in self.contributors:
            if single_sent_input and len(contributor.lines) > 1:
                continue
            if skip_exact_lines and (self.label == contributor.lines[0]
                    or self.label + '.' == contributor.lines[0]):
                continue
            if skip_exact_labels and self.label == contributor.label:
                continue
            self.filtered_contributors.append(contributor)

        if len(self.filtered_contributors) <= max_inputs:
            return self.filtered_contributors

        # Try to reduce inputs to fit max_sents by only keeping the ones most
        # similar to the SCU label. In this case, we use absolute word
        # overlap as a proxy for similarity.
        output_words = set(self.split_to_words(self.label))
        for contrib in self.filtered_contributors:
            line_words = self.split_to_words(' '.join(contrib.lines))
            contrib.scu_overlap = len(output_words.intersection(line_words))

        overlap_sorted = sorted(self.filtered_contributors,
                key=attrgetter('scu_overlap'),
                reverse=True)
        self.filtered_contributors = overlap_sorted[:max_inputs]
        return self.filtered_contributors


class PyramidContributor(object):
    """A class to describe a single contributor to an SCU.
    """
    def __init__(self, contrib_etree, lines, *args):
        """Initialize the contributor.
        """
        self.label = contrib_etree.get('label')

        self.parts = []
        self.line_idxs = set()
        for part_etree in contrib_etree:
            part = PyramidContributorPart(part_etree, lines, *args)
            self.parts.append(part)
            self.line_idxs.update(part.line_idxs)

        self.lines = [lines[idx] for idx in sorted(self.line_idxs)]


class PyramidContributorPart(object):
    """A class to describe an excerpt for an SCU contributor.
    """
    def __init__(self, part_etree, lines, text, offsets, char_to_idx):
        """Initialize the excerpt and adjust offsets to match the text.
        """
        self.start = int(part_etree.get('start'))
        self.end = int(part_etree.get('end'))
        self.label = part_etree.get('label').replace('&quot;', '"')

        self.fix_offsets(offsets, text)
        self.line_idxs = (range(char_to_idx[self.start],
                               char_to_idx[self.end] + 1))

        if len(self.line_idxs) == 1 and \
                self.label not in lines[self.line_idxs[0]]:
            print "Part label: ", self.label
            print self.start, self.end
            print self.line_idxs[0]
            print "Line:", lines[self.line_idxs[0]]

    def fix_offsets(self, offsets, text):
        """Adjust the offsets of the part to match the text.
        """
        if self.start > len(text) or self.start >= self.end:
            self.start = 0
            self.end = self.start + len(self.label)

        retrieved_label = self.collapse_newline(text[self.start:self.end])
        if self.label != retrieved_label:
            actual_start = self.find_label_nearby(text, offsets)
            if actual_start != -1:
                offset = actual_start - self.start
                if offset not in offsets:
                    # We haven't seen this offset before, so throw a warning
#                    sys.stdout.write("\nWARNING: adjusted part offset by " +
#                            str(offset) + " chars\n")
                    offsets.add(offset)

                self.start = actual_start
                self.end += offset
            else:
                print "\nERROR at start offset", self.start
                print "EXP:", self.label
                print "GOT:", retrieved_label

    @staticmethod
    def collapse_newline(text):
        return text.translate({ord('\n'): None}) if isinstance(text, unicode) \
                else text.translate(None, '\n')

    def find_label_nearby(self, text, offsets, verbose=False):
        """Look for the expected label in the string, first moving back by
        some variable amount.
        """
        targets = [self.label]
        if '.' in self.label:
            # Also try variations with sentence-terminating newlines
            targets.append(self.label.replace('.', '.\n'))

        # Otherwise by default, move back by the length of the string.
        # Note that this is a negative shift value.
        backshift = self.start - self.end

        # Move back further if larger offsets have been encountered before
        if len(offsets) > 0:
            backshift = min(backshift, min(offsets))

        # Search for the string from the backshifted start offset
        return max(text.find(target, max(self.start + backshift, 0))
                for target in targets)


class SummaryLine(object):
    """A class to store summary lines which map to source document sentences.
    """
    def __init__(self, idx, line):
        """Initialize the line.
        """
        self.idx = idx
        self.line = line

        self.uids = []

    def add_scu(self, uid):
        """Add an SCU for this line using its UID.
        """
        self.uids.append(uid)

    def is_sourced(self, all_scus, min_src_overlap=1.0, **kwargs):
        """Check whether the line is fully covered by SCUs which have
        mentions in the source documents.
        """
        # We check for coverage by ensuring that the words in this line appear
        # in the part text of the SCUs that it covers.
        part_words = set()
        for uid in self.uids:
            line_scu = all_scus[uid]
            for contributor in line_scu.contributors:
                for part in contributor.parts:
                    if self.idx in part.line_idxs:
                        part_words.update(part.label.split())

        line_words = set(self.line.split())
        overlap = part_words.intersection(line_words)
        return len(overlap) / len(line_words) >= min_src_overlap

    def get_doc_filepaths(self, all_scus):
        """Return a list of filepaths of annotated source documents for this
        summary line.
        """
        return list(set(filepath for uid in self.uids
                for filepath in all_scus[uid].doc_filepaths))

    def get_unique_mentions(self, all_scus):
        """Return a list of unique mentions of SCUs from this summary sentence
        in the source docs.
        """
        return list(set(mention for uid in self.uids
                for mention in all_scus[uid].doc_mentions))

    def is_fusion_candidate(self, all_scus, **kwargs):
        """Returns whether there is more than one source doc sentence that
        contributes to covering a summary sentence.
        """
        return self.is_sourced(all_scus, **kwargs) and \
                len(self.get_unique_mentions(all_scus)) > 1
