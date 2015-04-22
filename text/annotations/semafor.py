#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import os
import re
import subprocess
import tempfile
import text.sentence
import text.structure
from xml.etree import ElementTree


class Semafor(object):
    """Interface to the Semafor frame-semantic parser.
    """
    def __init__(self,
            path='/proj/fluke/users/kapil/tools/semafor/'
                 'semafor-semantic-parser'):
        """Initialize the path to the Semafor parser.
        """
        self.path = path

        # Token mappings for words that are tokenized differently by MST
        self.token_mappings = {'\"': '``',
                               '(': '-LRB-',
                               ')': '-RRB-',
                               'gonna': 'gon na', 'Gonna': 'Gon na',
                               'wanna': 'wan na', 'Wanna': 'Wan na',
                               'gotta': 'got ta', 'Gotta': 'Got ta'}

        # Regular expressions for keeping text consistent
        self.numeric_symbols_re = re.compile(r"\d([:,])\d")

    def run_on_corpus(self, corpus, show_output=False):
        """Write sentences to a temporary file as strings of words, run the
        Stanford parser on the file and retrieve the parsed results,
        then delete the file.
        """
        # Check if the corpus consists of Sentences or MultiSentences, and
        # get a single list of Sentences either way
        sentences = []
        if corpus[0].__class__ == text.sentence.MultiSentence:
            for multisentence in corpus:
                # Collect the Sentence objects from each MultiSentence
                sentences.extend(multisentence.sentences)
        else:
            sentences = corpus

        # Generate a temporary file for the sentences
        f, in_filepath = tempfile.mkstemp('.txt',
                                          'sents_',
                                          self.path + '/../temp',
                                          text=True)
        out_filepath = in_filepath + '.out'

        # Add in sentence terminators to prevent accidental sentence merging
        try:
            # Write sentences out to the temporary file as strings of words
            with os.fdopen(f, 'w') as temp_file:
                for sentence in sentences:
                    sentence_text = self.preprocess_text(sentence.tokens)
                    print>>temp_file, sentence_text

            # Run Semafor parser on the file and retrieve the typed
            # dependencies.
            # ./release/fnParserDriver.sh $in_filename $out_filename
            process = subprocess.Popen([self.path +
                                            '/release/fnParserDriver.sh',
                                        in_filepath,
                                        out_filepath],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if show_output:
                print stdout
                print stderr

            xml_sent_parses = self.unpack_xml(out_filepath, sentences)
        finally:
            # Delete temporary files
            os.remove(in_filepath)
            if os.path.exists(out_filepath):
                os.remove(out_filepath)

        # Process sentence
        for sentence, xml_sent_parse in zip(sentences, list(xml_sent_parses)):
            frames = self.process_sentence(sentence, xml_sent_parse)

            if not sentence.has_annotation('frames', annotator='semafor'):
                sentence.add_structure(frames, name='frames',
                                       annotator='semafor')

    def preprocess_text(self, raw_tokens):
        """Return a stringified version of the sentence tokens that is
        adjusted to avoid errors.
        """
        return ' '.join(self.preprocess_token(token) for token in raw_tokens)

    def preprocess_token(self, token):
        """Preprocess a single token to avoid weird tokenization by the MST
        parser within Semafor.
        """
        # Drop commas within numbers (e.g., 2,000)
        if re.search(self.numeric_symbols_re, token) is not None:
            token = token.replace(',', '')
            token = token.replace(':', '')

        # Map words that are tokenized differently
        if token in self.token_mappings:
            return self.token_mappings[token]

        return token

    def unpack_xml(self, xml_filepath, sentences):
        """Unpack the Semafor XML format to get subtrees corresponding to
        a frame-semantic parse for each sentence
        """
        etree = ElementTree.parse(xml_filepath)
        corpus = etree.getroot()

        per_sent_parses = None
        for documents in corpus:
            for document in documents:
                for paragraphs in document:
                    for paragraph in paragraphs:
                        for sentences in paragraph:
                            per_sent_parses = sentences
                            break

        return per_sent_parses

    def process_sentence(self, sentence, xml_sent_parse):
        """Process a frame annotation for a single sentence.
        """
        sent_text, annotation_sets = list(xml_sent_parse)

        offset_to_idx = self.get_offset_mappings(sentence, sent_text.text)

        num_tokens = len(sentence.tokens)
        frame_dag = text.structure.DependencyDag(num_tokens)

        for annotation_set in annotation_sets:
            assert len(list(annotation_set)) == 1   # sanity check

            # Each annotation set is a new frame
            frame_name = annotation_set.get("frameName")
            frame_node = frame_dag.add_aux_node(name=frame_name)

            for layers in annotation_set:
                assert len(list(layers)) == 2       # sanity check

                for layer in layers:
                    assert len(list(layer)) == 1    # sanity check

                    layer_name = layer.get("name")

                    for labels in layer:
                        for label in labels:

                            label_name = label.get("name")
                            start_offset = int(label.get("start"))
                            end_offset = int(label.get("end"))

                            start, end = None, None
                            try:
                                start = offset_to_idx[start_offset]
                                end = offset_to_idx[end_offset]
                            except KeyError:
                                print "ERROR: can't map offsets",
                                print "[" + str(start_offset) + ",",
                                print str(end_offset) + "]",
                                print "of frame element", label_name,
                                print "for the frame", frame_name,
                                print "to token(s)", str(start) + ",",
                                print str(end), "in sentence:"
                                print sent_text.text
                                print "\noffset to idx mappings:\n",
                                self.print_offset_mappings(sentence,
                                                           offset_to_idx)
                                raise

                            # We assume for now that the last word in the
                            # span is the head but add an attribute to
                            # each edge indicating the span.
                            end_node = frame_dag.nodes[end]
                            idx_span = list(range(start, end+1))
                            if layer_name == "Target":
                                assert label_name == layer_name
                                tgt_edge = frame_dag.add_edge(
                                        frame_node,
                                        end_node,
                                        lex_idxs=idx_span,
                                        target=True)
                                frame_node.add_attributes(
                                        tgt_idxs=idx_span,
                                        tgt_edge=tgt_edge)
                            elif layer_name == "FE":
                                fe_edge = frame_dag.update_edge(
                                        frame_node,
                                        end_node,
                                        lex_idxs=idx_span,
                                        fe=label_name)
                                frame_node.add_attributes(
                                        fe_edges={label_name: fe_edge})
                            else:
                                print "WARNING: skipping unrecognized layer",
                                print "\'" + layer_name + "\' for sentence",
                                print sent_text
        return frame_dag

    def get_offset_mappings(self, sentence, sent_text):
        """Map start and end character offsets from the SEMAFOR sentence text
        to indices of corresponding tokens in the sentence object.
        """
        # Avoid encoding errors from the XML output
        sent_text = sent_text.encode('utf8')

        offset_to_idx = {}
        offset = 0
        non_ascii_offset = 0
        for w, token in enumerate(sentence.tokens):
            preprocessed_token = self.preprocess_token(token)
            try:
                i = sent_text.find(preprocessed_token, offset)
            except UnicodeDecodeError:
                print sentence.tokens
                print sent_text
                print "Looking for token", w, "i.e.,", token, repr(token)
                raise

            if i is -1:
                print "\nERROR: can't find token '" + preprocessed_token + "'",
                print "starting at offset", offset
                print "in sentence:\n", sent_text
                raise Exception

            offset = i + len(preprocessed_token)

            # The token now covers [i, offset-1] in the given text
            start = i
            end = offset - 1

            offset_to_idx[start + non_ascii_offset] = w

            # UTF8 conversion adds an extra character for each non-ASCII
            # symbol. This must be dropped in the remainder of the sentence
            # to be consistent with SEMAFOR's offsets.
            # NOTE: doesn't account for multiple non-ASCII symbols per token
            try:
                preprocessed_token.decode("ascii")
            except UnicodeDecodeError:
                non_ascii_offset -= 1

            offset_to_idx[end + non_ascii_offset] = w

            # If the preprocessing resulted in splitting up words, record
            # the internal start and end indices of these words as well
            if ' ' in preprocessed_token:
                j = 0       # NOTE: assumes space is not the first character
                while j != -1:
                    j = preprocessed_token.find(' ', j + 1)

                    offset_to_idx[j - 1 + non_ascii_offset] = w
                    offset_to_idx[j + 1 + non_ascii_offset] = w

        return offset_to_idx

    @classmethod
    def print_offset_mappings(cls, sentence, offset_to_idx):
        """Dispay the offset-to-token mapping of a sentence for debugging.
        """
        idx_to_offsets = [[] for token in sentence.tokens]
        for offset, idx in offset_to_idx.iteritems():
            recorded_offsets = idx_to_offsets[idx]
            if len(recorded_offsets) == 0:
                recorded_offsets.append(offset)
            elif len(recorded_offsets) == 1:
                # Keep offsets sorted
                if recorded_offsets[0] <= offset:
                    recorded_offsets.append(offset)
                else:
                    idx_to_offsets[idx] = [offset, recorded_offsets[0]]
            else:
                print "ERROR: too many offsets for token", idx
                print "in sentence:\n",  ' '.join(sentence.tokens)
                print offset_to_idx

        for idx, offsets in enumerate(idx_to_offsets):
            print offsets, "\t\t" if len(offsets) == 1 else "\t",
            print idx, sentence.tokens[idx]
