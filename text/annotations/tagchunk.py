#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import os
import re
import subprocess
import tempfile
import text.sentence


class Tagchunk(object):
    """Interface to the TagChunk tool.
    """
    def __init__(self, path='/home/kapil/research/tools/TagChunk/'):
        """Initialize the path to TagChunk and regular expressions for
        processing its BIO-tagged output format.
        """
        self.path = path
        self.exec_path = ''.join((self.path, 'tagchunk.i686'))
        self.weights_path = ''.join((self.path, 'weights/w-5'))
        self.resources_path = ''.join((self.path, 'resources'))

        self.token_re = re.compile(r"^(.*)_(.*)_(.)-(.*)$")

    def run_on_corpus(self, corpus):
        """Write sentences to a temporary file as strings of words, run
        TagChunk on the file and retrieve the tagged results, then delete the
        file.
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
        f, temp_filepath = tempfile.mkstemp('.txt',
                                            'sents_',
                                            self.path,
                                            text=True)
        try:
            # Write sentences out to the temporary file as strings of words
            with os.fdopen(f, 'w') as temp_file:
                for sentence in sentences:
                    sentence_text = ' '.join(sentence.tokens)
                    print>>temp_file, sentence_text

            # Run TagChunk process and retrieve POS-tagged and chunked
            # sentences in BIO format
            # path/tagchunk.i686 -predict . path/weights/w-5 path/$filename
            # path/resources
            process = subprocess.Popen([self.exec_path,
                                        '-predict',
                                        '.',
                                        self.weights_path,
                                        temp_filepath,
                                        self.resources_path],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

        finally:
            # Delete temporary file
            os.remove(temp_filepath)

        # Split output into strings of BIO-tagged words
        strings_BIO = stdout.split('\n');
        del strings_BIO[-1]

        # Process sentence
        for sentence, string_BIO in zip(sentences, strings_BIO):
            pos_tags, chunks = self.process_BIO_string(string_BIO)

            sentence.add_token_tags(pos_tags, name='pos_tags',
                    annotator='tagchunk')
            sentence.add_span_tags(chunks, name='chunks',
                    annotator='tagchunk')

    def process_BIO_string(self, string_BIO):
        """Convert a sentence string consisting of space-separated words in
        the format word_POS_BIOtag-chunktype into a tuple of POS tags and a
        tuple-to-string dictionary of chunk spans.
        """
        tokens_BIO = string_BIO.split(' ')

        pos_tags = []
        chunks = {}
        current_span_start = None
        current_type = ''
        for i, token_BIO in enumerate(tokens_BIO):
            match_obj = re.match(self.token_re, token_BIO)
            if match_obj:
                word, POS_tag, BIO_tag, chunk_type = match_obj.groups()

                # Append POS to list of pos_tags
                pos_tags.append(POS_tag)

                # If an incomplete chunk exists and is ended by the current
                # token, finish the chunk and add it to the dictionary of
                # chunks
                if current_span_start and BIO_tag != 'I':
                    span = current_span_start, i-1
                    chunks[span] = current_type

                    # Clear current span for future iterations
                    current_span_start = None

                # If the current word begins a new chunk
                if BIO_tag == 'B' and chunk_type != 'O':
                    current_span_start = i
                    current_type = chunk_type
            else:
                print "ERROR: Can't parse BIO string", token_BIO
                raise Exception

        # If an incomplete chunk remains, wrap it up
        if current_span_start:
            span = current_span_start, len(tokens_BIO)-1
            chunks[span] = current_type

        return pos_tags, chunks
