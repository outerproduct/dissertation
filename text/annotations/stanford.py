#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import os
import re
import subprocess
import tempfile
import text.sentence
import text.structure


class Stanford(object):
    """Interface to the Stanford dependency parser.
    """
    def __init__(self,
            path='/home/kapil/research/tools/stanford-parser-2012-11-12'):
            #path='/proj/fluke/users/kapil/tools/stanford-parser-2012-11-12'):
        """Initialize the path to the Stanford parser.
        """
        self.path = path

        # Regular expression for the typed dependency format
        self.dep_re = re.compile(
                r"^([\w\d]+)\((.+)\-([\d]+), (.+)\-([\d]+)\)$")
        self.inner_punc_re = re.compile(r"^(.*\w?)[!\?]+(\w.*)$")
        self.abbrev_re = re.compile(r'[A-Z].*\.')

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
        f, temp_filepath = tempfile.mkstemp('.txt',
                                            'sents_',
                                            self.path,
                                            text=True)

        # Add in sentence terminators to prevent accidental sentence merging
        try:
            # Write sentences out to the temporary file as strings of words
            with os.fdopen(f, 'w') as temp_file:
                for sentence in sentences:
                    sentence_text = self.preprocess_text(sentence.tokens)
                    print>>temp_file, sentence_text

            # Run Stanford parser on the file and retrieve the typed
            # dependencies.
            # java -mx200m -cp "path/*:"
            # edu.stanford.nlp.parser.lexparser.LexicalizedParser
            # -outputFormat "typedDependencies" -outputFormatOptions
            # "basicDependencies,includePunctuationDependencies"
            # -retainTmpSubcategories
            # edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz
            # $filename
            classname = 'edu.stanford.nlp.parser.lexparser.LexicalizedParser'
            output_format = ['typedDependencies']
            output_format_options = ['basicDependencies',
                                     'includePunctuationDependencies']
            model_path = 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
            process = subprocess.Popen(['java',
                                        '-mx500m',
                                        '-cp', self.path + '/*:',
                                        classname,
                                        '-outputFormat',
                                        ','.join(output_format),
                                        '-outputFormatOptions',
                                        ','.join(output_format_options),
                                        '-retainTmpSubcategories',
                                        '-sentences', 'newline',
                                        '-tokenized',
                                        model_path,
                                        temp_filepath],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if show_output:
                print stdout
                print stderr
        finally:
            # Delete temporary file
            os.remove(temp_filepath)

        # Split output into strings of per-sentence parses
        parse_strings = stdout.split('\n\n');
        parse_strings.pop()

        # Process sentence
        for sentence, parse_string in zip(sentences, parse_strings):
            dparse = self.process_parse(sentence, parse_string)

            if not sentence.has_annotation('dparse', annotator='stanford'):
                sentence.add_structure(dparse, name='dparse',
                                       annotator='stanford')

    def preprocess_text(self, raw_tokens):
        """Return a stringified version of the sentence tokens that is
        adjusted to avoid parser errors.
        """
        fixed_tokens = []
        for t, token in enumerate(raw_tokens):
            # Drop trailing periods, which trigger sentence splitting
            #if token.endswith('.'):# and t != len(raw_tokens) - 1:
            rstripped = token.rstrip('.?!')
            if len(rstripped) > 0:
                token = rstripped

#            # Replace inner punctuation (e.g., C?te d' Ivore from RTE) with
#            # underscores
            token = re.sub(self.inner_punc_re, '\\1_\\2', token)

            # Copy the token to avoid writing over the sentences's own tokens
            fixed_tokens.append(token)
        return ' '.join(fixed_tokens)

    def process_parse(self, sentence, parse, tree=True, check_tokens=True):
        """Process the Stanford parse format.
        """
        num_tokens = len(sentence.tokens)
        dependencies = parse.split('\n')
        if check_tokens and num_tokens != len(dependencies):
            print "ERROR: Mismatch between tokens and dependencies in:",
            print ' '.join(sentence.tokens),
            print '(' + str(num_tokens), 'tokens)',
            print parse
            print '(' + str(len(dependencies)), 'tokens)'
            raise Exception

        if tree:
            dparse = text.structure.DependencyTree(num_tokens)
        else:
            dparse = text.structure.DependencyDag(num_tokens)

        for dependency in dependencies:
            match = re.match(self.dep_re, dependency)
            if match is None:
                print "ERROR: Unexpected Stanford dependency format"
                print dependency
                raise Exception

            label, token0, t0, token1, t1 = match.groups()
            if label == 'root':
                continue
            dparse.add_edge(src=int(t0) - 1, tgt=int(t1) - 1, label=label)

        return dparse
