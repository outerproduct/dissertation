#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import itertools
import os
import re
import subprocess
import sys
import text.sentence
import text.structure


class Rasp(object):
    """Interface to the Rasp dependency parser.
    """
    def __init__(self, path='/home/kapil/research/tools/rasp3os/scripts'):
            #path='/proj/fluke/users/kapil/tools/rasp3os/scripts'):
        """Initialize the path to the RASP parser.
        """
        self.path = path

        # Regular expre;ssion for the relation format
        label = r"\|([\w\d]+)\|"
        subtype0 = '|'.join((r" _", r" \|[^\:]+\|"))
        subtype1 = '|'.join((subtype0, r" \|that:[^\|]*\|"))
        token = r" \|.+\:(\d+)_[A-Z0-9\$\.\&]+\|"
        token_or_to = '|'.join((token, r" \|to\|"))
        self.rel_re = re.compile(r"^\(" +
                                 label +
                                 r"(?:" + subtype0 + ")?" +
                                 r"(?:" + subtype1 + ")?" +
                                 token +
                                 r"(?:" + token_or_to + ")" +
                                 r"(?:" + subtype0 + ")?" +
                                 r"\)")

        self.splitter_re = re.compile(r"\|(?: _)? \|")
        self.token_re = re.compile(r"(.+)\:(\d+)_[A-Z0-9\$\.]+")

        self.inner_punc_re = re.compile(r"^(.*\w?)[!\?]+(\w.*)$")
        self.abbrev_re = re.compile(r'[A-Z].*\.')

    def run_on_corpus(self, corpus, show_output=False):
        """Write sentences to a temporary file as strings of words, run the
        RASP parser on the file and retrieve the parsed results,
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

#        input_text = '\n\n'.join(self.preprocess_text(sentence.tokens)
#                for sentence in sentences)

        rasp_env = os.environ.copy()
        rasp_env['rasp_tokenise'] = 'cat'
        for s, sentence in enumerate(sentences):
            sys.stdout.write(str(s) + "/" + str(len(sentences)) + "\r")

            process = subprocess.Popen([self.path + '/rasp.sh', '-m'],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        env=rasp_env)
            sent_text = self.preprocess_text(sentence.tokens)
            stdout, stderr = process.communicate(input=sent_text)
            if show_output:
                print stdout
                print stderr
            relgraph = self.process_parse(sentence, stdout)

            if not sentence.has_annotation('relgraph', annotator='rasp'):
                sentence.add_structure(relgraph, name='relgraph',
                                       annotator='rasp')

        # Run RASP parser on the file and retrieve grammatical relations.
        # ./rasp.sh < $text -m

        # Turn off sentence boundary detection and tokenization to
        # maintain consistency with the input.

        # Split output into strings of per-sentence parses
#        parse_strings = stdout.split('\n\n\n');
#        parse_strings.pop()
#
#        # Process sentence
#        for sentence, parse_string in zip(sentences, parse_strings):

    def preprocess_text(self, raw_tokens):
        """Return a stringified version of the sentence tokens that is
        adjusted to avoid parser errors.
        """
        fixed_tokens = []
        for t, token in enumerate(raw_tokens):
            # Drop trailing periods and ellipses, which trigger sentence
            # splitting.
            rstripped = token.rstrip('.?!')
            if len(rstripped) > 0:
                token = rstripped
            else:
                token = ';'

            # Drop trailing periods, which trigger sentence splitting
            #if token.endswith('.'):# and t != len(raw_tokens) - 1:
            rstripped = token.rstrip('.?!')
            if len(rstripped) > 0:
                token = rstripped

            # Replace inner punctuation (e.g., C?te d' Ivore from RTE) with
            # underscores
            token = re.sub(self.inner_punc_re, '\\1_\\2', token)

            # Copy the token to avoid writing over the sentences's own tokens
            fixed_tokens.append(token)
        return ' '.join(fixed_tokens)

    def process_parse(self, sentence, parse, check_tokens=True):
        """Process the grammatical relations from the RASP format.
        """
        num_tokens = len(sentence.tokens)
        relations = parse.split('\n')
        relgraph = text.structure.DependencyGraph(num_tokens)

        for relation in itertools.islice(relations, 2, len(relations)):
            # Skip blank lines and known problems
            if not relation.startswith('(') or \
                    relation.startswith('(|passive|') or \
                    relation.startswith('(|ta| |quote| |ellip|') or \
                    relation.startswith('(|conj| |;|') or \
                    relation.startswith('(|ncmod| |poss| |ellip|'):
                continue

            match = re.match(self.rel_re, relation)
            if match is None:
                print "WARNING: Unexpected RASP relation format"
                print relation, '\n'
                print ' '.join(sentence.tokens)
                continue
            label, t0, t1 = match.groups()
            t0 = int(t0) - 1
            t1 = int(t1) - 1 if t1 is not None \
                    else sentence.tokens.index('to', t0)  # next 'to'

            relgraph.add_edge(src=t0, tgt=t1, label=label)

        return relgraph
