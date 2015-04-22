#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
import glob
from interfaces import gigaword
import os
import re
from stemming import porter2
from text import tokens


class CompressionDoc(object):
    """A single compression document.
    """
    # Regular expressions for extracting line text. Note that XML readers
    # fail here because the IDs are not quoted.
    original_re = re.compile(r"<original id=([^>]+)>(.*)<\/original>$")
    compressed_re = re.compile(r"<compressed id=([^>]+)>(.*)<\/compressed>$")

    def __init__(self, docid, gw, tokenizer=tokens.Tokenizer(),
            stemming=False):
        """Note the doc id.
        """
        self.docid = docid
        self.gw = gw
        self.tokenizer = tokenizer
        self.filepaths = []

        # A list of instances, where each is itself a list of sentences. The
        # first sentence is the original, the remaining are compressions by
        # different annotators.
        self.instance_list = []

        # A mapping from a sentence id to the index of that sentence in the
        # instance list, where the instances from a particular document
        # are located
        self.sentid_idx_mapper = {}

        # Frequency of words in this document
        self.word_counts = defaultdict(int)
        self.stemming = stemming

    def load_from_file(self, filepath):
        """Parse a single compression XML file.
        """
        self.filepaths.append(filepath)
        #print self.docid,

        with open(filepath) as f:
            for line in f:
                if line.startswith('<original'):
                    orig_match = re.match(self.original_re, line)
                    sid, sent = orig_match.groups()
                    self.add_original(sid, sent)
                elif line.startswith('<compressed'):
                    comp_match = re.match(self.compressed_re, line)
                    sid, sent = comp_match.groups()
                    self.add_compression(sid, sent)

    def add_original(self, sent_id, sent):
        """Add an original (uncompressed) sentence.
        """
        sent = self.preprocess_sent(sent)

        if sent_id not in self.sentid_idx_mapper:
            # First time seeing this sentence
            idx = len(self.instance_list)
            self.instance_list.append([sent])
            self.sentid_idx_mapper[sent_id] = idx

            # Count words in this document
            for word in self.tokenizer.tokenize(sent):
                word = word.lower()
                if self.stemming:
                    word = porter2.stem(word)
                self.word_counts[word] += 1
        else:
            # Don't add the original sentence again, just check to see if it's
            # the same
            idx = self.sentid_idx_mapper[sent_id]
            if sent != self.instance_list[idx][0]:
                print "ERROR: mismatched original sentences in", self.docid
                print self.filepaths[0]
                print self.instance_list[idx][0]
                print self.filepaths[-1]
                print sent

    def add_compression(self, sent_id, sent):
        """Add a compressed sentence.
        """
        sent = self.preprocess_sent(sent)

        if sent_id not in self.sentid_idx_mapper:
            print "ERROR:", self.filepaths[-1], "has no original" + \
                    "sentence for", sent_id
        else:
            idx = self.sentid_idx_mapper[sent_id]

            # Look through the original sentence and, if a token isn't found
            # because of a joined period, separate it in the original
            # and update the word counts.
            original_sent = self.instance_list[idx][0]
            original_tokens = set(self.tokenizer.tokenize(original_sent))
            compressed_tokens = set(self.tokenizer.tokenize(sent))

            for mismatch in compressed_tokens - original_tokens:
                if mismatch == '.':
                    continue

                # Check for words which are appended to the following period.
                # NOTE: this is usually due to multiple sentences collapsed
                # into a single sentence in the input.
                target = mismatch + '.'
                if target in original_tokens:
                    print "NOTE: replacing \'%s\' with \'%s\' in %s" % \
                            (target, mismatch + ' .', original_sent)
                    original_sent = original_sent.replace(target,
                                                          mismatch + ' .')
                    self.word_counts[target] -= 1
                    self.word_counts[mismatch] += 1
                    self.word_counts['.'] += 1
                else:
                    print "WARNING: could not match token", mismatch, "in"
                    print "INPUT:", original_sent
                    print "GOLD:", sent

            self.instance_list[idx][0] = original_sent
            self.instance_list[idx].append(sent)

    def preprocess_sent(self, sent):
        """Fix errors in sentences.
        """
        # Fix specific annotator rewrites to ensure reachability and avoid
        # deletion in the following step.
        sent = re.sub(r"\[ Calfa \]", " he ", sent)
        sent = re.sub(r"\[ Franken \]", " Franken ", sent)
        sent = re.sub(r" do it \[ accept the presidency \]",
                      " accept the presidency ", sent)

        # Remove speech transcription artifacts like '[sp]', largely from the
        # broadcast news corpus. These only appear twice in output
        # annotations and are never the majority.
#        sent = sent.replace("[ sp ] ", '').replace("[ sp ? ] ", '')
        sent = re.sub(r"\s*\[[^\[]*\]\s*", " ", sent)

        # Fix broken quoting in the original corpus: trailing end-quotes are
        # reduced to apostrophes.
        if sent.endswith('. \''):
            sent += '\''

        return sent

    def get_tfidf(self, sent):
        """Calculate partial importance score over nouns and verbs as per
        Clarke & Lapata (2008). The score for a noun/word is
        f_i * log (F_a / F_i), where f_i and F_i are the document and corpus
        frequency of the term respectively and F_a is the number of all such
        terms in the corpus. We use the Gigaword corpus (formerly the Penn
        Treebank WSJ section) for the estimation of F_i and F_a.
        """
        scores = {}
        for word in self.tokenizer.tokenize(sent):
            word = word.lower()
            if self.stemming:
                word = porter2.stem(word)

            # We don't check for duplicates because it makes no difference
            scores[word] = self.word_counts[word] * \
                    ( - self.gw.get_logprob(word, alpha=1))

        return scores

    def export_instances(self, corpus, min_length=2, max_length=100,
            one_per_gold=False):
        """Export compression instances in this document to the corpus.
        """
        for instance in self.instance_list:
            # Not tokenizing for checking the length
            orig_words = instance[0].split(' ')
            if max_length is not None and len(orig_words) > max_length:
                print "WARNING: skipping instance with", len(orig_words),
                print "tokens"
                continue
            compressed_words = instance[1].split(' ')
            if min_length is not None and len(compressed_words) < min_length:
                print "WARNING: skipping instance with", len(compressed_words),
                print "tokens"
                continue

            # We hack around the corpus interface to add word scores to each
            # instance
            tfidf_scores = self.get_tfidf(instance[0])
            if one_per_gold:
                # Add the original sentence once for each gold compression
                for i in range(1,len(instance)):
                    corpus.add_instance([[instance[0]]], [instance[i]])
                    setattr(corpus.instances[-1], 'tfidf', tfidf_scores)
            else:
                # Add the original sentence followed by all gold compressions
                corpus.add_instance([[instance[0]]], instance[1:])
                setattr(corpus.instances[-1], 'tfidf', tfidf_scores)


class CompressionCorpus(object):
    """A corpus of deletion-focused sentence compression instances from James
    Clarke's datasets.
    """
    def __init__(self, path='/proj/fluke/users/kapil/resources/compression/',
            name='written-compressions',
            gigaword_path='/proj/fluke/users/kapil/resources/' +
            'gigaword_eng_4/gigaword.pickle'):
        """Read a compression corpus into a GoldTransductionCorpus.
        """
        # Hack to maintain different versions of the broadcastnews corpus.
        #self.one_per_gold = False
        # Now supplied directly at export time.

        corpus_path = path + '/' + name
        self.load_docs(corpus_path, gigaword_path)
        self.load_partitions(corpus_path)

    def load_docs(self, corpus_path, gigaword_path):
        """Load all the documents into the corpus.
        """
        # A mapping from a doc id to a list of indices where the instances
        # from a particular document are located
        self.doc_mapper = {}

        # Initialize the counts of noun and verb stems from a large background
        # corpus (the WSJ section of PTB)
#        wc = wordcounts.WordCounter()
#        nv_counts = wc.get_noun_verb_counts()

        # Initialize word statistics from a large background corpus
        # (Gigaword v4)
        gw = gigaword.CorpusStats.from_pickle(gigaword_path)

        # Load the corpus
        filepaths = glob.glob(corpus_path + '/*/*')
        for filepath in filepaths:
            if os.path.isdir(filepath) or filepath.endswith('gz'):
                continue

            docid = filepath.split('/')[-1]
            if docid[-10:-1] == 'annotator':
                docid = docid[:-11]
            try:
                doc = self.doc_mapper[docid]
            except KeyError:
                doc = CompressionDoc(docid, gw)
                self.doc_mapper[docid] = doc

            doc.load_from_file(filepath)

    def load_partitions(self, corpus_path):
        """Load the corpus paritions.
        """
        # A mapping from a predefined corpus partition to doc ids.
        self.partition_mapper = {}

        # Load the corpus
        filepaths = glob.glob(corpus_path + '/*.filelist')
        for filepath in filepaths:
            partition = filepath[filepath.rfind('/')+1:filepath.rfind('.')]
            print partition, "partition:", filepath

            with open(filepath) as f:
                self.partition_mapper[partition] = [line.strip() for line in f]

    def export_instances(self, corpus, one_per_gold=False, **kwargs):
        """Export the instances as a GoldTransductionCorpus.
        """
        last_offset = len(corpus.instances)
        for partition in ('train', 'dev', 'test'):
            for docid in self.partition_mapper[partition]:
                doc = self.doc_mapper[docid]
                doc.export_instances(corpus, one_per_gold=one_per_gold,
                                     **kwargs)

            # Set the partition slices in the corpus object
            next_offset = len(corpus.instances)
            kwarg = {partition: (last_offset, next_offset)}
            corpus.set_slices(**kwarg)
            print partition, "partition:", next_offset - last_offset
            last_offset = next_offset
