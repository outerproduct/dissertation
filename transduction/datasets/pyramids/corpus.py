#!/usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import numpy as np
import pyramids, sourcedocs
import random
import sys
import unicodedata


class PyramidCorpus(object):
    """A class to represent a DUC/TAC Corpus with Pyramid annotations
    for a given year.
    """
    # A list of tuples of paths to documents that have SCU annotations.
    docs = ['docs/2005/SCU-MarkedCorpus/*.scu',
            'docs/2006/SCU-marked corpus/*.scu',
            'docs/2007/2007 SCU-marked corpus/*.scu',
            'docs/2007/2007 SCU-marked update corpus/*.scu',
            'docs/2008/2008 SCU-marked update corpus/*.scu',
            ]

    # A list of tuples of paths to pyramid files
    evals = ['evals/2005/9models/9models/pyramids_final/*/*.pyr',
             'evals/2006/allpyramids/*.pyr',
             'evals/2007/mainPyramidEval/allpyramids/*.pyr',
             'evals/2007/updateEval/Pyramid/allpyramids/*.pyr',
             'evals/2008/manual/pyramids/*.pyr',
             'evals/2009/manual/pyramids/*.pyr',
             'evals/2010/manual/pyramids/*.pyr',
             'evals/2011/manual/pyramids/*.pyr',
            ]

    def __init__(self, basepath):
        """Store corpus location.
        """
        self.basepath = basepath

    def load_src_collections(self):
        """Load the annotated source documents.
        """
        self.collections = defaultdict(list)
        for doc_path in self.docs:
            year = int(doc_path[5:9])

            scu_files = glob.glob(self.basepath + '/' + doc_path)
            for scu_file  in scu_files:
                collection = sourcedocs.SourceDocs(scu_file, year)
                cid = (year, collection.docid)

                if cid in self.collections:
                    print "WARNING: duplicate annotated doc", collection.docid,
                    print "for year", year
                self.collections[cid].append(collection)

    def load_pyramids(self):
        """Load the pyramid evaluation documents.
        """
        self.pyramids = {}
        for eval_path in self.evals:
            year = int(eval_path[6:10])

            pyr_files = glob.glob(self.basepath + '/' + eval_path)
            for pyr_file  in pyr_files:
                sys.stdout.write(pyr_file[len(self.basepath)+1:] +
                                 ' '*15 + '\r')
                pyramid = pyramids.Pyramid(pyr_file, year)
                pid = (year, pyramid.docid)

                assert pid not in self.pyramids
                self.pyramids[pid] = pyramid

    def get_srcdoc_fusions(self):
        """Return fusions from source doc sentences to SCU labels.
        """
        if not hasattr(self, 'collections'):
            self.load_src_collections()

        fusions = []
        for source_docs in self.collections.itervalues():
            for source_doc in source_docs:
                fusions.extend(source_doc.get_fusions())
        return fusions

    def get_pyr_fusions(self, **kwargs):
        """Return fusions from summary sentences to SCU labels.
        """
        if not hasattr(self, 'pyramids'):
            self.load_pyramids()

        fusions = []
        for pyramid in self.pyramids.itervalues():
            fusions.extend(pyramid.get_fusions(**kwargs))
        return fusions

    def get_xref_fusions(self, **kwargs):
        """Return fusions from source doc sentences to summary sentences.
        """
        if not hasattr(self, 'collections'):
            self.load_src_collections()
        if not hasattr(self, 'pyramids'):
            self.load_src_pyramids()

        fusions = []
        for pid, pyramid in self.pyramids.iteritems():
            year, docid = pid
            if year > 2008:
                continue

            collections = self.collections[pid]
            pyramid.map_source_docs(collections)
            fusions.extend(pyramid.get_xref_fusions(**kwargs))
        return fusions

    @classmethod
    def fusion_summary(cls, fusion_instances, num_samples=3, max_sents=4,
            size=True, year=True, size_year=True):
        """Print fusion statistics.
        """
        print
        print "Num fusion instances:", len(fusion_instances)
        print "Avg cardinality:", np.mean([instance.get_cardinality()
                for instance in fusion_instances])
        print "Avg length ratio:", np.mean([instance.get_length_ratio()
                for instance in fusion_instances])
        print

        size_distribution = defaultdict(int)
        year_distribution = defaultdict(int)
        def dd(): return defaultdict(int)
        size_year_distribution = defaultdict(dd)
        num_identical_inputs = 0
        num_identical_contribs = 0
        for instance in fusion_instances:
            size_distribution[len(instance.input_sents)] += 1
            year_distribution[instance.year] += 1
            size_year_distribution[len(instance.input_sents)][instance.year] \
                    += 1

            if len(instance.input_sents) <= max_sents:
                if (instance.output_sent + '.') in instance.input_sents:
                    num_identical_inputs += 1
                if instance.output_sent in instance.labels:
                    num_identical_contribs += 1

        if size:
            print "Size distribution"
            for key in sorted(size_distribution.iterkeys()):
                print key, '-', size_distribution[key]
#            cls.distplot(size_distribution, "# input sentences",
#                         "# instances", range(0,1100,200))

        if year:
            print "Year distribution"
            for key in sorted(year_distribution.iterkeys()):
                print key, '-', year_distribution[key]
#            cls.distplot(year_distribution, "year", "# instances",
#                    range(0,600,100))

        if size_year:
            print "Size x year distribution"
            for size in sorted(size_year_distribution.iterkeys()):
                for year in sorted(size_year_distribution[size].iterkeys()):
                    print size, '-', year, '-',
                    print size_year_distribution[size][year]

        print "Identical inputs", num_identical_inputs
        print "Identical contribs", num_identical_contribs

        for i in range(num_samples):
            j = random.randrange(0, len(fusion_instances) - 1)
            while len(fusion_instances[j].input_sents) > max_sents:
                j = random.randrange(0, len(fusion_instances) - 1)
            fusion_instances[j].print_summary()

    @classmethod
    def distplot(cls, dist, xlabel, ylabel, yticks):
        """Plot a distribution.
        """
        # Outward ticks
        from matplotlib import rcParams
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'

        sorted_keys = sorted(dist.keys())
        sorted_vals = [dist[key] for key in sorted_keys]
        min_key = sorted_keys[0]
        max_key = sorted_keys[-1]
        key_scale = (max_key - min_key + 1) / len(sorted_keys)
        x_room = 0.49 * key_scale

        fig = plt.figure()
        plt.bar(sorted_keys, sorted_vals,
                width = 1,
                bottom = 3,
                color = 'k',
                edgecolor = [0.5,0.5,0.5],
                linewidth = 0,
                align = 'center')
        plt.axis([min_key - x_room, max_key + x_room, 0, yticks[-1]])
        for ax in fig.axes:
            ax.yaxis.set_label_text(ylabel, size=17, weight='roman')
            ax.yaxis.grid(True)
            ax.yaxis.set_ticks(yticks)
            for label in ax.yaxis.get_ticklabels():
                label.set_fontsize(17)

            ax.xaxis.set_label_text(xlabel, size=17, weight='roman')
            labels = ax.xaxis.set_ticklabels([''] + \
                    [str(key) for key in sorted_keys])
            for label in labels:
                label.set_fontsize(17)
        plt.show()

    @staticmethod
    def normalize(text):
        """Normalize unicode text.
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore') \
                if isinstance(text, unicode) \
                else text

    @staticmethod
    def shuffle_across_years(fusion_instances, instances_by_year):
        """Shuffle the instances to randomize by year.
        """
        counter = defaultdict(int)
        for year in instances_by_year.iterkeys():
            counter[year] = 0
        idx_to_year = sorted(counter.iterkeys())
        current_idx = 0

        shuffled_instances = []
        while len(shuffled_instances) < len(fusion_instances):
            # Pick a year
            year = idx_to_year[current_idx]
            current_idx = (current_idx + 1) % len(counter)

            # Add an instance from that year, if available
            if counter[year] < len(instances_by_year[year]):
                    shuffled_instances.append(
                        instances_by_year[year][counter[year]])
                    counter[year] += 1

        # Reverse the instances to ensure that dev and test distributions
        # come from all years.
        #return shuffled_instances
        return reversed(shuffled_instances)

    def export_instances(self, corpus, corpus_split=None, use_labels=False,
            **kwargs):
        """Export these instances to a fusion corpus.
        """
        fusion_instances = self.get_pyr_fusions(**kwargs)
        instances_by_year = defaultdict(list)
        for instance in fusion_instances:
            instances_by_year[instance.year].append(instance)

        if corpus_split is not None:
            # Randomly shuffle the instances across years
            final_instances = self.shuffle_across_years(fusion_instances)
        else:
            # Use the TAC corpus for training and DUC for testing
            final_instances = []
            for year in range(2008,2012):
                final_instances.extend(instances_by_year[year])
            for year in range(2005,2008):
                final_instances.extend(instances_by_year[year])

        # Append the instances to the corpus
        for instance in final_instances:
            gold_sents = [self.normalize(instance.output_sent)]
            if use_labels:
                input_sents = [[self.normalize(label)]
                    for label in instance.labels]
                corpus.add_instance(input_sents, gold_sents)
            else:
                input_sents = [[self.normalize(sent)]
                    for sent in instance.input_sents]
                label_sents = [self.normalize(label)
                    for label in instance.labels]
                corpus.add_instance(input_sents, gold_sents,
                        label_sentences=label_sents)

        # Enforce the corpus split
        num_instances = len(corpus.instances)
        if corpus_split is not None:
            assert sum(corpus_split) == 100
            train_test = int(corpus_split[0] * 0.01 * num_instances)
        else:
            train_test = sum(len(instances_by_year[year])
                           for year in range(2008,2012))
        corpus.set_slices(train=(0,train_test),
                          test=(train_test, num_instances))

        print ' '.join(("Train:", str(len(corpus.train_instances)),
                str(len(corpus.train_instances) / num_instances * 100) + '%'))
        print ' '.join(("Test:", str(len(corpus.test_instances)),
                str(len(corpus.test_instances) / num_instances * 100) + '%'))


if __name__ == '__main__':

    pyr_corpus = PyramidCorpus('/proj/fluke/users/kapil/resources/DUC')
    print "\nBEFORE FILTERING"
    fusions = pyr_corpus.get_pyr_fusions(
                corpus_split=None,
                use_labels=False,
                skip_exact_lines=False,
                skip_exact_labels=False,
                min_inputs=1,
                max_inputs=7,
                min_words=1,
                max_words=700,
                drop_verbless=False,
                single_sent_input=False,
                min_part_line_ratio=0,
                min_scu_part_ratio=0,
                min_scu_line_overlap=0)
    pyr_corpus.fusion_summary(fusions, num_samples=0, max_sents=7,
                              size=True, year=True, size_year=False)


    pyr_corpus = PyramidCorpus('/proj/fluke/users/kapil/resources/DUC')
    print "\nAFTER FILTERING"
    fusions = pyr_corpus.get_pyr_fusions(
                corpus_split=None,
                use_labels=False,
                skip_exact_lines=False,
                skip_exact_labels=False,
                min_inputs=2,
                max_inputs=4,
                min_words=5,
                max_words=100,
                min_part_line_ratio=0.5,
                min_scu_part_ratio=0.5,
                min_scu_line_overlap=1)
    pyr_corpus.fusion_summary(fusions, num_samples=0, max_sents=4,
                              size=True, year=True, size_year=False)
    print "\nTOTAL", len([scu for pyramid in pyr_corpus.pyramids.itervalues()
        for scu in pyramid.scus.itervalues()])

#    pyr_corpus.fusion_summary(pyr_corpus.get_srcdoc_fusions())
#    pyr_corpus.fusion_summary(pyr_corpus.get_xref_fusions())
