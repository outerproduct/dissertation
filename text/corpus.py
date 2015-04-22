#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import cPickle
import instance as instancemod
import os.path
import re
import sys
from text import annotations
from utils import timer


class Corpus(object):
    """A corpus consisting of instances, each of which contains Sentence-like
    objects. Supports text annotation of sentences and pickling. Subclass for
    further specialization.
    """
    def __init__(self, name, restore=True, path='.',
            instancecls=instancemod.Instance):
        """Initialize a corpus from an iterable using a given Instance class.
        """
        self.name = name
        self.savepath = ''.join((path, '/', self._sanitize_filename(name),
                                 '.corpus'))
        self.instancecls = instancecls
        self.loaded = False

        # Try restoring from a saved corpus
        if restore and self.restore():
            self.loaded = True
        else:
            # Initialize a default empty corpus
            self.instances = []
            self.annotations = {}

            sys.stderr.write("Initializing new corpus \'" + self.name +
                             "\' at " + self.savepath + "\n")

    def _sanitize_filename(self, filename):
        """Sanitize a string to produce a valid filename.
        """
        filename = re.sub(r"[^\w \+\-\,\.]", '', filename.lower())
        filename = re.sub(r" +", '_', filename)
        return filename

    def add_instance(self, sentences, idx=None, **kwargs):
        """Add a single new instance.
        """
        new_instance = self.instancecls(sentences,
                idx=len(self.instances) if idx is None else idx,
                **kwargs)
        self.instances.append(new_instance)

    def set_slices(self, **kwargs):
        """Generate an indexable slice of the instances to indicate a training,
        test or development corpus.
        """
        for key, value in kwargs.iteritems():
            if len(value) == 0:
                # If given an empty tuple, delete the slice
                delattr(self, key + '_slice')
                delattr(self, key + '_instances')
            elif len(value) == 2:
                # If given a 2-value tuple, generate the slice
                setattr(self, key + '_slice', value)
                setattr(self, key + '_instances',
                        self.instances[value[0]:value[1]])
            else:
                print "ERROR: invalid slice format", value
                print "Slice must be a (begin, end+1) tuple or empty"
                raise Exception

    def retrieve_slice(self, name=None, idxs=None):
        """Retrieve all instances from a named slice or, if specified, a list
        of instance indices.
        """
        if idxs is not None:
            # Shortcut for convenience: if idxs is [-N], the first N
            # instances are returned.
            if len(idxs) == 1 and idxs[0] < 0:
                idxs = range(0, -idxs[0])

            idxs_set = set(idxs)
            return [instance for instance in self.instances
                    if instance.idx in idxs_set]
        elif name is None:
            # Copy instance refs for ordering safety
            return self.instances[:]
        else:
            slice_name = name + '_instances'
            try:
                return getattr(self, slice_name)
            except AttributeError:
                raise
                print "Ensure that corpus.set_slices(" + name + \
                        "=(BEGIN,END+1)) was run."

    def annotate_with(self, *annotator_names):
        """Choose annotations to supply for all sentences in the corpus. Note
        that we assume all annotations can be supplied at the level of
        Sentence objects.
        """
        if len(annotator_names) == 0:
            print "WARNING: no annotator specified"
            return
        if len(annotator_names) == 1 and \
                (isinstance(annotator_names[0], tuple) or
                 isinstance(annotator_names[0], list)):
            # If annotators are provided in a list or tuple
            annotator_names = annotator_names[0]

        # Collect sentences for annotation
        all_sents = [sent for instance in self.instances
                          for sent in instance.get_sentences()]

        for annotator_name in annotator_names:
            # Collect sentences which haven't been annotated by this annotator
            unannotated_sents = all_sents
            if annotator_name in self.annotations:
                len_annotated = self.annotations[annotator_name]
                if len_annotated < len(all_sents):
                    unannotated_sents = all_sents[len_annotated:]
                else:
                    # All sentences are already annotated
                    continue

            # Load the annotator with a dynamic import
            annotator = annotations.load_annotator(annotator_name)

            # TODO: perhaps a command-line argument?
            batch_size = len(unannotated_sents)
            idx = 0
            while idx < len(unannotated_sents):
                batch_sents = unannotated_sents[idx:idx+batch_size]
                idx += batch_size

                # Annotate the sentences in this batch with this annotator
                with timer.AvgTimer(len(batch_sents)):
                    sys.stderr.write("\nAnnotating " +
                            str(len(batch_sents)) + "/" +
                            str(len(all_sents)) + " sentences with " +
                            annotator_name + " ")
                    annotator.run_on_corpus(batch_sents)

                # Note the annotations in a corpus-local list
                # TODO: maybe include tag names for usage?
                self.annotations[annotator_name] = min(idx, len(all_sents))

                # Save the newly-annotated corpus
                self.save()

    def restore(self):
        """Restore a pickled corpus.
        """
        if not os.path.exists(self.savepath):
            return False

        sys.stderr.write("Restoring \'" + self.name + "\' corpus from " +
                         self.savepath + "\n")
        with open(self.savepath) as f:
            other = cPickle.load(f)
            self.__dict__.update(other.__dict__)
        return True

    def save(self):
        """Pickle this corpus object and store it in the given path.
        """
        sys.stderr.write("Saving \'" + self.name + "\' corpus to " +
                         self.savepath + "\n")
        with open(self.savepath, 'wb') as f:
            cPickle.dump(self, f, 2)

    # TODO Define an instance yielder ?
