#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import annotations
from pyutilib.enum import Enum
import scipy as sp
import structure
from tokens import tokenizer, untokenizer


class Sentence(object):
    """A class to support word-based, span-based and structural annotations
    over a sentence.
    """
    # An enumerator for the type of annotations currently supported
    tag_type = Enum('TOKEN', 'SPAN', 'STRUCT')

    def __init__(self, text):
        """Initialize with a list of words or a string representing a sentence.
        """
        # If a string is supplied, it is tokenized using the standard Tokenizer
        if isinstance(text, basestring):
            self.raw = text
            self.tokens = tuple(self._tokenize(text))
        else:
            self.raw = ' '.join(text)
            self.tokens = tuple(text)

        # Keep track of length for quick lookups
        # TODO: maybe create a 'SCALAR' tag type for fields like this
        self.length = len(self.tokens)

        # "annotator_name" -> tag type
        self.annotation_types = {}

        # "name" -> list of annotators
        self.annotators = {}

    def _tokenize(self, text, tokenizer=tokenizer.Tokenizer()):
        """Tokenize the raw text of the sentence in a standard way that should
        be compatible with all potential annotation schemes.
        """
        # Default argument tokenizer will only be initialized once
        # TODO: should this be a classmethod?
        return tokenizer.tokenize(text)

    def untokenize(self):
        """Untokenize the words of the sentence in a standard way and return a
        readable sentence.
        """
        return untokenizer.Untokenizer.untokenize(self.tokens)

    def has_annotation(self, name, annotator=None):
        """Return whether the sentence has a particular annotation, optionally
        specifying the annotator. The default annotator used for any
        annotation is the last one added.
        """
        if annotator is None:
            return hasattr(self, name)
        else:
            return hasattr(self, '_'.join((annotator.lower(), name)))

    def get_annotation(self, name, annotator=None):
        """Return an annotation, optionally specifying a particular
        annotator. The default annotator used for any annotation is the
        last one added.
        """
        if annotator is None:
            return getattr(self, name)
        else:
            return getattr(self, '_'.join((annotator.lower(), name)))

    def _save_annotation(self, annotation, name, annotator, tag_type):
        """Save the annotation.
        """
        tag_name = '_'.join((annotator.lower(), name))
        if hasattr(self, tag_name):
            print "ERROR: Annotation name \'", tag_name, "\' already in use"
            raise Exception
        setattr(self, tag_name, annotation)

        # Redirect the default annotation name to this newest annotation
        setattr(self, name, getattr(self, tag_name))

        # Update metadata
        self.annotation_types[tag_name] = tag_type
        annotator_name = annotator.lower()
        if name not in self.annotators:
            self.annotators[name] = [annotator_name]
        elif annotator_name not in self.annotators[name]:
            self.annotators[name].append(annotator_name)

    def annotate_with(self, *annotator_names):
        """Annotate this sentence with the given annotators.
        """
        annotations.annotate(self, *annotator_names)

    def add_token_tags(self, tags, name, annotator):
        """Add annotations in the form of a list of tags, one for each word.
        The list must be the same size as the list of words. A common use-case
        is the introduction of part-of-speech tags for the sentence.
        """
        if len(tags) != self.length:
            print len(tags), " tags given for ", self.length, " words"
            print "ERROR: Number of tags must equal number of words"
            raise Exception
        else:
            self._save_annotation(tags, name, annotator, self.tag_type.TOKEN)

    def add_span_tags(self, spans, name, annotator):
        """Add span annotations in the form of a dictionary of begin/end
        indices to labels. The begin and end indices should describe a valid
        span in the sentence. A common use-case is the introduction of chunk
        tags for the sentence.
        """
        error = False
        for span in spans.keys():
            if span[0] > span[1]:
                error = True
                print "ERROR: Span ", span, " is badly formed"
                print "Start index greater than end index"
                raise Exception
            elif span[0] < 0 or span[1] >= self.length:
                error = True
                print "ERROR: Span ", span, " is badly formed"
                print "Index beyond sentence indexing boundaries"
                raise Exception

        if not error:
            self._save_annotation(spans, name, annotator, self.tag_type.SPAN)

    def add_structure(self, structure, name, annotator):
        """Add structural annotations in the form of an object which
        contains relations between words in which both the words and relations
        may have tags.
        """
        self._save_annotation(structure, name, annotator, self.tag_type.STRUCT)


class MultiSentence(object):
    """A wrapper class around Sentences that allows snippets consisting of
    multiple sentences to be handled as Sentence objects themselves.
    """
    def __init__(self, sentence_list, copy=False):
        """Initialize with a list of Sentence objects.
        """
        if copy:
            self.sentences = sentence_list[:]
        else:
            self.sentences = sentence_list

        # TODO: allow MultiSentences to be created directly from text by
        # splitting sentences.

        # Add a descriptor for the 'tokens' field to allow the MultiSentence
        # to be accessed like a regular Sentence
        self._add_descriptor('tokens', Sentence.tag_type.TOKEN)

        # Just use the sum of the sentence lengths for the 'length' field
        self.length = sum(sentence.length for sentence in self.sentences)

        # Initialize annotation metadata by merging the metadata from the
        # component sentences. Then update the descriptors to account for all
        # available annotations
        self._merge_metadata()
        self.update_descriptors()

    def __getattribute__(self, name):
        """Allow instance-level descriptors. Works for non-data descriptors
        only; to allow data descriptors, __setattr__ should also be modified.
        Source: http://blog.brianbeck.com/post/74086029/instance-descriptors
        """
        try:
            # Access the descriptor for the required annotation
            value = object.__getattribute__(self, name)
        except AttributeError:
            # Avoid recursion in update_descriptors() call
            if name in ('sentences', 'annotators', 'annotation_types'):
                raise AttributeError

            # Try updating the descriptors in case annotations were added to
            # the component Sentences
            self.update_descriptors()
            value = object.__getattribute__(self, name)

        if hasattr(value, '__get__'):
            value = value.__get__(self, self.__class__)
        return value

    def _merge_metadata(self):
        """Merge the metadata of the component Sentences and, crucially,
        change their references to point to the MultiSentence's version.
        This implies that the annotations_types and annotators information
        of the component Sentences is now linked. Note that this does
        avoid cyclic references.
        """
        # TODO: write an exporter for component Sentences that undoes this
        # merging and copies out the metadata
        for s, sent in enumerate(self.sentences):
            if s == 0:
                self.annotation_types = sent.annotation_types
                self.annotators = sent.annotators
            else:
                if self.annotation_types != sent.annotation_types:
                    print "ERROR: inconsistent annotation_type mapper",
                    print "seen when merging sentences"
                    print self.annotation_types
                    print sent.annotation_types
                    raise Exception
                sent.annotation_types = self.annotation_types

                if self.annotators != sent.annotators:
                    print "ERROR: inconsistent annotators mapper",
                    print "seen when merging sentences"
                    print self.annotators
                    print sent.annotators
                    raise Exception
                sent.annotators = self.annotators

    def _add_descriptor(self, tag_name, tag_type):
        """Provide access to new annotations on the component Sentences.
        """
        setattr(self, tag_name, AnnotationDesc(tag_name, tag_type))

    def update_descriptors(self):
        """Update the available MultiSentence descriptors from the tags in the
        component Sentences.
        """
        # Record the existing annotation types in the first Sentence and add
        # appropriate descriptors for each one
        for tag_name, tag_type in self.annotation_types.iteritems():
            self._add_descriptor(tag_name, tag_type)

        # Now update the default annotation names (without annotator names)
        # to link to the most recently added tags for that name.
        for name, annotators in self.annotators.iteritems():
            tag_name = '_'.join((annotators[-1].lower(), name))
            setattr(self, name, getattr(self, tag_name))

    def has_annotation(self, name, annotator=None):
        """Return whether the sentence has a particular annotation, optionally
        specifying the annotator. The default annotator used for any
        annotation is the last one added.
        """
        if annotator is None:
            return hasattr(self, name)
        else:
            return hasattr(self, '_'.join((annotator.lower(), name)))

    def get_annotation(self, annotation_name, annotator=None):
        """Return an annotation, optionally specifying a particular
        annotator. The default annotator used for any annotation is the
        last one added.
        """
        if annotator is None:
            return getattr(self, annotation_name)
        else:
            return getattr(self, '_'.join((annotator.lower(),
                annotation_name)))


class AnnotationDesc(object):
    """A descriptor used by MultiSentence to combine the annotations of its
    constituent Sentences.
    """
    def __init__(self, tag_name, tag_type):
        """Initialize the name and type of the annotations.
        """
        self.tag_name = tag_name
        self.tag_type = tag_type

    def __get__(self, obj, cls=None):
        """Retrieve the cached result if present, otherwise combine the tags.
        """
        # If the cache hasn't been defined, get combined tags
        if hasattr(self, 'cache'):
            return self.cache
        elif len(obj.sentences) == 1:
            # If the MultiSentence contains just one Sentence, we just need a
            # reference to this Sentence's annotations
            self.cache = obj.sentences[0].__dict__[self.tag_name]
            return self.cache
        else:
            # Combine the tags and store in the cache
            if self.tag_type == Sentence.tag_type.TOKEN:
                self.cache = self._combine_token_tags(obj)
            elif self.tag_type == Sentence.tag_type.SPAN:
                self.cache = self._combine_span_tags(obj)
            elif self.tag_type == Sentence.tag_type.STRUCT:
                self.cache = self._combine_structural_tags(obj)
            else:
                raise AttributeError
            return self.cache

    def _combine_token_tags(self, obj):
        """Concatenate the token tags of the constituent Sentences.
        """
        new_token_tags = [tag for sentence in obj.sentences
                             for tag in sentence.__dict__[self.tag_name]]
        return new_token_tags

    def _combine_span_tags(self, obj):
        """Combine the span tags of the constituent Sentences, adjusting
        indices as needed.
        """
        new_span_tags = {}
        offset = 0

        for sentence in obj.sentences:
            for span, label in sentence.__dict__[self.tag_name].items():
                # Update the span to account for previous sentences
                new_span = tuple(sp.array(span) + offset)
                new_span_tags[new_span] = label

            # Increase the offset to account for the previous sentence
            offset += sentence.length
        return new_span_tags

    def _combine_structural_tags(self, obj):
        """Create a new DependencyDag or DependencyTree that combines the
        nodes, edges and other metadata of the constituent Sentences,
        adjusting indices as needed.
        """
        offset = 0

        # Populate the members required for a DependencyDag or DependencyTree
        # instance
        nodes = []
        edges = []
        root_idxs = set()
        token_idxs = []
        aux_idxs = []
        max_depths = []

        # Note the class name of the structure from the first sentence
        classname = obj.sentences[0].__dict__[self.tag_name].__class__.__name__

        for sentence in obj.sentences:
            struct = sentence.__dict__[self.tag_name]
            if struct.__class__.__name__ != classname:
                print "ERROR: Can't combine inconsistent structures",
                print classname, "and", struct.__class__.__name__
                raise Exception

            # Copy over the structure's members after shifting their
            # indices by the current offset
            # TODO: keep all token nodes before all aux nodes in order to
            # be able to reuse all idxs from span tags in MultiSentences.
            # Indexing in MultiSentences is currently broken if auxiliary
            # nodes are included.
            nodes.extend([node.offset_idxs(offset)
                for node in struct.nodes])
            edges.extend([edge.offset_idxs(offset)
                for edge in struct.edges])
            token_idxs.extend([token_idx + offset
                for token_idx in struct.token_idxs])
            aux_idxs.extend([aux_idx + offset
                for aux_idx in struct.aux_idxs])

            if not classname.endswith('Graph'):
                root_idxs.update([root_idx + offset
                    for root_idx in struct.root_idxs])
                if classname.endswith('Tree'):
                    max_depths.append(struct.max_depth)

            # Update the offset
            offset += len(struct.nodes)

        # Create the final structure to encompass structures from all
        # Sentences in the MultiSentence
        # Note: using __new__ instead of regular instance creation to
        # avoid initialization of members that will be instantly overwritten
        cls_obj = getattr(structure, classname)
        multisentence_struct = cls_obj.__new__(cls_obj)
        multisentence_struct.nodes = nodes
        multisentence_struct.edges = edges
        multisentence_struct.token_idxs = token_idxs
        multisentence_struct.aux_idxs = aux_idxs

        if not classname.endswith('Graph'):
            multisentence_struct.root_idxs = root_idxs
            if classname.endswith('Tree'):
                multisentence_struct.max_depth = max(max_depths)

        # Nodes reference edges directly, so we need to reset the references
        # of each node to point to the new edges (with offset indices).
        multisentence_struct.reset_edge_refs()

        return multisentence_struct

    # TODO: replace fully ASAP
    def _combine_structural_tags_old(self, obj):
        """Create a new Tree that combines the nodes and roots of Trees
        of the constituent Sentences, adjusting indices as needed.
        """
        offset = 0

        # Populate the members required for a Tree class instance
        nodes = []
        root_idxs = []
        word_idxs = []
        aux_idxs = []
        max_depths = []

        for sentence in obj.sentences:
            tree = sentence.__dict__[self.tag_name]

            if offset == 0:
                # If the offset is zero, copy over the Tree's members directly
                nodes.extend(tree.nodes)
                root_idxs.extend(tree.root_idxs)
            else:
                # Copy over the Tree's members after shifting their indices by
                # the current offset
                nodes.extend([node.offset_indices(offset)
                                   for node in tree.nodes])
                word_idxs.extend([word_idx + offset
                                  for word_idx in tree.word_idxs])
                aux_idxs.extend([aux_idx + offset
                                  for aux_idx in tree.aux_idxs])
                root_idxs.extend([root_idx + offset
                                  for root_idx in tree.root_idxs])

            # Since we are attempting to avoid initialzing a new Tree,
            # we must also keep track of the auxiliary members (max_depth)
            # found in each sentence's Tree
            max_depths.append(tree.max_depth)

            # Update the offset
            offset += len(tree.nodes)

        # Create the final Tree to encompass Trees from all Sentences in the
        # MultiSentence
        # Note: using __new__ instead of regular instance creation to
        # avoid initialization of members that will be instantly overwritten
        multisentence_tree = structure.Tree.__new__(structure.Tree)
        multisentence_tree.nodes = nodes
        multisentence_tree.word_idxs = word_idxs
        multisentence_tree.aux_idxs = aux_idxs
        multisentence_tree.root_idxs = root_idxs
        multisentence_tree.max_depth = max(max_depths)

        return multisentence_tree
