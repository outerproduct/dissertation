#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import itertools
import numpy as np
from operator import itemgetter
from utils import sparse


class Features(object):
    """A virtual class for working with large sets of features.
    groups of features over transduction instances. Can be instantiated
    in order to incorporate more involved feature organization.
    """
    def __init__(self, feature_conf, standardize=False, **kwargs):
        """Initialize and store the full list of features.
        """
        # Retrieve full list of features
        self.features = self.get_feat_templates(feature_conf, **kwargs)

        # A list of indices for features whose weights should not be
        # modified
        self.fixed_feat_idxs = []

        # A list of feature categories, where a category is specified in
        # the feature name.
        self.categories = []
        self.cat_sizes = []
        self.feats_by_cat = {}
        self.sizes_by_cat = {}
        self.indis_by_cat = {}

        self.num_feats = 0
        for feature in self.features:
            # By convention, categories are determined by the first 'word'
            # in the feature name
            cat = feature[0].split('_')[0]
            size = self.get_template_size(feature)
            indi = self.is_indicator_template(feature)

            if cat not in self.categories:
                self.categories.append(cat)
                self.cat_sizes.append(0)
                self.feats_by_cat[cat] = []
                self.sizes_by_cat[cat] = []
                self.indis_by_cat[cat] = []
            self.cat_sizes[self.categories.index(cat)] += size
            self.feats_by_cat[cat].append(feature)
            self.sizes_by_cat[cat].append(size)
            self.indis_by_cat[cat].append(indi)

            if feature[0].endswith('_fixed'):
                self.fixed_feat_idxs.extend(
                        range(self.num_feats, self.num_feats + size))
            self.num_feats += size

        # Whether the features should be standardized to zero mean and
        # unit variance
        self.standardize = standardize
        if standardize:
            self.sum_values = np.zeros(self.num_feats)
            self.sum_squared_values = np.zeros(self.num_feats)
            self.num_vectors_seen = 0
            self.done_sampling = False


    def __len__(self):
        """Return the total number of features. This allows an instantiated
        object to replace a plain list of features.
        """
        return self.num_feats

    def __str__(self):
        """Return a string representation of the stored features.
        This allows an instantiated object to replace a plain list of features.
        """
        # We make no attempt here to expand indicator features. This should
        # be added if the dimensionality of this string is important.
        return '\n'.join(self.get_template_str(feat) for feat in self.features)

    def get_feat_vector(self, instance, word_idxs, given_cat, **kwargs):
        """Construct a sparse feature vector for word indices from a
        given category.
        """
        idx_offset = sum(self.cat_sizes[:self.categories.index(given_cat)])
#        if instance.idx == 0:
#            print "----------- Offset", idx_offset

        feat_vector_components = []
        for feat, feat_size, is_indicator in zip(self.feats_by_cat[given_cat],
                                                 self.sizes_by_cat[given_cat],
                                                 self.indis_by_cat[given_cat]):
            if is_indicator:
                # Vectorized feature function for efficiency, particularly
                # useful for indicator features
                feat_values = getattr(self, feat[0])(instance, word_idxs,
                                                     *feat[1:],
                                                     ext_offset=idx_offset,
                                                     **kwargs)
                # TODO: check that this is well-formed
                feat_vector_components.extend(feat_values)

#                if instance.idx == 0 and len(feat_values) > 0:
#                    for idx, feat_value in feat_values:
#                        print '[' + str(idx_offset) + '-' + \
#                                str(idx_offset+feat_size) + ']', \
#                                idx, feat_value, '  ', \
#                                word_idxs, \
#                                self.get_feat(feat, idx - idx_offset)
            else:
                # Normal feature function which returns a real valued number
                feat_value = getattr(self, feat[0])(instance, word_idxs,
                                                    *feat[1:], **kwargs)
                if feat_value != 0:
                    feat_vector_components.append((idx_offset, feat_value))

#                if instance.idx == 0:
#                    print str(idx_offset) + '-' + \
#                            str(idx_offset+feat_size), \
#                            word_idxs, feat[0], feat_value

            idx_offset += feat_size

        # Sanity check
        assert idx_offset == sum(
                self.cat_sizes[:self.categories.index(given_cat)+1])

        if self.standardize:
            if self.done_sampling:
                return self.standardize_feat_vector(
                        sparse.from_nonzero(feat_vector_components))
            else:
                # Get information to compute mean and std deviation for
                # later feature standardization
                for idx, feat_value in feat_vector_components:
                    self.sum_values[idx] += feat_value
                    self.sum_squared_values[idx] += feat_value * feat_value
                    self.num_vectors_seen += 1

        return sparse.from_nonzero(feat_vector_components)

    def standardize_feat_vector(self, feat_vector):
        """Standardize a sparse feature vector.
        """
        assert sparse.is_sparse(feat_vector)

        if not self.done_sampling:
            # This must be the first time the standardization was called.
            # We convert the sum vectors into a mean and std deviation.
            self.means = self.sum_values / self.num_vectors_seen
            squared_means = self.sum_squared_values / self.num_vectors_seen
            std_dev = np.sqrt(squared_means - np.power(self.means, 2))
            self.inv_std_devs = np.reciprocal(std_dev + 1e-7)

            # Ensure that this operation isn't repeated
            self.done_sampling = True
            del self.sum_values, self.sum_squared_values, self.num_vectors_seen
            print "Freezing feature standardization parameters"

        return sparse.from_list(
                sparse.multiply(
                    sparse.subtract(feat_vector, self.means),
                    self.inv_std_devs))

    def sanitize_weights(self, weights):
        """Add in 1.0 weight components for the fixed features, if any.
        """
        if len(self.fixed_feat_idxs) == 0:
            return weights
        else:
            # Avoid overwriting the existing weight vector, which may be
            # a numpy array
            sanitized_weights = list(weights)[:]
            for idx in self.fixed_feat_idxs:
                sanitized_weights[idx] = 1
            return sanitized_weights

    def sanitize_feat_vector(self, feat_vector, instance):
        """Ensure that the weights aren't updated for the fixed feature
        idxs by setting them to their corresponding gold values.
        """
        if len(self.fixed_feat_idxs) == 0:
            return feat_vector
        else:
            if not hasattr(instance, 'fixed_gold_feats'):
                fixed_gold_values = sparse.get_values(
                        instance.gold_feat_vector, self.fixed_feat_idxs)
                # For efficiency, save this with the instance since it
                # will never change. Note that zeros are also stored in
                # this case; they should be harmless for the representation.
                instance.fixed_gold_feats = sparse.from_nonzero(
                        zip(self.fixed_feat_idxs, fixed_gold_values))

            return sparse.update(feat_vector, instance.fixed_gold_feats)

    def print_with_values(self, *args, **kwargs):
        """Print the values associated with each feature template where values
        are provided as lists, optionally sparse.
        """
        value_lists = list(args)
        for v in range(len(value_lists)):
            if sparse.is_sparse(value_lists[v]):
                value_lists[v] = sparse.to_list(value_lists[v],
                                                self.num_feats)

        idx_offset = 0
        for cat in self.categories:
            for feat, feat_size in zip(self.feats_by_cat[cat],
                                       self.sizes_by_cat[cat]):
                value_sublists = [values[idx_offset:idx_offset + feat_size]
                        for values in value_lists]
                print '[' + str(idx_offset) + '-' + \
                        str(idx_offset + feat_size) + ']', \
                        self.get_template_str(feat, *value_sublists, **kwargs)
                idx_offset += feat_size

        assert idx_offset == len(values)

###############################################################################
# Utility functions

    @classmethod
    def get_feat_templates(cls, feature_conf, **kwargs):
        """Return a list of tuples representing feature function names and
        corresponding arguments that will be called explicitly.
        """
        feat_templates = []
        for feat_template in feature_conf:
            feat_templates.extend(getattr(cls, feat_template)(**kwargs))
        return feat_templates

    @classmethod
    def get_feat(cls, feat_template, idx):
        """Return the feature corresponding to an index into a template.
        """
        return cls.expand_template(feat_template)[idx]

    @classmethod
    def expand_template(cls, feat_template):
        """Expand the feature template and return a list of individual
        fully-parameterized features.
        """
        list_wrapped_template = []
        for field in feat_template:
            if isinstance(field, list) or \
                    isinstance(field, tuple):
                list_wrapped_template.append(field)
            elif isinstance(field, dict):
                list_wrapped_template.append(cls.to_list(field))
            else:
                list_wrapped_template.append([field])
        return list(itertools.product(*list_wrapped_template))

    @classmethod
    def is_indicator_template(cls, feat_template):
        """Return whether the feature is an indicator template, i.e., it
        will return a list of feature values instead of a single value.
        """
        for field in feat_template:
            if isinstance(field, list) or \
                    isinstance(field, tuple) or \
                    isinstance(field, dict):
                return True
        return False

    @classmethod
    def get_template_size(cls, feat_template):
        """Return the number of features that this feature template describes.
        """
        size = 1
        for field in feat_template:
            if isinstance(field, list) or \
                    isinstance(field, tuple) or \
                    isinstance(field, dict):
                size = size * len(field)
        return size

    @classmethod
    def get_template_str(cls, feature, *args, **kwargs):
        """Return a string representation of the given feature template,
        optionally abbreviating lists of indicators.
        """
        # Default kwarg for list cutoff
        list_cutoff = kwargs['list_cutoff'] if 'list_cutoff' in kwargs else 3

        value_lists = args
        feat_str = ''
        if len(value_lists) > 0:
            feat_str += '  '.join("%+.2f" % (sum(values),)
                                  for values in value_lists) + '  '
        feat_str += feature[0]
        for arg in feature[1:]:
            feat_str += ' '

            if not (isinstance(arg, list) or
                    isinstance(arg, tuple) or
                    isinstance(arg, dict)):
                feat_str += str(arg)
                continue

            if isinstance(arg, dict):
                # We sort indicator dictionaries by values here so that
                # feature strings will be identical when models are identical
                arg = [item[0]
                       for item in sorted(arg.iteritems(), key=itemgetter(1))]

            if len(arg) <= list_cutoff:
                feat_str += str(arg)
            else:
                feat_str += '[' + \
                        ', '.join(str(x) for x in arg[:list_cutoff]) + \
                        ' ... ' + str(len(arg)) + ' total]'
        return feat_str

    @classmethod
    def scale_feat_values(cls, feat_vals, coeff):
        """Scale the features by a given coefficient.
        """
        return sparse.scale(feat_vals, coeff)

    @classmethod
    def sum_feat_values(cls, feat_vals_list, average=False):
        """Sum or average the given feature values.
        """
        if len(feat_vals_list) == 0:
            # Keep the default coarseness setting (False) to be consistent
            # with the call to sparse.from_nonzero() in cls.get_feat_vector()
            return sparse.initialize()

        if average:
            return sparse.avg_all(feat_vals_list)
        else:
            return sparse.add_all(feat_vals_list)

    @classmethod
    def get_score(cls, feat_values, weights):
        """Compute a dot product for the given feature values with the
        given weights.
        """
        return sparse.dot(feat_values, weights)

    @classmethod
    def to_dict(cls, input_list):
        """Convert a large list to a dictionary for efficient lookups in
        feature functions.
        """
        large_list = []
        if isinstance(input_list, set):
            large_list = sorted(input_list)
        else:
            large_list = sorted(set(input_list))

        return dict((contents, idx)
                for idx, contents in enumerate(large_list))

    @classmethod
    def to_list(cls, input_dict):
        """Convert a dictionary representation of a list back to the large
        list.
        """
        large_list = [None for i in range(len(input_dict))]
        for key, val in input_dict.iteritems():
            assert large_list[val] is None
            large_list[val] = key
        return large_list
