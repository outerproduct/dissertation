#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import numpy as np
import onlinelearner
import sys
from utils import sparse


class StructPerceptron(onlinelearner.OnlineLearner):
    """An structured perceptron with parameter averaging.
    """
    def decode_instance(self, instance, weights, idx, learning_rate=None,
            num_instances=None, epoch=None, display_score=True, **kwargs):
        """Runs decoding for an instance. Returns a vector representing
        the difference between the gold and the current feature vectors
        for this instance.
        """
        feat_vector = instance.decode(weights, self.features, **kwargs)
#        self.print_feats(instance, feat_vector)

        if epoch is not None:
            sys.stdout.write("[Epoch " + str(epoch) + "] ")
        if num_instances is not None:
            sys.stdout.write("Decoding " + str(num_instances) + " instances: ")
        else:
            sys.stdout.write("Decoding instance: ")
        sys.stdout.write(str(idx+1))
        if display_score:
            if feat_vector is None:
                sys.stdout.write(" [FAILED]")
                if hasattr(instance, 'score'):
                    delattr(instance, 'score')
                sys.stdout.write(' '*10 + '\n')
            else:
                old_score = instance.score \
                        if hasattr(instance, 'score') else 0
                new_score = sparse.dot(feat_vector, weights) \
                        if sparse.is_sparse(feat_vector) \
                        else [f * w for f, w in zip(feat_vector, weights)]
                instance.score = new_score
                sys.stdout.write(" [%+.3g]" % (new_score - old_score,))
                sys.stdout.write(' '*10 + '\r')

        # Permit the learning rate to be supplied for train_slave() since
        # slaves are not synchronized with the master.
        if learning_rate is None:
            learning_rate = self.learning_rate.value()

        if feat_vector is None:
            return None
        elif sparse.is_sparse(feat_vector):
            return sparse.scale(sparse.subtract(instance.gold_feat_vector,
                                                feat_vector),
                                learning_rate)
        else:
            return (np.array(instance.gold_feat_vector) - feat_vector) * \
                    learning_rate

    def update_weights(self, weight_update):
        """Update the weights using the result of decoding.
        """
        if weight_update is not None:
            self.current_weights = sparse.add(self.current_weights,
                                              weight_update)
            self.sum_weights += self.current_weights
            self.num_updates_seen += 1
        else:
            print "Skipping weight update"

    def finish_epoch(self, display_weights=False, tuning_fn=None,
            tuning_params=None, interruptable=False, debug_idxs=None,
            timeout=None, **kwargs):
        """Update final weights and save the model.
        """
        if self.num_updates_seen > 0:
            # Track running average over all intermediate weight vectors to
            # monitor convergence
            self.final_weights = self.sum_weights / self.num_updates_seen
            if display_weights:
                self.print_weights()
        else:
            print "WARNING: No weight updates seen in epoch",
            print self.current_epoch

        self.current_epoch += 1
        self.learning_rate.update()

        if self.num_updates_seen > 0:
            if tuning_fn is not None:
                # Call some external function to help with tuning
                # e.g., evaluation on a dev set
                tuning_params['timeout'] = timeout
                tuning_params['relax'] = False
                tuning_fn(self, **tuning_params)

            # Permit training to be aborted without saving the model
            if interruptable:
                prompt = None
                while prompt.lower() not in ('y', 'n'):
                    prompt = raw_input(
                            "Finished iteration. Save and continue (y/n)? ")
                if prompt.lower() == 'n':
                    return True

        # Save current parameters for later resumption of training
        if debug_idxs is None:
            self.save()
        return False  # no interrupt
