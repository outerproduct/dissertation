#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement


class LearningRate(object):
    """A learning rate with standard update functionality.
    """
    def __init__(self, rate_updater='constant', eta_0=1.0, tau=2, alpha=0.5,
            multiplier=0.8, **kwargs):
        """Initialize various learning rate parameters.
        """
        # Active parameters
        self.k = 0
        self.eta = eta_0

        # Parameters for updating \eta
        self.update_fn = '_update_' + rate_updater
        self.eta_0 = eta_0
        self.tau = tau
        self.alpha = alpha
        self.multiplier = multiplier

    # TODO: Write a function to plot various learning rates against each other
    # for comparison
    def reset(self):
        """Reset the current iteration k to 0 and the learning rate \eta to the
        initial rate \eta_0.
        """
        self.k = 0
        self.eta = self.eta_0

    def update(self):
        """Update the learning rate using the default update function.
        """
        self.k += 1
        getattr(self, self.update_fn)()

    def value(self):
        """Return the current learning rate value.
        """
        return self.eta

    def _update_constant(self):
        """Keep the learning rate fixed.
        """
        pass

    def _update_standard(self):
        """Update the learning rate \eta_k for iteration k according to
        \eta_k = \eta_0 \frac{\tau}{\tau + k}
        Source: Finkel, Kleeman & Manning (2008) "Efficient, Feature-Based,
        Conditional Random Field Parsing".
        """
        self.eta = self.eta_0 * self.tau / (self.tau + self.k)

    def _update_multiplier(self):
        """Update the learning rate \eta_k for iteration k according to
        \eta_k = \eta_{k-1} * multiplier
               = \eta_0 * multiplier^k
        Source: MacCartney, Galley & Manning (2008) "A Phrase-based Alignment
        Model for Natural Language Inference".
        """
        self.eta *= self.multiplier

    def _update_exponent(self):
        """Update the learning rate \eta_k for iteration k according to
        \eta_k = (\tau + k)^(-\alpha)
        such that \sum_k^\inf \eta_k = \inf, and \sum_k^\inf \eta_k^2 < \inf;
        for example: \tau = 2 and 0.5 < \alpha <= 1.
        Source: Liang & Klein (2009) "Online EM for Unsupervised Models"
        """
        self.eta = (self.tau + self.k) ** (- self.alpha)
