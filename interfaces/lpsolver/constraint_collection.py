#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from constraint import Constraint
import numpy as np

try:
    from lpsolve.lpsolve55 import lpsolve
except ImportError:
    pass


class ConstraintCollection(object):
    """A collection of bounds and linear inequality constraints for variables
    of a particular type.
    """
    def __init__(self, var_type):
        """Set default values.
        """
        # The target variable type which the constraints will apply to
        self.var_type = var_type

        # The lower and upper bound of variables of this type
        self.lower_bound = None
        self.upper_bound = None

        # A flag indicating whether these variables must be integers
        self.is_integer = False

        # Bounds on the sum of variables of this type; useful for counting
        # the number of active indicator variables
        self.sum_lower_bound = None
        self.sum_upper_bound = None

        # A list of linear inequality constraints to be applied for each
        # variable of this type
        self.constraints = []

    def apply_constraints(self, lp, variables, only_bounds=False, relax=False):
        """Apply all constraints in this collection to all variables of the
        appropriate type. This is only used for LPSolve LPs.
        """
        num_vars = sum(len(var_list) for var_list in variables.itervalues())

        for var in variables[self.var_type]:
            # Apply constraints on individual variable values
            if self.upper_bound is not None:
                lpsolve('set_upbo', lp, var.idx() + 1, self.upper_bound)
            if self.lower_bound is not None:
                lpsolve('set_lowbo', lp, var.idx() + 1, self.lower_bound)
            if self.is_integer and not relax:
                lpsolve('set_int', lp, var.idx() + 1, 1)

            # Apply inequality constraints involving the links between
            # variables
            if not only_bounds:
                for constraint in self.constraints:
                    constraint.apply_to_program(lp, var, num_vars)

        # Apply constraints on the sum of all variables of this type if these
        # constraints were provided. Note that LPsolve internally maps <=, >=
        # and == to 1, 2 and 3 respectively.
        if self.sum_upper_bound is not None \
        and self.sum_upper_bound == self.sum_lower_bound:
            self.apply_sum_bounds(lp, variables, num_vars, 3,
                                  self.sum_upper_bound)
        else:
            if self.sum_lower_bound is not None:
                self.apply_sum_bounds(lp, variables, num_vars, 2,
                                      self.sum_lower_bound)
            if self.sum_upper_bound is not None:
                self.apply_sum_bounds(lp, variables, num_vars, 1,
                                      self.sum_upper_bound)

    def apply_sum_bounds(self, lp, variables, num_vars, ineq, rhs):
        """Apply limits on the total sum of the values of variables of this
        type. This is only for LPSolve LPs.
        """
        # Initialize an empty constraint vector
        row = np.zeros((num_vars))

        # Get indices of all variables of this type
        indices = [var.idx() for var in variables[self.var_type]]
        row[indices] = 1

        # Add the vectorized constraint to the LP
        lpsolve('add_constraint', lp, row.tolist(), ineq, rhs)

    def readable_bounds(self):
        """Return a readable form of the bounds for all variables of this type.
        """
        string = "[%(lb)s,%(ub)s]%(int)s" % \
                        {'lb': self.lower_bound
                            if self.lower_bound is not None else '',
                         'ub': self.upper_bound
                            if self.upper_bound is not None else '',
                         'int': ' INTEGER' if self.is_integer else ''}

        if self.sum_lower_bound is not None \
        or self.sum_upper_bound is not None:
            string += "\tSum: [%(lb)s,%(ub)s]" % \
                        {'lb': self.sum_lower_bound
                            if self.sum_lower_bound is not None else '',
                         'ub': self.sum_upper_bound
                            if self.sum_upper_bound is not None else ''}
        return string

    def readable_constraints(self, variables):
        """Return a readable form of the inequality constraints.
        """
        string = ''
        for constraint in self.constraints:
            string += constraint.readable() + '\n'

            for var in variables[self.var_type]:
                string_var = constraint.readable_with_var(var, only_lhs=True)
                if string_var != '':
                    string += string_var + '\n'
            string += '\n'
        return string

    def add_constraint(self, constraint_type, *args, **kwargs):
        """Add a constraint for this variable type to the collection.
        """
        if constraint_type in self.__dict__:
            # Set bounds or integer constraints (ignoring keyword arguments)
            setattr(self, constraint_type, args[0])
        else:
            # Set linear inequality constraints
            getattr(self, constraint_type)(*args, **kwargs)

###############################################################################
# Constraints

    def general(self, *args, **kwargs):
        """A basic constraint; expects all standard arguments for generating an
        inequality constraint.
        """
        self.constraints.append(Constraint(*args, **kwargs))

    def is_exactly(self, *args, **kwargs):
        """A constraint that forces the variable to take a specific value.
        Useful for a sanity check.
        """
        # v = N
        N = args[0]
        self.constraints.append(Constraint(['own_idx'],
                                           [1],
                                           '=', N,
                                           **kwargs))

    def has_exactly(self, *args, **kwargs):
        """A constraint that forces exactly N of the variable's (indicator)
        links to be active, regardless of its own value.
        """
        # Σ x = N
        N, link = args
        self.constraints.append(Constraint([link],
                                           [1],
                                           '=', N,
                                           **kwargs))

    def has_at_most(self, *args, **kwargs):
        """A constraint that forces at most N of the variable's (indicator)
        links to be active, regardless of its own value.
        """
        # Σ x <= N
        N, link = args
        self.constraints.append(Constraint([link],
                                           [1],
                                           '<=', N,
                                           **kwargs))

    def has_at_least(self, *args, **kwargs):
        """A constraint that forces at least N of the variable's (indicator)
        links to be active, regardless of its own value.
        """
        # Σ x >= N
        N, link = args
        self.constraints.append(Constraint([link],
                                           [1],
                                           '>=', N,
                                           **kwargs))

    def iff(self, *args, **kwargs):
        """A constraint that enforces that if the (indicator) variable is
        active, all of its (indicator) links must also be active (and vice
        versa).
        """
        # X*v - Σ x = 0
        link = args[0]
        self.constraints.append(Constraint(['own_idx', link],
                                           [link, -1],
                                           '=', 0,
                                           **kwargs))

    def iff_exactly(self, *args, **kwargs):
        """A constraint that enforces that if the (indicator) variable is
        active, exactly N of its (indicator) links must also be active (and
        vice versa).
        """
        # N*v - Σ x = 0
        N, link = args
        self.constraints.append(Constraint(['own_idx', link],
                                           [N, -1],
                                           '=', 0,
                                           **kwargs))

    def iff_multiple(self, *args, **kwargs):
        """A constraint that enforces that if the (indicator) variable is
        active, exactly N of its (indicator) links must also be active (and
        vice versa).
        """
        # N*v - Σ x - Σ y - Σ z = 0
        N = args[0]
        links = args[1:]
        self.constraints.append(Constraint(['own_idx'] + [l for l in links],
                                           [N] + [-1 for l in links],
                                           '=', 0,
                                           **kwargs))

    def implies(self, *args, **kwargs):
        """A constraint that enforces that, if the (indicator) variable is
        active, all of its (indicator) links must be active.
        """
        # X*v - Σ x <= 0
        link = args[0]
        self.constraints.append(Constraint(['own_idx', link],
                                           [link, -1],
                                           '<=', 0,
                                           **kwargs))

    def implies_at_least(self, *args, **kwargs):
        """A constraint that enforces that, if the (indicator) variable is
        active, at least N of its (indicator) links must be active.
        """
        # N*v - Σ x <= 0
        N, link = args
        self.constraints.append(Constraint(['own_idx', link],
                                           [N, -1],
                                           '<=', 0,
                                           **kwargs))

    def implied_by(self, *args, **kwargs):
        """A constraint that enforces that, if any (indicator) links are
        active, the (indicator) variable must also be active.
        """
        # X*v - Σ x >= 0
        link = args[0]
        self.constraints.append(Constraint(['own_idx', link],
                                           [link, -1],
                                           '>=', 0,
                                           **kwargs))

    def has_flow_over(self, *args, **kwargs):
        """A constraint that enforces single commodity flow over the variable
        and grounds whether flow is active or not based on one of its
        (indicator) links.
        """
        # v - Σ F*x <= 0
        link, max_flow = args
        self.constraints.append(Constraint(['own_idx', link],
                                           [1, -max_flow],
                                           '<=', 0,
                                           **kwargs))

        # Also set the maximum flow value as an upper bound for these variables
        self.upper_bound = max_flow

    def consumes_flow_between(self, *args, **kwargs):
        """A constraint that enforces flow consumption from connected flow
        variables, thereby ensuring their connectivity in a tree structure.
        """
        # Σ x - Σ y = 1
        incoming_link, outgoing_link = args
        self.constraints.append(Constraint([incoming_link, outgoing_link],
                                           [1, -1],
                                           '=', 1,
                                           **kwargs))

    def requires_flow_between(self, *args, **kwargs):
        """A constraint that enforces flow consumption from connected flow
        variables if the current (indicator) variable is active. This doesn't
        enforce a tree structure by itself without other constraints to
        activate this variable.
        """
        # Σ x - Σ y = v
        incoming_link, outgoing_link = args
        self.constraints.append(Constraint([incoming_link, outgoing_link,
                                            'own_idx'],
                                           [1, -1, -1],
                                           '=', 0,
                                           **kwargs))
