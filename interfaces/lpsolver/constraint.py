#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import numpy as np
import sys

###############################################################################
solver_imported = False

try:
    from lpsolve.lpsolve55 import lpsolve
    solver_imported = True
except ImportError:
    pass

try:
    # Have to account for different versions of Python (default 2.7)
    if sys.version_info[:2] == (2,6):
        import gurobipy26 as gurobipy
    else:
        import gurobipy
    solver_imported = True
except ImportError:
    raise

if not solver_imported:
    print "No LPSolve or Gurobi bindings found"
    sys.exit()

###############################################################################

lpsolve_ineq_mapper = {'<=': 1,
                       '=<': 1,
                       '>=': 2,
                       '=>': 2,
                       '==': 3,
                       '=': 3}

gurobi_ineq_mapper = {'<=': gurobipy.GRB.LESS_EQUAL,
                      '=<': gurobipy.GRB.LESS_EQUAL,
                      '>=': gurobipy.GRB.GREATER_EQUAL,
                      '=>': gurobipy.GRB.GREATER_EQUAL,
                      '==': gurobipy.GRB.EQUAL,
                      '=': gurobipy.GRB.EQUAL}


class Constraint(object):
    """A generalized linear inequality constraint over the variables in linear
    programs.
    """
    # For memory efficiency and performance, Constraints make use of
    # __slots__. This means they do not have an internal __dict__, cannot be
    # pickled and do not support weak references, among other things.
    __slots__ = ['links', 'coeffs', 'lpsolve_ineq', 'gurobi_ineq',
                 'ineq_symbol', 'rhs', 'conditions', 'warnings']

    def __init__(self, links, coeffs, ineq_symbol, rhs, **kwargs):
        """Initialize the default attributes for constraints.
        """
        # Parallel lists of named links involved in the constraint and the
        # coefficients assigned to them
        if len(links) != len(coeffs):
            print "ERROR: lists of link names", links,
            print "and coefficients", coeffs,
            print "should have the same number of entries"
        self.links = links
        self.coeffs = coeffs

        # The inequality symbol in the constraint
        if ineq_symbol not in lpsolve_ineq_mapper:
            print "ERROR: unexpected inequality symbol", ineq_symbol
        self.ineq_symbol = ineq_symbol
        self.lpsolve_ineq = lpsolve_ineq_mapper[ineq_symbol]
        self.gurobi_ineq = gurobi_ineq_mapper[ineq_symbol]

        # The right hand side of the constraint
        self.rhs = rhs

        # A flag indicating whether warnings should be turned on or off
        # when applying this constraint
        self.warnings = True
        if 'warnings' in kwargs:
            self.warnings = kwargs['warnings']

    def apply_to_program(self, lp, var, num_vars):
        """Apply this constraint to a given variable in a LPsolve linear
        program.
        """
        # Initialize an empty constraint vector
        row = np.zeros((num_vars))

        for link, coeff in zip(self.links, self.coeffs):
            # For dynamic coefficients, retrieve the value of the coefficient
            # from the number of indices in the specified link name
            if isinstance(coeff, basestring):
                # A string coefficient is a sign to use the number of links
                # as the coefficient
                if self.check_link(coeff, var):
                    coeff = len(var.linked_idxs[coeff])
                else:
                    # Don't apply constraints with zero-valued coefficients
                    return

            # Assign the coefficient to the linked indices in the row with
            # array slicing
            if self.check_link(link, var):
                indices = var.linked_idxs[link]
                row[indices] = coeff
            else:
                # Don't apply constraints with missing terms
                return

        # Add the vectorized constraint to the LP
        lpsolve('add_constraint', lp, row.tolist(), self.lpsolve_ineq,
                self.rhs)

    def apply_to_gmodel(self, lp, var, all_gvars):
        """Apply this constraint to a given variable in a Gurobi model.
        """
        linked_gvars = []
        gvar_coeffs = []
        for link, coeff in zip(self.links, self.coeffs):
            # For dynamic coefficients, retrieve the value of the coefficient
            # from the number of indices in the specified link name
            if isinstance(coeff, basestring):
                # A string coefficient is a sign to use the number of links
                # as the coefficient
                if self.check_link(coeff, var):
                    coeff = len(var.linked_idxs[coeff])
                else:
                    # Don't apply constraints with zero-valued coefficients
                    return

            # Assign the coefficient to all linked Gurobi variables
            if self.check_link(link, var):
                for idx in var.linked_idxs[link]:
                    linked_gvars.append(all_gvars[idx])
                    gvar_coeffs.append(coeff)
            else:
                # Don't apply constraints with missing terms
                return

        # Create a linear expression for the LHS and construct the constraint
        #expr = gurobipy.LinExpr(coeffs=gvar_coeffs, vars=linked_gvars)
        expr = gurobipy.LinExpr(gvar_coeffs, linked_gvars)
        lp.addConstr(expr, self.gurobi_ineq, self.rhs)

    def check_link(self, link, var):
        """Return True if the given link is present in the variable and has at
        least one index, otherwise return false and produce a warning.
        """
        if link in var.linked_idxs and len(var.linked_idxs[link]) > 0:
            return True

        if self.warnings:
            print "\nWARNING: Unexpected constraint over", var.type, "variable"
            print self.readable()
            print self.readable_with_var(var)

            if link not in var.linked_idxs:
                print link, "not found"
            else:
                print link, "=", var.linked_idxs[link]
        return False

    def readable(self):
        """Return a readable form of a general constraint.
        """
        # Get LHS terms from each link group
        lhs_terms = []
        for link_name, coeff in zip(self.links, self.coeffs):

            # Convert coefficients to strings
            coeff_str = ''
            if isinstance(coeff, basestring):
                coeff_str = "#(%(coeff)s)" % {'coeff': coeff}
            else:
                coeff_str = "%(coeff) .4g" % {'coeff': coeff}

            term = "%(coeff)s x %(name)s" % \
                    {'coeff': coeff_str,
                     'name': link_name}
            lhs_terms.append(term)

        # Return the LHS term, (in)equality symbol and RHS for the constraint
        return "%(lhs)s %(ineq)s %(rhs)g" % \
                    {'lhs': ', '.join(lhs_terms),
                     'ineq': self.ineq_symbol,
                     'rhs': self.rhs}

    def readable_with_var(self, var, only_lhs=False):
        """Return a readable form of a linear inequality constraint applied to
        a specific variable.
        """
        # Get LHS terms from each link group
        lhs_terms = []
        for link_name, coeff in zip(self.links, self.coeffs):

            # Convert coefficients to strings
            coeff_str = ''
            has_coeff = True
            if isinstance(coeff, basestring):
                if coeff not in var.raw_linked_idxs:
                    has_coeff = False
                else:
                    coeff_str = str(len(var.raw_linked_idxs[coeff]))
            else:
                coeff_str = "%(coeff) .4g" % {'coeff': coeff}

            if has_coeff and link_name in var.linked_type:
                term = "%(coeff)s x %(type)s %(idxs)s" % \
                        {'coeff': coeff_str,
                         'idxs': var.raw_linked_idxs[link_name],
                         'type': var.linked_type[link_name].lower()}
                lhs_terms.append(term)

        if only_lhs:
            return ", ".join(lhs_terms)
        else:
            return "%(lhs)s %(ineq)s %(rhs)g" % \
                    {'lhs': ', '.join(lhs_terms),
                     'ineq': self.ineq_symbol,
                     'rhs': self.rhs}
