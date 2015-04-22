#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import numpy as np
from variable import Variable
from constraint_collection import ConstraintCollection
import sys

###############################################################################
solver_imported = False

try:
    from lpsolve.lpsolve55 import lpsolve, IMPORTANT
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
    pass

if not solver_imported:
    print "No LPSolve or Gurobi bindings found"
    sys.exit()

###############################################################################


class LinearProgram:
    """A mixed integer linear program which can be solved via LPsolve or
    Gurobi.
    """
    def __init__(self, maximization=True):
        """Initialize the linear program.
        """
        self.is_max = maximization

        # Initialize variables
        self.variables = {}
        self.num_vars = 0
        self.grounding_mapper = {}

        # Initialize collections of type-specific constraints
        self.constraint_collections = {}

        # Initialize LP-specific attributes
        self.obj_value = None

        # Initialize a place to store built LPs for repeated decoding
        self.saved_lps = {}

        # Whether the last iteration of the program was an LP-relaxation
        self.was_relaxed = False

#    def __del__(self):
#        """Unload the linear program.
#        """
#        if self.lp is not None:
#            lpsolve('delete_lp', self.lp)

    def create_boolean_variable(self, var_type, **kwargs):
        """Create a new integer variable and add constraints setting its lower
        and upper bounds to 0 and 1 respectively.
        """
        new_var = self.create_integer_variable(var_type, **kwargs)
        self.add_constraint(var_type, 'lower_bound', 0)
        self.add_constraint(var_type, 'upper_bound', 1)
        return new_var

    def create_integer_variable(self, var_type, **kwargs):
        """Create a new variable and add an integer constraint.
        """
        new_var = self.create_variable(var_type, **kwargs)
        self.add_constraint(var_type, 'is_integer', True)
        return new_var

    def create_variable(self, var_type, **kwargs):
        """Create a new variable of the specified type and add it to the linear
        program.
        """
        new_var = Variable(var_type, **kwargs)
        self.add_existing_variable(new_var)

        # Return a reference to the new variable so that the calling function
        # can add links and refer to its index
        return new_var

    def add_existing_variable(self, new_var):
        """Add an existing variable of the specified type to the linear program
        (useful for subclassed variables).
        """
        self.num_vars += 1

        if new_var.type not in self.variables:
            new_var.set_idx(0)
            self.variables[new_var.type] = [new_var]
            self.grounding_mapper[new_var.type] = {}
        else:
            new_var.set_idx(len(self.variables[new_var.type]))
            self.variables[new_var.type].append(new_var)

        # Note the variables grounding attributes in a mapper so that it can
        # be retrieved easily
        string = self.get_grounding_string(new_var.grounding)
        if string in self.grounding_mapper[new_var.type]:
            print "ERROR: duplication in variable grounding for", new_var.type,
            print "with grounding:", string
            raise Exception
        self.grounding_mapper[new_var.type][string] = new_var.raw_idx()

    def update_variable_coeff(self, new_coeff, var_type, **kwargs):
        """Update a variable's coefficient in the objective function. This
        can be used to modify and resolve LPs without rebuilding them (set
        rebuild=False in solve()).
        """
        var = self.retrieve_variable(var_type, **kwargs)
        var.set_coeff(new_coeff)

        # Also update the saved LPs
        if 'lpsolve' in self.saved_lps:
            lpsolve('set_obj', self.saved_lps['lpsolve'], var.idx() + 1,
                    new_coeff)
        if 'gurobi' in self.saved_lps:
            self.saved_lps['gurobi'][1][var.idx()].setAttr('Obj',
                    float(new_coeff))

    def get_grounding_string(self, grounding):
        """Return a stringified version of a variable's grounding dictionary.
        """
        return ";".join(("%s=%s" % (key, grounding[key])
                         for key in sorted(grounding.iterkeys())
                         if not key.startswith('_') ))

    def retrieve_variable(self, var_type, **kwargs):
        """Return the variable of the specified type whose grounding matches
        the given attributes.
        """
        string = self.get_grounding_string(kwargs)
        raw_var_idx = self.grounding_mapper[var_type][string]

        return self.variables[var_type][raw_var_idx]

    def retrieve_all_variables(self, var_type):
        """Return all variables of the specific types.
        """
        return self.variables[var_type]

    def retrieve_links(self, var, link_name):
        """Return a list of variables which are specified by the link name for
        the given variable.
        """
        if link_name not in var.raw_linked_idxs:
            return []

        var_type = var.linked_type[link_name]
        raw_idxs = var.raw_linked_idxs[link_name]
        return [self.variables[var_type][raw_idx] for raw_idx in raw_idxs]

    def initialize_offsets(self):
        """Initialize offsets for each type of variable so that all variables
        can be uniquely indexed in a giant vector (for LPsolve).
        """
        offsets = {}
        current_offset = 0
        for var_type in sorted(self.variables.iterkeys()):
            var_list = self.variables[var_type]
            offsets[var_type] = current_offset
            current_offset += len(var_list)

        for var_type, var_list in self.variables.iteritems():
            for var in var_list:
                var.apply_offsets(offsets)

    def add_constraint(self, var_type, constraint_type, *args, **kwargs):
        """Create a new constraint.
        """
        if var_type not in self.variables:
            print "\nWARNING: creating constraint for unknown variable type:",
            print var_type

        # Initialize a constraint collection for this variable unless one is
        # already present
        if var_type not in self.constraint_collections:
            self.constraint_collections[var_type] = ConstraintCollection(
                                                    var_type)

        self.constraint_collections[var_type].add_constraint(constraint_type,
                                                             *args, **kwargs)

    def display_variables(self):
        """Print the current variables and their grounding.
        """
        for var_type, var_list in self.variables.iteritems():
            num_vars = len(var_list)
            avg_coeff = sum(var.coeff for var in var_list)/len(var_list)
            print "\n%(type)s x %(num)d with average coeff: %(coeff) .2g" % \
                                {'type': var_type,
                                 'num': num_vars,
                                 'coeff': avg_coeff}

            for var in var_list:
                print var.type,
                sorted_keys = sorted(var.grounding.iterkeys())
                for key in sorted_keys:
                    print ": ".join((str(key), str(var.grounding[key]))),
                print

    def display_constraints(self):
        """Print the current configuration of the linear program.
        """
        for var_type, var_list in self.variables.iteritems():
            num_vars = len(var_list)
            avg_coeff = sum(var.coeff for var in var_list)/len(var_list)
            print "\n%(type)s x %(num)d with average coeff: %(coeff) .2g" % \
                                {'type': var_type,
                                 'num': num_vars,
                                 'coeff': avg_coeff}

            if var_type in self.constraint_collections:
                collection = self.constraint_collections[var_type]
                print collection.readable_bounds()
                print collection.readable_constraints(self.variables)

            # This dumps a lot of information; impossible to read for large LPs
#            for var in var_list:
#                readable_var_constraints = var.readable_constraints()
#                if readable_var_constraints != '':
#                    print readable_var_constraints

    def solve(self, solver='gurobi', save=True, rebuild=True, relax=False,
            **kwargs):
        """Solve the LP using either LPsolve or Gurobi.
        """
        if solver not in ('lpsolve', 'gurobi'):
            print "ERROR: unfamiliar solver", solver
            raise Exception

        # If the relaxation flag has changed, the LP must be rebuilt
        if self.was_relaxed != relax:
            self.was_relaxed = relax
            rebuild = True

        # Recover previously built LP if any
        if rebuild or solver not in self.saved_lps:
            lp_conf = getattr(self, 'build_' + solver)(relax=relax, **kwargs)
        else:
            lp_conf = self.saved_lps[solver]

        getattr(self, 'optimize_' + solver)(*lp_conf, **kwargs)

        if save or solver in self.saved_lps:
            self.saved_lps[solver] = lp_conf
        else:
            # To avoid dangling pointers in the LPSolve interface
            if solver == 'lpsolve':
                lpsolve('delete_lp', lp_conf[0])

    def optimize_lpsolve(self, lp, timeout=None, break_at_first=False,
            debug=False, **kwargs):
        """Convert the stored variables and constraints into an LPsolve linear
        program and optimize it.
        """
        # Update solving parameters
        if timeout is not None:
            lpsolve('set_timeout', lp, timeout)
        if break_at_first:
            # TODO test this
            lpsolve('break_at_first', lp, True)

        # Ensure that the LP produces a solution
        result = lpsolve('solve', lp)
        if result not in (0, 1, 11, 12):
            print "\nERROR: Failed to find a solution",
            print "(result = %d)" % result
            if debug:
                #self.display_constraints()
                self.debug_lpsolve()
        else:
            if result in (1, 11, 12):
                print "\nWARNING: Suboptimal solution",
                print "(result = %d)" % result

            # Retrieve the solution
            self.obj_value, self.values, duals, ret = lpsolve(
                    'get_solution', lp)

    def optimize_gurobi(self, lp, gvars, timeout=None, singlecore=False,
            **kwargs):
        """Convert the stored variables and constraints into a Gurobi model
        and solve it. This should probably be rewritten to fit Gurobi better.
        """
        # Set the time limit and optimize the model
        if timeout is not None:
            lp.setParam('TimeLimit', int(timeout))
        if singlecore:
            lp.setParam('Threads', 1)
        lp.optimize()

        # Check status of optimization and extract solution if present
        status = lp.getAttr('Status')
        if status not in (gurobipy.GRB.OPTIMAL, gurobipy.GRB.SUBOPTIMAL):
            print "\nERROR: Failed to find a solution",
            print "(status = %d)" % status
            return
        else:
            if status == gurobipy.GRB.SUBOPTIMAL:
                print "\nWARNING: Suboptimal solution",
                print "(status = %d)" % status

            # Save the solution
            self.obj_value = lp.getAttr('ObjVal')
            self.values = [gvar.getAttr('X') for gvar in gvars]

    def build_lpsolve(self, timeout=None, break_at_first=False,
            relax=False, **kwargs):
        """Build an LPsolve linear program object from the stored variables
        and constraints.
        """
        self.initialize_offsets()

        # Make the LPsolve linear program object
        lp = lpsolve('make_lp', 0, self.num_vars)
        lpsolve('set_verbose', lp, IMPORTANT)
        if self.is_max:
            lpsolve('set_maxim', lp)
        else:
            lpsolve('set_minim', lp)
        if timeout is not None:
            lpsolve('set_timeout', lp, timeout)
        if break_at_first:
            # TODO test this
            lpsolve('break_at_first', lp, True)

        # For all variables
        for var_type, var_list in self.variables.iteritems():

            # Apply type-specific constraints
            if var_type in self.constraint_collections:
                collection = self.constraint_collections[var_type]
                collection.apply_constraints(lp, self.variables,
                                             relax=relax)

            for var in var_list:
                # Set the value of each variable in the objective function
                lpsolve('set_obj', lp, var.idx() + 1, var.coeff)

                # Apply variable-specific constraints
                for constraint in var.constraints:
                    constraint.apply_to_program(lp, var, self.num_vars)

        return lp,  # tuple expected

    def build_gurobi(self, timeout=None, singlecore=False, relax=False,
            **kwargs):
        """Build a Gurobi linear program object from the stored variables
        and constraints.
        """
        self.initialize_offsets()

        # Initialize the model
        lp = gurobipy.Model(name='GurobiModel')
        if self.is_max:
            lp.setAttr('ModelSense', -1)
        else:
            lp.setAttr('ModelSense', +1)

        # Set some parameters
        lp.setParam('OutputFlag', 0) # Don't log everything to STDOUT
        lp.setParam('LogFile', '')   # Don't create a logfile
#        lp.setParam('MIPFocus', 1)   # Focus on good feasible solutions
#        lp.setParam('MIPGap', 1e-2)  # Looser relative optimality gap (1e-4)
#        lp.setParam('Presolve', 2)   # Aggressive presolve
#        lp.setParam('DualReductions', 0) # XXX Diagnose INF_OR_UNBD behavior
        if timeout is not None:
            lp.setParam('TimeLimit', int(timeout))
#            lp.setParam('ImproveStartTime',  # Stop proving optimality and
#                        min(int(timeout/4),  # search for feasible solutions
#                            180))
        if singlecore:
            lp.setParam('Threads', 1)

        # For all variables
        gvars = [None] * self.num_vars
        for var_type, var_list in self.variables.iteritems():

            # Apply type-specific constraints
            vartype_attributes = {}
            sum_lower_bound = None
            sum_upper_bound = None
            if var_type in self.constraint_collections:
                collection = self.constraint_collections[var_type]
                sum_lower_bound = collection.sum_lower_bound
                sum_upper_bound = collection.sum_upper_bound

                # TODO: these attributes can just be added via gvar.setAttr()
                if collection.is_integer:
                    if collection.lower_bound == 0.0 and \
                            collection.upper_bound == 1.0:
                        # Boolean variable
                        vartype_attributes['vtype'] = gurobipy.GRB.BINARY \
                                if not relax else gurobipy.GRB.CONTINUOUS
                        vartype_attributes['lb'] = 0.0
                        vartype_attributes['ub'] = 1.0
                    else:
                        # Non-boolean integer variable
                        vartype_attributes['vtype'] = gurobipy.GRB.INTEGER \
                                if not relax else gurobipy.GRB.CONTINUOUS
                        if collection.lower_bound is not None:
                            vartype_attributes['lb'] = collection.lower_bound
                        if collection.upper_bound is not None:
                            vartype_attributes['ub'] = collection.upper_bound
                else:
                    # Real-valued variable
                    vartype_attributes['vtype'] = gurobipy.GRB.CONTINUOUS
                    if collection.lower_bound is not None:
                        vartype_attributes['lb'] = collection.lower_bound
                    if collection.upper_bound is not None:
                        vartype_attributes['ub'] = collection.upper_bound

            # Add all variables of this type to the model
            vartype_gvars = []
            for var in var_list:
                var_attributes = {'obj': var.coeff}
                var_attributes.update(vartype_attributes)

                gvar = lp.addVar(**var_attributes)
                gvars[var.idx()] = gvar
                vartype_gvars.append(gvar)

            # Finally, if any constraint was defined only on all variables of
            # a specific type, address it here
            if sum_lower_bound is not None or sum_upper_bound is not None:
                lp.update()
                coeffs = [1 for i in range(len(vartype_gvars))]
                expr = gurobipy.LinExpr(coeffs, vartype_gvars)

                if sum_lower_bound is not None and sum_upper_bound is not None:
                    # This should adequately capture equality constraints
                    lp.addRange(expr, sum_lower_bound, sum_upper_bound)
                else:
                    if sum_lower_bound is not None:
                        lp.addConstr(expr, gurobipy.GRB.GREATER_EQUAL,
                                     sum_lower_bound)
                    if sum_upper_bound is not None:
                        lp.addConstr(expr, gurobipy.GRB.LESS_EQUAL,
                                     sum_upper_bound)

        # Force a model update to make the variables usable
        lp.update()

#        for i, gvar in enumerate(gvars):
#            if gvar is None:
#                print "Found None variable at position", i

        # Now that all the Gurobi variables are generated, walk through all
        # variables again and apply cross-variable constraints
        for var_type, var_list in self.variables.iteritems():

            # Collect vartype-level constraints
            vartype_constraints = []
            if var_type in self.constraint_collections:
                vartype_constraints.extend(
                        self.constraint_collections[var_type].constraints)

            for var in var_list:
                # Apply vartype-level constraints
                for constraint in vartype_constraints:
                    constraint.apply_to_gmodel(lp, var, gvars)

                # Apply variable-specific constraints
                for constraint in var.constraints:
                    constraint.apply_to_gmodel(lp, var, gvars)

        # Force another model update and return the program
        lp.update()

# Not sure why this doesn't work
#        if relax:
#            return lp.relax(), gvars
#        else:
        return lp, gvars

    def debug_lpsolve(self):
        """Run the LP repeatedly with all but one constraints and check if a
        solution is found, thereby identifying a single constraint that may be
        preventing the LP from working successfully.
        """
        print "Potential problematic constraints:"

        # First, collect all constraints in the program
        constraint_tuples = set()
        for var_type, collection in self.constraint_collections.iteritems():
            for constraint in collection.constraints:
                constraint_tuples.add((var_type, constraint.readable()))

        for var_type, var_list in self.variables.iteritems():
            for var in var_list:
                for constraint in var.constraints:
                    constraint_tuples.add((var_type, constraint.readable()))

        # Now run the program repeatedly, each time omitting a single type of
        # constraint
        problem_constraints = []
        for constraint_tuple in constraint_tuples:
            self.initialize_offsets()

            # Make the LPsolve linear program object
            lp = lpsolve('make_lp', 0, self.num_vars)
            lpsolve('set_verbose', lp, IMPORTANT)
            if self.is_max:
                lpsolve('set_maxim', lp)
            else:
                lpsolve('set_minim', lp)

            # For all variables
            for var_type, var_list in self.variables.iteritems():

                # Apply type-specific constraints
                if var_type in self.constraint_collections:
                    collection = self.constraint_collections[var_type]
                    collection.apply_constraints(lp, self.variables,
                            only_bounds=True)

                    for constraint in collection.constraints:
                        if constraint_tuple == (var_type,
                                constraint.readable()):
                            continue
                        for var in var_list:
                            constraint.apply_to_program(lp, var, self.num_vars)

                for var in var_list:
                    # Set the value of each variable in the objective function
                    lpsolve('set_obj', lp, var.idx() + 1, var.coeff)

                    # Apply variable-specific constraints
                    for constraint in var.constraints:
                        if constraint_tuple != (var_type,
                                constraint.readable()):
                            constraint.apply_to_program(lp, var, self.num_vars)

            # Solve the LP using LPsolve and check for error codes
            result = lpsolve('solve', lp)
            if result in (0, 1, 11, 12):
                print constraint_tuple[0], constraint_tuple[1]
                problem_constraints.append(constraint_tuple)

    def get_value(self, var, ndigits=2):
        """Return the value of the variable in the current solution.
        """
        return round(self.values[var.idx()], ndigits)

    def get_all_values(self, var_type, ndigits=2):
        """Return the value of all variables of a particular type in the
        current solution. Since the offset (to raw indices) for variables of
        a particular type should be the same, and the variables are stored
        ordered by raw index, we can avoid looking up the value of each
        variable separately.
        """
        all_vars = self.variables[var_type]
        values = np.round(self.values[all_vars[0].idx():all_vars[-1].idx()+1],
                        decimals=ndigits)
        assert len(all_vars) == len(values)
        return values

    def is_active(self, var):
        """Return whether the variable is active or not in the current
        solution. An active variable is defined as being one whose value
        (rounded to some reasonable number of significant digits) is not
        zero.
        """
        return self.get_value(var) != 0

    def retrieve_active_vars(self, var_type):
        """Retrieve a list of all active variables of a particular type.
        """
        active_vars = []
        # Set the values for each variable
        for var in self.variables[var_type]:
            if self.is_active(var):
                active_vars.append(var)
        return active_vars

    def retrieve_active_chain(self, type_to_link, start_var,
            supplementary_links=None):
        """Retrieve a chain of connected (indicator) variables that are active
        (non-zero) in the LP solution by supplying a mapping between required
        variable types and the links that interconnect them. An additional
        mapping can be supplied to optionally capture connected variables (if
        they exist) that are not already in the chain.
        """
        # An ordered list of active variables from the most recent solution to
        # the LP
        chain = []

        next_vars = [start_var]
        while len(next_vars) == 1:
            current_var = next_vars[0]

            if current_var in chain:
                print "WARNING: found loop in chain"
                idx = chain.index(current_var)
                for i in range(idx, len(chain)):
                    print chain[i].type, chain[i].readable_grounding(),
                    print '-->', type_to_link[chain[i].type]
                print current_var.type, current_var.readable_grounding(),
                return None

            # Add this variable to the chain
            chain.append(current_var)

            # Before moving on, we can optionally add variables connected to
            # the current variable but not in the chain. Note that we don't
            # insist on the presence of these variables but currently only
            # permit one.
            if supplementary_links is not None:
                if current_var.type in supplementary_links:
                    link_names = supplementary_links[current_var.type]
                    for link_name in link_names:
                        link_type = current_var.linked_type[link_name]
                        supplementary_vars = []
                        for link_idx in current_var.raw_linked_idxs[link_name]:
                            link_var = self.variables[link_type][link_idx]
                            if self.is_active(link_var):
                                supplementary_vars.append(link_var)

                        if len(supplementary_vars) > 1:
                            print "WARNING: dropping multiple active",
                            print "supplementary variables from chain"
                        if len(supplementary_vars) == 1 and \
                                supplementary_vars[0] not in chain:
                            chain.append(supplementary_vars[0])

            # Get the type of the next variable
            link_name = type_to_link[current_var.type]
            if link_name not in current_var.linked_type:
                # Assume that this must be the end of the chain
                return chain
            link_type = current_var.linked_type[link_name]
            if link_type not in type_to_link:
                print "ERROR: link name", link_name, "is of type", link_type,
                print "which is not listed in", type_to_link
                raise Exception

            # Collect a list of active variables that can be next in the chain
            next_vars = []
            for link_idx in current_var.raw_linked_idxs[link_name]:
                link_var = self.variables[link_type][link_idx]
                if self.is_active(link_var):
                    next_vars.append(link_var)

            if len(next_vars) > 1:
                print "ERROR: more than one", link_name, "link from",
                print current_var.type, "variable found active"
                return None
            elif len(next_vars) == 0:
                print "WARNING: terminating at", current_var.type, "variable",
                print "as no", link_name, "links found active"
                return chain

        return chain
