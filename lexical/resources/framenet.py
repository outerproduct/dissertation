#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from nltk.corpus import framenet


###############################################################################

# Names of all frames in Framenet (1019 total)
frames = sorted(frame.name for frame in framenet.frames())


# Names of all possible FEs (1170 total)
fes = sorted(set(fe for frame in framenet.frames() for fe in frame.FE.keys()))


# Names of all possible frames and FEs (9633 total)
frame_fes = sorted([(frame.name, fe)
                        for frame in framenet.frames()
                        for fe in frame.FE.iterkeys()],
                   key=lambda x: x[0] + x[1])

###############################################################################

# Core types of FEs
coretypes = ['Core', 'Peripheral', 'Extra-Thematic']


# Names of all possible FEs and coretypes (1491 total)
fe_coretypes = sorted(set((fe, frame_element.coreType)
                            for frame in framenet.frames()
                            for fe, frame_element in frame.FE.iteritems()),
                        key=lambda x: x[0] + x[1])


# Names of all possible frames and FEs and coretypes (9633 total)
frame_fe_coretypes = sorted([(frame.name, fe, frame_element.coreType)
                                for frame in framenet.frames()
                                for fe, frame_element in frame.FE.iteritems()],
                            key=lambda x: x[0] + x[1] + x[2])

###############################################################################

# Names of core FEs that are specific to a frame (857 total)
core_fes = sorted(set(fe for frame in framenet.frames()
                    for fe, frame_element in frame.FE.iteritems()
                    if frame_element.coreType == 'Core'))


# Names of extra-thematic FEs that belong to larger frames (236 total)
# - intersected with core: 90
extrathematic_fes = sorted(set(fe for frame in framenet.frames()
                    for fe, frame_element in frame.FE.iteritems()
                    if frame_element.coreType == 'Extra-Thematic'))


# Names of peripheral FEs that aren't specific to a frame (349 total)
# - intersected with core: 157
# - intersected with extra-thematic: 102
# - intersected with both: 63
peripheral_fes = sorted(set(fe for frame in framenet.frames()
                    for fe, frame_element in frame.FE.iteritems()
                    if frame_element.coreType == 'Peripheral'))

###############################################################################

def get_frame_ancestors(frame_name, limit=None):
    """Return names of all ancestors of the given frame in the Framenet
    taxonomy up to a specified distance.
    """
    traversed = []
    successors = [relation.Parent.name
                  for relation in framenet.frame_relations(frame_name)
                  if relation.type.name == 'Inheritance'
                  and relation.Parent.name != frame_name]

    if limit is not None:
        limit -= 1
    if limit is None or limit >= 0:
        for successor in successors:
            traversed.append(successor)
            traversed.extend(get_frame_ancestors(successor, limit=limit))

    return traversed


def get_coretype(frame_name, fe):
    """Return the core type of the given FE from the set of string types
    {'Core', 'Peripheral', 'Extra-Thematic'}
    """
    return framenet.frame(frame_name).FE[fe].coreType
