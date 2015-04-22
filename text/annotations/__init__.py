#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

import imp
from utils import timer


def load_annotator(name):
    """Dynamically load an annotation module, instantiate the annotator
    and store a local instance of the resulting object.
    """
    # Load the module dynamically
    mod_name = name.lower()
    module_spec = imp.find_module('text/annotations/' + mod_name)
    if module_spec is None:
        print "ERROR: can't find annotator named", name
        raise ImportError
    try:
        mod = imp.load_module(mod_name, *module_spec)
    finally:
        # Close the open file handle in module_spec
        module_spec[0].close()

    # Instantiate and return object from the main annotator class
    cls_name = mod_name.capitalize()
    return getattr(mod, cls_name)()


def annotate(sents, *annotator_names):
    """Annotate one or more sentences with the given annotators.
    """
    if len(annotator_names) == 0:
        print "WARNING: no annotator specified"
        return
    if len(annotator_names) == 1 and \
            (isinstance(annotator_names[0], tuple) or
                isinstance(annotator_names[0], list)):
        # Annotators may be provided in a list or tuple
        annotator_names = annotator_names[0]

    if not isinstance(sents, tuple) and not isinstance(sents, list):
        # One or more sentences may be provided
        sents = [sents]

    for annotator_name in annotator_names:
        annotator = load_annotator(annotator_name)
        with timer.Timer():
            annotator.run_on_corpus(sents)
