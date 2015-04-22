#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

import structperceptron
import os
import sys


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "USAGE: ./show.py [full_model_path]"
        sys.exit()

    model_filepath = sys.argv[1]
    if not os.path.exists(model_filepath):
        print "ERROR: can't find model file", model_filepath
        sys.exit()

    i = model_filepath.rfind('/') + 1
    j = model_filepath.rfind('.')
    model_name = model_filepath[i:j]
    model_path = model_filepath[:i]
    m = structperceptron.StructPerceptron(model_name, None,
            model_path=model_path,
            restore=True, reload_check=False)
    m.print_model()
