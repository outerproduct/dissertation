#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import cPickle
import os.path
import re
import sys


class StoredModel(object):
    """A generic model that can be subclassed by a learner. Provides saving
    and restoring via pickling.
    """
    def __init__(self, name, restore=True, model_path='.', silent=False):
        """Record the name and save location for this model.
        """
        self.name = name
        self.savepath = ''.join((model_path, '/',
                                 self._sanitize_filename(name),
                                 '.model'))
        self.loaded = False

        # Try restoring from a saved model
        if restore and self.restore():
            self.loaded = True
            if not silent:
                sys.stderr.write("Restoring model \'" + self.name +
                                 "\' from " + self.savepath + "\n")
        else:
            if not silent:
                sys.stderr.write("Initializing new model \'" + self.name +
                                 "\' at " + self.savepath + "\n")

    def _sanitize_filename(self, filename):
        """Sanitize a string to produce a valid filename.
        """
        filename = re.sub(r"[^\w \+\-\,\.\/]", '', filename.lower())
        filename = re.sub(r" +", '_', filename)
        if filename.endswith('.model'):
            filename = filename[:-6]
        return filename

    def restore(self):
        """Restore a pickled model.
        """
        if not os.path.exists(self.savepath):
            return False

        with open(self.savepath) as f:
            try:
                other = cPickle.load(f)
            except EOFError:
                sys.stderr.write("EOFError when loading model \'" + self.name +
                                 "\' from " + self.savepath + "\n")
                return False
            self.__dict__.update(other.__dict__)
        return True

    def save(self):
        """Pickle this model object and store it in the given path.
        """
        if self.name.lower().startswith('debug'):
            sys.stderr.write("Didn't save temporary model \'" + \
                    self.name + "\'\n")
            return

        sys.stderr.write("Saving model \'" + self.name + "\' to " +
                        self.savepath + "\n")
        with open(self.savepath, 'wb') as f:
            cPickle.dump(self, f, 2)
            f.flush()
            os.fsync(f)
