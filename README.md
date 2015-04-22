This repository contains the code used in my dissertation research on
sentence compression and fusion. The system implements supervised
structured prediction for text transformation in which the
inference approach relies on integer programming
algorithms to jointly produce output sentences characterized by
* a sequence of n-grams (bigrams or trigrams)
* an edge-factored dependency tree
* a SEMAFOR-style frame-semantic parse (compression only)
These models are described and evaluated in Chapter 3, the latter half
of Chapter 6 and and Chapter 7 of my dissertation.

Usage
-----
Honestly, it is unlikely that this codebase will be directly usable.
It has been extracted from a larger library without modification,
has not been tested outside the development environment and
ultimately suffers from all the usual pitfalls of research code
written under deadline pressure.
Instead, interested users are encouraged to use this repository
for reference or as a source of piecemeal solutions in reimplementation
efforts.

Nevertheless, if you wish to attempt to get this code running, here
is a likely-incomplete list of the known requirements:
* Python 2.6 or 2.7
* Ensure the distributed modules are on the $PYTHONPATH.
* Module dependencies
  * argparse (for Python 2.6)
  * nltk (with Wordnet and Framenet corpora)
  * psutil
  * pyutilib.Enum
  * simplejson
  * srilm_swig
  * stemming
* External software
  * Gurobi
  * LPsolve
  * SRILM
  * Stanford parser 2.0.4 or similar older version which produces projective trees
  * SEMAFOR
  * RASP
  * TagChunk
* Data
  * Dependency-converted Penn treebank for interfaces/treebank/depmodel.py (not necessary for default features)
  * Clarke & Lapata datasets for compression (contact me for dataset splits)
  * Pyramid evaluation data for fusion, available from NIST
* Update all paths in the code with appropriate paths to your installations
* Launch servers
  * LM servers through interfaces/srilm.py
  * Optionally, PTB servers through interfaces/treebank/depmodel.py
* Entry points to the code are transduction/compression.py and transduction/pyrfusion.py.
  * Run these with --help for command-line options.
  * Structural configurations are chosen by selecting features through transduction/featconfigs.py. The default options have simple names like 'word', 'ngram', 'dep' and are defined at the top of the file.
* Contact me if you want the model files or system outputs from my experiments.


Support
-------
This code is provided as-is and without any implicit or explicit assurance
of support. Bugs will not be addressed but will be listed in this README.
