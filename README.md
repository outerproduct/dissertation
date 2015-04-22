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
1. argparse (for Python 2.6)
2. nltk (with Wordnet and Framenet corpora)
3. psutil
4. pyutilib.Enum
5. simplejson
6. srilm_swig
7. stemming
* External software
1. Gurobi
2. LPsolve
3. SRILM
4. Stanford parser 2.0.4 or similar older version which produces projective trees
5. SEMAFOR
6. RASP
7. TagChunk
* Data
1. Dependency representation of Penn treebank for interfaces/treebank/depmodel.py (not necessary)
2. Clarke & Lapata datasets for compression (contact me for dataset splits)
3. Pyramid evaluation data for fusion, available from NIST
* Update all paths in the code with appropriate paths to your installations
* Launch LM servers through interfaces/srilm.py and optionally PTB servers through interfaces/treebank/depmodel.py
* Entry points to the code are transduction/compression.py and transduction/pyrfusion.py.
1. Run these with --help for details.
2. Structural configurations are chosen by selecting features through transduction/featconfigs.py. The default options have simple names like 'word', 'ngram', 'dep' and are defined at the top of the file.
* Contact me if you want the model files or system outputs from my experiments.


Support
-------
This code is provided as-is and without any implicit or explicit assurance
of support. Bugs will not be addressed but will be listed in this README.
