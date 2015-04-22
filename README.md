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
* Ensure the distributed modules are on the `$PYTHONPATH`.
* Module dependencies
  * [argparse](https://code.google.com/p/argparse/) (for Python 2.6)
  * [nltk 3](http://www.nltk.org/) (with Wordnet and Framenet corpora)
  * [psutil](https://code.google.com/p/psutil/)
  * [pyutilib.enum](https://pypi.python.org/pypi/pyutilib.enum)
  * [simplejson](https://pypi.python.org/pypi/simplejson/)
  * [swig-srilm](https://github.com/desilinguist/swig-srilm/blob/master/README.md)
  * [stemming](https://pypi.python.org/pypi/stemming/1.0)
* External software
  * [Gurobi 6.0](http://www.gurobi.com/) (offers academic licensing)
  * [LPsolve](http://lpsolve.sourceforge.net/)
  * [SRILM](http://www.speech.sri.com/projects/srilm/)
  * [Stanford parser 2.0.4](http://nlp.stanford.edu/software/lex-parser.shtml#Download) (or similar older version which produces projective trees)
  * [SEMAFOR](http://www.ark.cs.cmu.edu/SEMAFOR/)
  * [RASP 3.x](http://users.sussex.ac.uk/~johnca/rasp/)
  * [TagChunk](https://www.umiacs.umd.edu/~hal/TagChunk/)
* Data
  * Dependency-converted [Penn treebank](https://catalog.ldc.upenn.edu/LDC99T42) for `interfaces/treebank/depmodel.py` (not necessary for default features)
  * [Clarke & Lapata datasets](http://jamesclarke.net/research/resources) for compression (contact me for dataset splits)
  * Pyramid evaluation data from [DUC 2005-2007](http://www-nlpir.nist.gov/projects/duc/data.html) and [TAC 2008-2011](http://www.nist.gov/tac/data/index.html) for fusion, available from NIST
* Update all paths in the code with appropriate paths to your installations
* Launch servers
  * LM servers through `interfaces/srilm.py`
  * Optionally, PTB servers through `interfaces/treebank/depmodel.py`
* Entry points to the code are `transduction/compression.py` and `transduction/pyrfusion.py`.
  * Run these with `--help` for command-line options.
  * Structural configurations are inferred through feature configurations, defined in `transduction/featconfigs.py`. The default options have simple names like `word`, `ngram`, `dep` and are listed at the top of the file.
* Contact me if you want the model files or system outputs from my experiments.


Support
-------
This code is provided as-is and without any implicit or explicit assurance
of support. Bugs will not be addressed but will be listed in this README.
