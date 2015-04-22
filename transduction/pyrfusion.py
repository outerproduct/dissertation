#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import argparse # TODO: remove backported module for 2.7+
from datasets import pyramids
from learning import onlinelearner, structperceptron
import transduction


data_path = '/home/kapil/research/projects/pyrfusion'


def add_corpus_args(parser):
    """Add command-line parameters for specifying the parameters of the
    pyramid corpus.
    """
    parser.add_argument('--pyramid_path', action='store',
            help="path to pyramid dataset",
            default='/proj/fluke/users/kapil/resources/DUC/')
    parser.add_argument('--corpus_path', action='store',
            help="path to store the annotated fusion corpus",
            default=data_path + '/corpora')
#    parser.add_argument('--gigaword_path', action='store',
#            help="path to pickled Gigaword corpus statistics",
#            default='/proj/fluke/users/kapil/resources/gigaword_eng_4/' +
#                    'gigaword.pickle')
    parser.add_argument('--use_labels', action='store_true',
            help="whether to use shorter labels as inputs instead of lines")
    parser.add_argument('--keep_exact_lines', action='store_true',
            help="skip input sentences which exactly match the output")
    parser.add_argument('--skip_exact_labels', action='store_true',
            help="skip input labels which exactly match the output")
    parser.add_argument('--min_inputs', action='store', type=int,
            help="minimum number of input sentences",
            default=2)
    parser.add_argument('--max_inputs', action='store', type=int,
            help="maximum number of input sentences",
            default=4)
    parser.add_argument('--min_words', action='store', type=int,
            help="minimum number of words for a sentence",
            default=5)
    parser.add_argument('--max_words', action='store', type=int,
            help="maximum number of words for a sentence",
            default=100)
    parser.add_argument('--min_scu_line_overlap', action='store', type=float,
            help="minimum ratio of output SCU words appearing in input lines",
            default=1)
    parser.add_argument('--min_scu_part_ratio', action='store', type=float,
            help="minimum ratio of number of SCU label words to " +
                 "any contributor's label words",
            default=0.5)
    parser.add_argument('--min_part_line_ratio', action='store', type=float,
            help="minimum ratio of number of each contributor part label " +
                 "words to its source line words",
            default=0.5)
    parser.add_argument('--corpus_split', action='store', nargs='+', type=int,
            help='train/test split; must sum to 100 (default: TAC/DUC)',
            default=None)
    parser.add_argument('--annotations', action='store', nargs='+',
            help='required annotations for corpus text',
            default=('Porter2', 'Tagchunk', 'Stanford', 'Rasp'))


def get_corpus_name(args):
    """Devise a relatively unique but interpretable name for the corpus.
    """
    return '_'.join((
            'pyr' + ('lbl' if args.use_labels else '') + \
            ('_xL' if args.skip_exact_labels
                else '_x' if not args.keep_exact_lines
                else ''),
            '-'.join((str(args.min_inputs), str(args.max_inputs))),
            '-'.join((str(args.min_words), str(args.max_words))),
            ''.join((str(args.min_scu_line_overlap).lstrip('0'),
                     str(args.min_scu_part_ratio).lstrip('0'),
                     str(args.min_part_line_ratio).lstrip('0'))),
            ('TD' if args.corpus_split is None
                else '+'.join(str(s) for s in args.corpus_split))
            ))


def get_corpus(args):
    """Load an annotated pyramid corpus or regenerate it if it doesn't exist.
    """
    trans_corpus = transduction.GoldTransductionCorpus(
            name=get_corpus_name(args),
            path=args.corpus_path)

    if not trans_corpus.loaded:
        raw_corpus = pyramids.PyramidCorpus(args.pyramid_path)
        raw_corpus.export_instances(
                trans_corpus,
                corpus_split=args.corpus_split,
                use_labels=args.use_labels,
                skip_exact_lines=not args.keep_exact_lines,
                skip_exact_labels=args.skip_exact_labels,
                min_inputs=args.min_inputs,
                max_inputs=args.max_inputs,
                min_words=args.min_words,
                max_words=args.max_words,
                min_part_line_ratio=args.min_part_line_ratio,
                min_scu_part_ratio=args.min_scu_part_ratio,
                min_scu_line_overlap=args.min_scu_line_overlap)
        print len(trans_corpus.instances), "instances created"
        trans_corpus.save()

    # Ensure the corpus has the required annotations
    trans_corpus.annotate_with(args.annotations)
    return trans_corpus

###############################################################################

def add_partition_args(parser):
    """Add command-line parameters to extract the specific instances for
    training or testing.
    """
    parser.add_argument('--dev_percent', action='store', type=int,
            help='percent of training corpus to use for development',
            default=10)
    parser.add_argument('--partition', action='store',
            help='partition to train on (train/dev/test)',
            default='train')
    parser.add_argument('--debug_idxs', action='store', nargs='+', type=int,
            help="ids of specific instances to debug (or -N for the first N)",
            default=None)
    parser.add_argument('--skip_idxs', action='store', nargs='+', type=int,
            help='ids of specific instances to skip',
            default=())

###############################################################################

def add_model_args(parser):
    """Add command-line parameters for specifying the fusion model to use.
    """
    parser.add_argument('--model_name', action='store',
            help="name of model to evaluate with",
            default=None)
    parser.add_argument('--model_path', action='store',
            help="path to store the trained fusion model",
            default=data_path + '/models')
    parser.add_argument('--features', action='store', nargs='+',
            help="feature configuration (word, ngram, dep, arity, range)",
            default=('word', 'ngram',))
    parser.add_argument('--norm', action='store', nargs='+',
            help="normalization for features (1, avg, num, sum)",
            default=('1', 'avg'))
    parser.add_argument('--vars', action='store', nargs='*',
            help="strategies for variable creation",
            default=())
    parser.add_argument('--constraints', action='store', nargs='*',
            help="constraint configuration (fusion)",
            default=())
#            default=('fusion',))
    parser.add_argument('--standardize', action='store_true',
            help="whether to standardize feature values")
    parser.add_argument('--ngram_order', action='store', type=int,
            help="order of n-grams used",
            default=2)
    parser.add_argument('--lm_servers', action='store', nargs='+',
            help="servers for an SRILM language model",
            default=('island1:8081', 'island2:8081', 'island3:8081',
                'island4:8081', 'island5:8081', #'coral9:8081', 'coral10:8081',
                'coral11:8081', 'coral12:8081', 'coral13:8081'))
    parser.add_argument('--dep_servers', action='store', nargs='+',
            help="servers for a dependency model",
            default=('island1:8082', 'island2:8082', 'island3:8082',
                'island4:8082', 'island5:8082', #'coral9:8082', 'coral10:8082',
                'coral11:8082', 'coral12:8082', 'coral13:8082'))


def init_features(trans_corpus, args):
    """Initialize the features and the different LM servers.
    """
    return transduction.TransductionFeatureConfigs(args.features,
            instances=trans_corpus.retrieve_slice(name=args.partition),
            norm_conf=args.norm,
            ngram_order=args.ngram_order,
            standardize=args.standardize)


def get_model_name(args):
    """Devise a relatively unique but interpretable name for the saved
    model.
    """
    return ''.join((
            ('DEBUG_' if args.debug_idxs is not None else '') + \
            get_corpus_name(args) + '_',
            args.partition + '-',
            str(args.dev_percent) + 'dev_',
            '+'.join(name for name in sorted(args.features)),
            '-std' if args.standardize else '',
            '_', '+'.join(name for name in sorted(args.norm)),
            '_n' + str(args.ngram_order),
            '' if len(args.vars) == 0 else '+',
            '+'.join(name[:3] for name in sorted(args.vars)),
            '' if len(args.constraints) == 0 else '_',
            '+'.join(name for name in sorted(args.constraints)),
            ))


def init_learner(trans_corpus, args):
    """Initialize a new or existing online learner (currently the structured
    perceptron).
    """
    # Initialize remote resources needed for computing features
    transduction.init_servers(args.lm_servers, args.dep_servers)

    if args.model_name is not None:
        return structperceptron.StructPerceptron(
                args.model_name,
                model_path=args.model_path)
    else:
        return structperceptron.StructPerceptron(
                get_model_name(args),
                model_path=args.model_path,
                features=init_features(trans_corpus, args),
                ngram_order=args.ngram_order,
                var_conf=args.vars,
                constraint_conf=args.constraints,
                max_flow=100)

###############################################################################

def add_learner_args(parser):
    """Add command-line parameters for the learning and inference.
    """
    onlinelearner.add_args(parser)
    parser.add_argument('--no_tuning', action='store_true',
            help="don't show performance against a tuning corpus")
    parser.add_argument('--tuning_partition', action='store',
            help="partition to evaluate on at each training iteration",
            default='dev')

    # Supplied to the decoder. Note that non-ILP decoders do not currently
    # work for fusion because they only handle forward-ordered bigrams.
    parser.add_argument('--decoder', action='store',
            help="decoding approach to use (ilp/dp/dp+ilp/dp+lp/dp+lp+mst)",
            default='ilp')
    parser.add_argument('--solver', action='store',
            help="the MILP solver to be used (gurobi/lpsolve)",
            default='gurobi')
    parser.add_argument('--timeout', action='store', type=int,
            help="optional time limit for each ILP problem",
            default=None)
    parser.add_argument('--singlecore', action='store_true',
            help="restrict Gurobi to use a single core for ILPs")
    parser.add_argument('--display_output', action='store_true',
            help="show the output produced during decoding")


def train(trans_corpus, args):
    """Train a new fusion model on the pyramid dataset.
    """
    # Add a dev partition within the training partition
    num_train = len(trans_corpus.train_instances)
    train_dev = int((100 - args.dev_percent) * 0.01 * num_train)
    trans_corpus.set_slices(train=(0, train_dev), dev=(train_dev, num_train))

    train_instances = trans_corpus.get_instances(
            partition=args.partition,
            debug_idxs=args.debug_idxs,
            skip_idxs=args.skip_idxs)

    learner = init_learner(trans_corpus, args)
    learner.train(
            train_instances,
            decoder=args.decoder,
            solver=args.solver,
            timeout=args.timeout,
            singlecore=args.singlecore,
            display_output=args.display_output,
            tuning_fn=trans_corpus.evaluate if not args.no_tuning else None,
            tuning_params={'partition': args.tuning_partition,
                           'decoder': args.decoder,
                           'solver': args.solver,
                           'timeout': args.timeout,
                           'singlecore': args.singlecore,
                           'streaming': False,
                           'eval_path': None,
                           'output_path': None},
            **onlinelearner.filter_args(args))

###############################################################################

def add_test_args(parser):
    """Add command-line parameters for evaluation.
    """
    parser.add_argument('--eval_path', action='store',
            help='path to save the detailed evaluation',
            default=data_path + '/evals')
    parser.add_argument('--output_path', action='store',
            help='path to save instance outputs from evaluation',
            default=data_path + '/outputs')
    parser.add_argument('--n_eval', action='store', nargs='+', type=int,
            help='ngram orders to evaluate with',
            default=(1,2,3,4))
    parser.add_argument('--overwrite_params', action='store', nargs='*',
            help='model parameters to overwrite with supplied arguments',
            default=())


def test(trans_corpus, args):
    """Add command line parameters for evaluation.
    """
    learner = init_learner(trans_corpus, args)
    if trans_corpus.name not in learner.name:
        print "WARNING: Testing on corpus", trans_corpus.name,
        print "with model from different corpus", learner.name

    trans_corpus.evaluate(learner,
            partition=args.partition,
            debug_idxs=args.debug_idxs,
            skip_idxs=args.skip_idxs,
            streaming=True,
            n_eval=args.n_eval,
            lm_proxy=transduction.model.lm_proxy,
            decoder=args.decoder,
            solver=args.solver,
            timeout=args.timeout,
            singlecore=args.singlecore,
            display_output=args.display_output,
            eval_path=args.eval_path,
            output_path=args.output_path,
            overwritten_params=get_overwritten_params(args),
            **onlinelearner.filter_args(args))


def get_overwritten_params(args):
    """Retrieve mappings for the overwritten parameters, supplied with the
    command-line arguments
        --overwrite_params arg0 kw=arg1 ...
    where --arg# is the normal argument name and kw (optional) is the variable
    name it is mapped to for keyword arguments, e.g., --constraints maps to
    'constraint_conf' so we modify model constraints with
        --overwrite_params constraint_conf=constraints
    """
    overwritten_params = {}
    for item in args.overwrite_params:
        if '=' in item:
            # Optionally map the command line argument ('--constraints') to
            # the stored parameter name ('constraint_conf')
            param_name, arg_name = item.split('=')
        else:
            param_name, arg_name = item, item
        overwritten_params[param_name] = vars(args)[arg_name]
    return overwritten_params


def baseline(trans_corpus, args):
    """Evaluate a baseline using labels if present.
    """
    # Initialize remote resources needed for computing features
    transduction.init_servers(args.lm_servers, args.dep_servers)

    trans_corpus.set_label_baseline(
            partition=args.partition,
            debug_idxs=args.debug_idxs,
            skip_idxs=args.skip_idxs)

    trans_corpus.evaluate(None,
            partition=args.partition,
            debug_idxs=args.debug_idxs,
            skip_idxs=args.skip_idxs,
            n_eval=args.n_eval,
            lm_proxy=transduction.model.lm_proxy,
            eval_path=args.eval_path,
            output_path=None)

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A text-to-text system.')
    parser.add_argument('mode', help='train|test')
    add_corpus_args(parser)
    add_partition_args(parser)
    add_model_args(parser)
    add_test_args(parser)
    add_learner_args(parser)

    args = parser.parse_args()
    if args.mode in locals():
        locals()[args.mode](get_corpus(args), args)
    else:
        print "Unknown operation:", args.mode
