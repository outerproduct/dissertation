#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import copy_reg
import learningrate, storedmodel
import multiprocessing
import numpy as np
import os
import random
import sys
import time
import traceback
import types
from utils import jsonrpc, sparse, timer


def _pickle_method(method):
    """Auxiliary function to enable instancemethods to be pickled so that the
    multiprocessing module can share the method between processes.
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class

    # For functions named __foo
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
        if cls_name:
            func_name = '_' + cls_name + func_name

    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """Auxiliary function to enable instancemethods to be unpickled so that
    the multiprocessing module can share the method between processes.
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


def add_args(parser):
    """Add arguments affecting the online learner to the argument
    parser.
    """
    parser.add_argument('--num_epochs', action='store', type=int,
            help="number of training epochs (default 30)",
            default=30)
    parser.add_argument('--num_relaxed_epochs', action='store', type=int,
            help="number of training epochs with relaxed inference " + \
                 "(default 10)",
            default=10)
    parser.add_argument('--minibatch_size', action='store', type=int,
            help="size of each minibatch, (default 1)",
            default=1)
    parser.add_argument('--no_load_balancing', action='store_true',
            help="whether to avoid balancing minibatches")
    parser.add_argument('--rate_updater', action='store',
            help="mechanism for updating the learning rate (constant)",
            default='constant')
    parser.add_argument('--max_processes', action='store', type=int,
            help="maximum #processes on a machine (default #cores)",
            default=None)
    parser.add_argument('--parallel_init', action='store_true',
            help="whether to use multiple processes for initialization")
    parser.add_argument('--parallel_decoding', action='store_true',
            help="whether to use multiple processes for decoding")
    parser.add_argument('--display_weights', action='store_true',
            help="whether to display the model weights at each iteration")
    parser.add_argument('--interruptable', action='store_true',
            help="prompt for continuation at every epoch")
    parser.add_argument('--slaves', action='store', nargs='+',
            help="servers that act as slaves for distributed training",
            default=())
    parser.add_argument('--master_oversees', action='store_true',
            help="whether the master is also used to train instances")


def filter_args(args):
    """Retrieve arguments affecting the online learner.
    """
    args_dict = vars(args)

    # This must match the list above
    valid_keys = ['num_epochs',
                  'num_relaxed_epochs',
                  'minibatch_size',
                  'no_load_balancing',
                  'rate_updater',
                  'max_processes',
                  'parallel_init',
                  'parallel_decoding',
                  'display_weights',
                  'interruptable',
                  'slaves',
                  'master_oversees']
    return dict((key, args_dict[key]) for key in valid_keys)


class OnlineLearner(storedmodel.StoredModel):
    """An online learner for structured prediction.
    """
    def __init__(self, name, features=None, restore=True, model_path='.',
                 reload_check=True, **kwargs):
        """Register methods to allow the instantiated object to use multiple
        processes.
        """
        # TODO: move this away. needed for tuning/eval
        self.model_path = model_path

        # Register methods to make instancemethods picklable for
        # multiprocessing
        copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

        # Initialize the model and (optionally) try to restore it from a
        # previously saved model if desired
        storedmodel.StoredModel.__init__(self, name, restore=restore,
                               model_path=model_path)

        if self.loaded:
            # Validate restored model
            if features is not None and reload_check:
                # Ensure that this is the model we're expecting by
                # comparing the restored features to the supplied features
                if str(self.features) != str(features):
                    print "ERROR: mismatch between supplied features:"
                    print features
                    print "and restored features:"
                    print self.features

                # Repeat the check for non-feature parameters
                # TODO make this also a comparison over sorted strings
                if self.params != kwargs:
                    print "WARNING: mismatch between supplied parameters:"
                    print kwargs
                    print "and restored parameters:"
                    print self.params
        else:
            # Initialize new model
            if features is None:
                print "ERROR: no features supplied for fresh model"
            else:
                self.features = features
                self.params = kwargs
                self.learning_rate = learningrate.LearningRate(**kwargs)
                self.reset()

    def reset(self):
        """Reset the learning rate and all feature weights to start training
        from scratch.
        """
        self.current_weights = np.zeros(len(self.features))
        self.final_weights = np.zeros(len(self.features))
        self.sum_weights = np.zeros(len(self.features))

        self.num_updates_seen = 0
        self.current_epoch = 0
        self.learning_rate.reset()
        self.loaded = True

    def set_learning_rate(self, **kwargs):
        """Select the method and parameters for updating the learning rate.
        """
        self.learning_rate = learningrate.LearningRate(**kwargs)

    def print_weights(self):
        """Print the current and average weights from a given model in a
        readable form for monitoring.
        """
        print "\nWeights  Average  Features"
        if isinstance(self.features, list):
            for cur_w, avg_w, feat in zip(self.current_weights,
                                          self.final_weights,
                                          self.features):
                print '  '.join((str(cur_w), str(avg_w), str(feat)))
        else:
            # A subclassed Feature object
            self.features.print_with_values(self.current_weights,
                                            self.final_weights)

    def print_feats(self, instance, feat_vector):
        """Print the current feature vector and gold feature vector in a
        readable form for debugging.
        """
        print "\nCurrent  Gold  Features for instance", instance.idx
        if isinstance(self.features, list):
            for value, gold_value, feat in zip(feat_vector,
                                               instance.gold_feat_vector,
                                               self.features):
                print '  '.join((str(value), str(gold_value), str(feat)))
        else:
            # A subclassed Feature object
            self.features.print_with_values(feat_vector,
                                            instance.gold_feat_vector)

    def print_model(self):
        """Print a summary of the model, particularly weights for each
        feature.
        """
        print self.name
        print "#Epochs", self.current_epoch
        print "#Updates", self.num_updates_seen

        if isinstance(self.features, list):
            print "\nWeights  Average  Features"
            for f, w in zip(self.features, self.final_weights):
                f_str = f
                if isinstance(f, tuple) or isinstance(f, list):
                    f_str = ', '.join([str(feat) for feat in f])
                w_str = "% .3f" % w
                print w_str + ' '*(10 - len(w_str)) + f_str

            print "\nAdditional parameters:"
            for key, val in self.params.iteritems():
                print key, "=", val
        else:
            # A subclassed Feature object
            self.features.print_with_values(self.final_weights)

    def run(self, instances, parallel_init=False, parallel_decoding=False,
            streaming=False, display_weights=False, overwritten_params=(),
            **kwargs):
        """Selects the appropriate implementation to use.
        """
        if display_weights:
            self.print_weights()

        kwargs.update(self.params)

        # Stored parameters take precedence over new parameters (which may
        # be default arguments) except for those which are explictly
        # marked as overwriting.
        if len(overwritten_params) > 0:
            for param_name, value in overwritten_params.iteritems():
                if param_name in self.params:
                    print "Overwriting stored value",
                    print self.params[param_name],
                    print "for", param_name, "with supplied value",
                    print value
            kwargs.update(overwritten_params)

        if parallel_init or parallel_decoding or not streaming:
            if parallel_init or parallel_decoding:
                print "WARNING: multiprocessed parallelization is not",
                print "recommended for LP-based decoding."
            self.initialize(instances, parallel_init=parallel_init, **kwargs)
            self.decode(instances, parallel_decoding=parallel_decoding,
                        **kwargs)
        else:
            # Single-processed, so we can stream initialization + decoding
            # and clean up afterwards to avoid memory overhead.
            num_instances = len(instances)
            with timer.AvgTimer(num_instances):
                for i, instance in enumerate(instances):
                    instance.initialize(self.features, **kwargs)
                    instance.decode(self.final_weights, self.features,
                            **kwargs)
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
                    sys.stdout.write("Initializing and decoding " +
                            str(num_instances) +
                            " instances: " + str(i+1) + '\r')
                print

    def initialize(self, instances, parallel_init=False, max_processes=None,
            **kwargs):
        """Initialize instances with the feature set. This can be optionally
        used by instances to preprocess for efficiency.
        """
        num_instances = len(instances)

        if parallel_init:
            num_processes = multiprocessing.cpu_count() \
                    if max_processes is None \
                    else min(max_processes, multiprocessing.cpu_count())
            print "Using", num_processes, "processes for initialization"

            pool = multiprocessing.Pool(processes=num_processes)
            args = [(idx, kwargs) for idx in range(len(instances))]
            self.instances = instances

            for i, returned_args in enumerate(pool.imap_unordered(
                    self.per_instance_init, args)):
                idx, instance = returned_args
                self.instances[idx] = instance
                sys.stdout.write("Initializing " + str(num_instances) +
                        " instances: " + str(i+1) + '\r')
                print

            instances = self.instances
            delattr(self, 'instances')
        else:
            with timer.AvgTimer(num_instances):
                for i, instance in enumerate(instances):
                    instance.initialize(self.features, **kwargs)
                    sys.stdout.write("Initializing " + str(num_instances) +
                            " instances: " + str(i+1) + '\r')
                print

    def per_instance_init(self, arg):
        idx, kwargs = arg
        self.instances[idx].initialize(self.features, **kwargs)
        return idx, self.instances[idx]

    def decode(self, instances, parallel_decoding=False, max_processes=None,
            **kwargs):
        """Initialize instances with the feature set. This can be optionally
        used by instances to preprocess for efficiency.
        """
        num_instances = len(instances)

        if parallel_decoding:
            num_processes = multiprocessing.cpu_count() \
                    if max_processes is None \
                    else min(max_processes, multiprocessing.cpu_count())
            print "Using", num_processes, "processes for decoding"

            pool = multiprocessing.Pool(processes=num_processes)
            args = [(idx, kwargs) for idx in range(len(instances))]
            self.instances = instances

            for i, returned_args in enumerate(pool.imap_unordered(
                    self.per_instance_init, args)):
                idx, instance = returned_args
                self.instances[idx] = instance
                sys.stdout.write("Decoding " + str(num_instances) +
                        " instances: " + str(i+1) + '\r')
                print

            instances = self.instances
            delattr(self, 'instances')
        else:
            with timer.AvgTimer(num_instances):
                for i, instance in enumerate(instances):
                    instance.decode(self.final_weights, self.features,
                            **kwargs)
                    sys.stdout.write("Decoding " + str(num_instances) +
                            " instances: " + str(i+1) + '\r')
                print

    def per_instance_decode(self, arg):
        idx, kwargs = arg
        self.instances[idx].decode(self.final_weights,
                self.features,
                **kwargs)
        return idx, self.instances[idx]

    def train(self, instances, parallel_decoding=False, minibatch_size=1,
            slaves=(), **kwargs):
        """Selects the appropriate implementation to use.
        """
        # NOTE: instances are supplied as copies to avoid shuffling the
        # original list

        if len(slaves) == 0 or len(slaves) > len(instances):
            # Single machine
            if minibatch_size > 1:
                print "WARNING: minibatches not yet supported on",
                print "single machines"

            if parallel_decoding:
                # Multiple processes
                return self.train_multi(instances[:], **kwargs)
            else:
                # Single process
                return self.train_single(instances[:], **kwargs)
        else:
            if parallel_decoding:
                print "WARNING: multiprocessed decoding not yet supported on",
                print "multiple machines"

            # Multiple machines
            hostname = os.environ['HOSTNAME']
            port = None
            machine_idx = 0
            for s, slave in enumerate(slaves):
                if slave.startswith(hostname):
                    machine_idx = s + 1
                    print "Slave #" + str(machine_idx)
                    port = slave[slave.rfind(':')+1:]
                    break

            if machine_idx == 0:
                print "Master"
                if minibatch_size == 1:
                    # Distributed serial decoding
                    return self.train_master_serial(instances[:],
                            slaves=slaves,
                            machine_idx=machine_idx,
                            minibatch_size=minibatch_size,
                            **kwargs)
                else:
                    # Parallelized minibatch decoding
                    return self.train_master_minibatch(instances[:],
                            slaves=slaves,
                            machine_idx=machine_idx,
                            minibatch_size=minibatch_size,
                            **kwargs)
            else:
                # Generic slave
                return self.train_slave(instances[:],
                        host=hostname,
                        port=port,
                        slaves=slaves,
                        machine_idx=machine_idx,
                        minibatch_size=minibatch_size,
                        **kwargs)

    def train_single(self, instances, num_epochs=20, num_relaxed_epochs=0,
            **kwargs):
        """Trains a model with the perceptron-style algorithm following
        Collins (2002). Saves model state after every epoch.
        """
        # External models that are needed for instance initialization and/or
        # decoding may be too large to save. We assume that these models
        # will be supplied directly to train(); these are then added to the
        # saved parameters for initialization and decoding.
        kwargs.update(self.params)
        self.initialize(instances, **kwargs)

        # Initialize instances with the feature set. This can be optionally
        # used by instances to preprocess for efficiency.
        num_instances = len(instances)
        for n in range(self.current_epoch, num_epochs):
            random.shuffle(instances)

            with timer.AvgTimer(num_instances):
                for i, instance in enumerate(instances):
                    weight_update = self.decode_instance(
                            instance, self.current_weights, i, epoch=n,
                            num_instances=num_instances,
                            relax=n < num_relaxed_epochs, **kwargs)
                    self.update_weights(weight_update)
                print

            interrupt = self.finish_epoch(**kwargs)
            if interrupt:
                break

    def train_multi(self, instances, num_epochs=20, num_relaxed_epochs=0,
            max_processes=None, **kwargs):
        """Trains a model with the perceptron-style algorithm from
        Collins (2002). Attempts to speed up computation by utilizing
        multiple cores with iterative parameter scaling as described in
        McDonald et al. (2009). Saves model state after every epoch.
        """
        # External models that are needed for instance initialization may
        # be too large to save. We assume that these models will be supplied
        # directly to train(); these are then added to the saved parameters
        # for initialization.
        kwargs.update(self.params)
        # TODO: These arguments are NOT passed to decoding when using
        # multiple processes to avoid a deep copy of the models for each
        # process. Shared memory across processes can be implemented using
        # Value, Array or, more generally, shared ctypes. This is currently
        # not implemented. If needed, use parallel_decoding=False.
        self.initialize(instances, **kwargs)

        # Determine number of parallel processes to use
        num_processes = multiprocessing.cpu_count() if max_processes is None \
                else min(max_processes, multiprocessing.cpu_count())

        # The size of each parallel shard is taken as the floor of the number
        # of instances per process in order to prevent biases in the
        # parameter mixing
        num_instances = len(instances)
        num_shards = num_processes
        shard_size = np.floor(num_instances / num_shards)
        shard_spans = [[i * shard_size, (i+1) * shard_size]
                       for i in range(num_shards)]
        shard_spans[-1][1] = num_instances

        for n in range(self.current_epoch, num_epochs):
            # Randomly shuffle training instances and break them into shards
            # using the indices created earlier
            random.shuffle(instances)
            shards = [instances[begin:end] for begin, end in shard_spans]

            # Pack shard indices and current weight vector together because
            # Pool doesn't support multiple function arguments for parallel
            # maps. Note that local keyword arguments that are not in
            # self.params will not be available for decoding.
            args = [(n, num_relaxed_epochs, shard, self.current_weights)
                    for shard in shards]

            # Initialize a process pool to run a single epoch on
            # each shard asychronously
            pool = multiprocessing.Pool(processes=num_processes)
            next_w = np.zeros(len(self.features))
            with timer.AvgTimer(num_instances):
                for returned_args in pool.imap_unordered(self.per_shard_epoch,
                                                         args):
                    # Unpack returned values
                    w, sum_w, num_shard_updates = returned_args

                    # The final weight vector from the shard, normalized by
                    # the size of the shard, contributes to the input weight
                    # vector for the next iteration
                    normalizer = num_shard_updates / len(instances)
                    next_w += (np.array(w) * normalizer)

                    # Update the running sum of shard weights
                    # TODO: move perceptron-specific update code to subclass
                    self.sum_weights += sum_w
                    self.num_updates_seen += num_shard_updates
                print

            # Terminate pool (processes should already be finished)
            pool.terminate()

            # Set weight vector for next parallel epoch
            self.current_weights = next_w[:]
            interrupt = self.finish_epoch(**kwargs)
            if interrupt:
                break

    def per_shard_epoch(self, arg):
        """Run a single epoch on the given shard of instances.
        Return all the weight vectors seen in the epoch.
        """
        epoch, num_relaxed_epochs, shard, w = arg
        sum_w = np.zeros(len(w))
        num_updates_seen = 0

        for i, instance in enumerate(shard):
            w_update = self.decode_instance(instance, w, i, epoch=epoch,
                                            relax=epoch < num_relaxed_epochs,
                                            **self.params)
            if w_update is not None:
                w = sparse.add(w, w_update)
                sum_w += w
                num_updates_seen += 1
            else:
                print "Skipping weight update"
        return w, sum_w, num_updates_seen

    def train_master_serial(self, instances, machine_idx=0, num_epochs=20,
            num_relaxed_epochs=0, master_oversees=False, **kwargs):
        """Run the master process of a round-robin distributed learner.
        """
        # Sanity check
        assert machine_idx == 0
        if master_oversees:
            print "WARNING: Master cannot oversee in serial mode"

        # Runtime parameters that are not saved with the model
        kwargs.update(self.params)

        # Map out splits for each slave and generate a mapper from instances
        # to slaves.
        spans_per_machine = self.machine_init_serial(
                instances,
                machine_idx=machine_idx,
                **kwargs)
        num_instances = len(instances)
        instance_mapper = [None] * num_instances
        for machine_idx, span in enumerate(spans_per_machine):
            for instance_idx in range(*span):
                instance_mapper[instance_idx] = machine_idx

        # Initialize slave proxies
        all_machines = self.master_init_slaves(
                len(self.current_weights), **kwargs)

        instance_idxs = range(num_instances)
        for n in range(self.current_epoch, num_epochs):
            random.shuffle(instance_idxs)

            with timer.AvgTimer(num_instances):
                prev_machine_idx = -1
                for i, instance_idx in enumerate(instance_idxs):
                    machine_idx = instance_mapper[instance_idx]

                    if machine_idx == 0:
                        instance = instances[instance_idx]
                        weight_update = self.decode_instance(instance,
                                self.current_weights, i, epoch=n,
                                num_instances=num_instances,
                                relax=n < num_relaxed_epochs, **kwargs)
                    else:
                        # If the last instance was decoded by the same slave
                        # we don't need to send the weights again since it
                        # already has them.
                        weight_update, num_updates = \
                                self.master_decode_instance(
                                all_machines[machine_idx],
                                [instance_idx],
                                self.current_weights,
                                self.learning_rate.value(),
                                relax=n < num_relaxed_epochs,
                                slave_has_weights=(
                                    machine_idx==prev_machine_idx),
                                slave_keeps_weights=True)

                    prev_machine_idx = machine_idx
                    self.update_weights(weight_update)
                print
            interrupt = self.finish_epoch(**kwargs)
            if interrupt:
                break

    def train_master_minibatch(self, instances, machine_idx=0, num_epochs=20,
            num_relaxed_epochs=0, master_oversees=False, **kwargs):
        """Run the master process of a round-robin distributed learner.
        """
        # Sanity check
        assert machine_idx == 0

        # Runtime parameters that are not saved with the model
        kwargs.update(self.params)

        # Divide instances into minibatches
        num_instances = len(instances)
        minibatches = self.machine_init_minibatch(
                instances,
                machine_idx=machine_idx,
                master_oversees=master_oversees,
                **kwargs)

        # Initialize slave proxies
        all_machines = self.master_init_slaves(
                len(self.current_weights),
                master_oversees=master_oversees,
                **kwargs)

        pool = multiprocessing.Pool(processes=len(all_machines),
                                    maxtasksperchild=1)
#        manager = multiprocessing.Manager()
#        instance_queue = manager.Queue() if not master_oversees else None
#        instance_queue = None

        for n in range(self.current_epoch, num_epochs):
            if n > 0:  # for comparing and debugging
                random.shuffle(minibatches)

            if not master_oversees:
                print "WARNING: minibatching will duplicate the heap",
                print "for the master."
                print "Consider running with --master_oversees instead."
                self.temp_instances = instances
                self.temp_params = kwargs

            with timer.AvgTimer(num_instances):
                for b, minibatch in enumerate(minibatches):

                    # Print current epoch and minibatch
                    sys.stdout.write("[Epoch " + str(n) + "] Batch " +
                            str(b+1) + "/" + str(len(minibatches)) + "\r")
#                        # Put instances assigned to the master in a queue
#                        for master_idx in minibatch[0]:
#                            instance_queue.put(instances[master_idx])

                    args = [(n, num_relaxed_epochs, all_machines[m],
                             instance_idxs)
#                            self.current_weights, self.learning_rate.value(),
#                            kwargs, instance_queue)
                            for m, instance_idxs in enumerate(minibatch)]

                    weight_update = np.zeros(len(self.current_weights))
                    num_minibatch_updates = 0
                    for returned_args in pool.imap_unordered(
                            self.minibatch_epoch, args):
                        if returned_args is None:
                            print "Failed to decode minibatch", b
                            continue

                        sum_weight_update, num_machine_updates = returned_args
                        if sum_weight_update is not None:
                            weight_update += sum_weight_update
                            num_minibatch_updates += num_machine_updates

                    if num_minibatch_updates > 0:
                        weight_update /= num_minibatch_updates
                    else:
                        weight_update = None
                    self.update_weights(weight_update)
                print

            if not master_oversees:
                del self.temp_instances
                del self.temp_params

            interrupt = self.finish_epoch(**kwargs)
            if interrupt:
                break

    def minibatch_epoch(self, args):
        """Invoke minibatch decoding for each machine.
        """
        (epoch, num_relaxed_epochs, slave_proxy, instance_idxs) = args
#            weights, learning_rate, params, instance_queue) = args

        weights = self.current_weights
        learning_rate = self.learning_rate.value()

        if slave_proxy is None:
            params = self.temp_params
            instances = self.temp_instances

            # Decode the entire batch and sum up the updates
            batch_weight_updates = []
            i = 0
#            while not instance_queue.empty():
#                instance = instance_queue.get()
            for instance_idx in instance_idxs:
                weight_update = self.decode_instance(
                        instances[instance_idx],
                        weights,
                        instance_idxs[i],
                        epoch=epoch,
                        learning_rate=learning_rate,
                        relax=epoch < num_relaxed_epochs,
                        **params)
                if weight_update is not None:
                    batch_weight_updates.append(weight_update)
                i += 1

            if len(batch_weight_updates) == 0:
                return None, None

            # Sum the batch updates and return None if the whole batch failed
            weight_update_sum = sparse.add_all(batch_weight_updates)
            if sparse.is_sparse(weight_update_sum):
                weight_update_sum = sparse.to_list(
                        weight_update_sum, len(weights))
            return len(weight_update_sum), len(batch_weight_updates)
        else:
            return self.master_decode_instance(
                    slave_proxy,
                    instance_idxs,
                    weights,
                    learning_rate,
                    relax=epoch < num_relaxed_epochs,
                    slave_has_weights=False,
                    slave_keeps_weights=False)

    def train_slave(self, instances, host, port, slaves=(), minibatch_size=1,
            timeout=None, limit=65536, **kwargs):
        """Run the slave process of a round-robin distributed learner. This
        stores all the instances and model parameters (including large
        initialized models) as object members and must therefore NEVER
        be saved to disk.
        """
        # Note the inversion here compared to the other methods. Slave models
        # will never be saved to disk and need to store all their state,
        # including runtime parameters, as class members.
        self.params.update(kwargs)
        self.params['timeout'] = timeout
        self.instances = instances

        if minibatch_size == 1:
            self.machine_init_serial(instances,
                    slaves=slaves,
                    **self.params)
        else:
            self.machine_init_minibatch(
                    instances,
                    minibatch_size=minibatch_size,
                    slaves=slaves,
                    **self.params)

        srv_timeout = 3 * timeout if timeout is not None else 1000
        server = jsonrpc.Server(jsonrpc.JsonRpc20(),
                                jsonrpc.TransportTcpIp(addr=(host, int(port)),
                                                       limit=limit,
                                                       timeout=srv_timeout))
        server.register_function(self.slave_init_weights)
        server.register_function(self.slave_receive_weights)
        server.register_function(self.slave_decode_instance)
        server.register_function(self.slave_fetch_weight_update)
        print "Serving at %s:%s with timeout %ds" % (host, port, srv_timeout)
        server.serve()

##############################################################################
# Helper functions for parallelization

    def machine_init_serial(self, instances, slaves=(), machine_idx=0,
            minibatch_size=1, **kwargs):
        """Split up the instances across machines and initialize the
        instances for the local machine.
        """
        num_instances = len(instances)
        num_machines = len(slaves) + 1

        num_per_machine = int(np.ceil(num_instances / num_machines))
        spans_per_machine = [(i * num_per_machine,
            min(num_instances, (i+1) * num_per_machine))
            for i in range(num_machines)]

        # Initialize the instances for this machine
        begin, end = spans_per_machine[machine_idx]
        self.initialize(instances[begin:end], **kwargs)

        print "Distributed serial learner #%d" % (machine_idx,),
        print "for instances %d-%d" % (begin, end-1)

        return spans_per_machine

    def machine_init_minibatch(self, instances, slaves=(), machine_idx=0,
            master_oversees=False, minibatch_size=1, no_load_balancing=False,
            **kwargs):
        """Split up the instances into appropriately-sized minibatches and
        distribute them among machines following Zhao & Huang (2013)
        """
        num_instances = len(instances)
        num_machines = len(slaves) + 1

        # Don't count the master if it's just overseeing
        if master_oversees:
            num_machines -= 1
            machine_idx -= 1

        if minibatch_size % num_machines != 0:
            print "ERROR: minibatch size (" + str(minibatch_size) + ")",
            print "must be an integer multiple of the number of machines",
            print "being used (" + str(num_machines) + ")"
            sys.exit()

        num_minibatches = int(np.ceil(num_instances / minibatch_size))
        minibatch_spans = [(i * minibatch_size,
            min(num_instances, (i+1) * minibatch_size))
            for i in range(num_minibatches)]

        all_minibatched_idxs = []
        local_instances = []
        if no_load_balancing:
            # Split the ordered list of instance indices within the
            # minibatch across machines
            for begin, end in minibatch_spans:
                machine_batch_size = int(np.ceil((end-begin) / num_machines))
                minibatch_idxs = []

                for m in range(num_machines):
                    minibatch_idxs.append(range(begin,
                        min(begin + machine_batch_size, end)))
                    begin += machine_batch_size

                local_instances.extend(instances[idx]
                        for idx in minibatch_idxs[machine_idx])
                all_minibatched_idxs.append(minibatch_idxs)
        else:
            # We maintain an incrementing index l to offset the machine
            # indices. Without it, the round-robin assignment of instance
            # pairs to machines below (i % num_machines) will always start
            # with machine 0. The largest instances from each batch would
            # therefore always end up at machine 0 and the asymmetry in
            # average instance size across machines delays initialization.
            l = 0

            # Pair the largest instance in each minibatch with the smallest
            # and so on
            for begin, end in minibatch_spans:
                minibatch_idxs = [[] for m in range(num_machines)]

                sorted_idxs = sorted(range(begin,end),
                        key=lambda x: instances[x].get_size(),
                        reverse=True)

                num_pairs = int(np.floor(len(sorted_idxs) /
                                (2 * num_machines)))

                i = 0
                j = len(sorted_idxs) - 1
                for pair_per_machine in range(num_pairs * num_machines):
                    m = (l + i) % num_machines
                    minibatch_idxs[m].append(sorted_idxs[i])
                    minibatch_idxs[m].append(sorted_idxs[j])
                    i += 1
                    j -= 1

                # Assign the remaining instances round-robin
                for k in range(i, j + 1):
                    m = (l + k) % num_machines
                    minibatch_idxs[m].append(sorted_idxs[k])

                # Start with a different machine in the next batch
                l += 1

                local_instances.extend(instances[idx]
                        for idx in minibatch_idxs[machine_idx])
                all_minibatched_idxs.append(minibatch_idxs)

        if machine_idx != -1:
            # This was called by the master which is supposed to only oversee
            self.initialize(local_instances, **kwargs)
            print "Distributed minibatch learner #%d" % (machine_idx,),
            print "for %d/%d instances" % (len(local_instances),
                    num_instances)

        return all_minibatched_idxs

    @classmethod
    def master_init_slaves(cls, num_weights, slaves=(), timeout=None,
            limit=65536, master_oversees=False, **kwargs):
        """Initialize proxies for each of the slaves and block on a working
        connection.
        """
        # Initialize server proxies for the slaves
        srv_timeout = 3 * timeout if timeout is not None else 1000
        all_machines = [] if master_oversees else [None]
        for slave in slaves:
            host, port = slave.split(':')
            all_machines.append(jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                jsonrpc.TransportTcpIp(addr=(host, int(port)),
                                       limit=limit,
                                       timeout=srv_timeout)))

            # Block on a working connection to each slave
            machine_idx = len(all_machines) - 1
            working_connection = False
            while not working_connection:
                try:
                    cls.call_slave(all_machines[machine_idx],
                                   'slave_init_weights',
                                   num_weights)
                    working_connection = True
                except jsonrpc.RPCTransportError:
                    print "Waiting to establish connection to slave", slave
                    time.sleep(10)
        return all_machines

    @classmethod
    def master_decode_instance(cls, slave_proxy, instance_idxs, weights,
            learning_rate, relax=False, slave_has_weights=False,
            slave_keeps_weights=False):
        """Decode a specific instance at a slave using the current feature
        weights and return the sparsified vector.
        """
        # Try to make the weight vector lighter to avoid hitting
        # inbound transmission limits
        # FIXME: doesn't appear to work
        #weights = [0 if w == 0 else w for w in self.current_weights]
        group_size = 30

        # Send the weight vector in groups of nonzero components
        if not slave_has_weights:
            cls.call_slave(slave_proxy,
                    'slave_init_weights',
                    len(weights))
            current_group = []
            for w, weight_component in enumerate(weights):
                if weight_component != 0:
                    current_group.append((w, weight_component))

                if len(current_group) >= group_size or \
                        (len(current_group) > 0 and
                        w == len(weights)):
                    cls.call_slave(slave_proxy,
                            'slave_receive_weights',
                            current_group)
                    current_group = []

        # Initiate remote decoding via the slave proxy
        slave_return = cls.call_slave(slave_proxy,
                'slave_decode_instance',
                instance_idxs,
                learning_rate,
                relax,
                slave_keeps_weights)
        if slave_return is None:
            return None, None
        num_components, num_updates = slave_return

        # Now retrieve the full weight update in groups
        # NOTE: we don't need to desparsify right here since the caller
        # should handle sparse vectors transparently
        start_idx = 0
        weight_update = np.zeros(len(weights))
        while start_idx < num_components:
            end_idx = min(start_idx + group_size, num_components)
            nonzero_components = cls.call_slave(slave_proxy,
                    'slave_fetch_weight_update',
                    start_idx,
                    end_idx)
            assert len(nonzero_components) == end_idx - start_idx
            start_idx = end_idx

            for u, update_component in nonzero_components:
                weight_update[u] = update_component

        return weight_update, num_updates

    @staticmethod
    def call_slave(slave_proxy, func_name, *args):
        """Call the slave's methods while being robust to JSON-RPC hiccups.
        """
        num_failures = 0
        failure_limit = 120
        cooldown_period = 5
        while num_failures < failure_limit:
            try:
                return getattr(slave_proxy, func_name)(*args)
            except (jsonrpc.RPCTransportError, jsonrpc.RPCParseError) as e:
                num_failures += 1
                if num_failures < failure_limit:
                    if func_name == 'slave_init_weights' and \
                            type(e).__name__ == 'RPCTransportError':
                        # Probably just establishing a connection;
                        # don't mention failure
                        pass
                    else:
                        sys.stdout.write("Got " + str(num_failures) + " " +
                                type(e).__name__ + "s for " + func_name +
                                "()\r")
                    # Wait a bit, then try again
                    time.sleep(cooldown_period)
                else:
                    raise

##############################################################################

    def slave_init_weights(self, num_weights):
        """Indicate the size of the incoming weight vector.
        """
        self.current_weights = np.zeros(num_weights)
        sys.stdout.write("Receiving " + str(num_weights) + " weights:" + '\r')

    def slave_receive_weights(self, weight_components):
        """Store the non-zero components of the sparsified weight vector.
        """
        if len(weight_components) > 0:
            for w, component in weight_components:
                self.current_weights[w] = component
            sys.stdout.write(' ' * 60 +
                    "\rReceiving " + str(len(self.current_weights)) +
                    " weights: " + str(weight_components[-1][0]) + '\r')

    def slave_decode_instance(self, instance_idxs, learning_rate, relax,
            slave_keeps_weights):
        """Decode a specific instance batch using the given feature weights
        and store an update vector for the weights.
        """
        try:
            # Decode the entire batch and sum up the updates
            batch_weight_updates = []
            for instance_idx in instance_idxs:
                weight_update = self.decode_instance(
                        self.instances[instance_idx],
                        self.current_weights,
                        instance_idx,
                        learning_rate=learning_rate,
                        relax=relax,
                        **self.params)
                if weight_update is not None:
                    batch_weight_updates.append(weight_update)

            # Sum the batch updates and return None if the whole batch failed
            if len(batch_weight_updates) == 0:
                return None
            elif len(batch_weight_updates) == 1:
                weight_update_sum = batch_weight_updates[0]
            else:
                weight_update_sum = sparse.add_all(batch_weight_updates)

            # If this is serial (not minibatch) learning, store the
            # current weights in the hope that the next instance
            # will also be sent to this slave.
            if slave_keeps_weights:
                self.current_weights = sparse.add(self.current_weights,
                                                  weight_update_sum)

            # We only transmit sparsified weight updates. This is almost
            # certainly ensured by the batch addition above.
            if not sparse.is_sparse(weight_update_sum):
                self.weight_update_components = [(idx, val)
                        for idx, val in weight_update_sum if val != 0]
            else:
                # Need to convert to Python datatypes for JSON-RPC
                self.weight_update_components = [(int(idx), float(value))
                        for idx, value
                        in sparse.to_nonzero(weight_update_sum)]

            return (len(self.weight_update_components),
                    len(batch_weight_updates))

        except BaseException:
            # Print the exception so we know what's happening and return
            # a blank update vector
            traceback.print_exception(*sys.exc_info())
            return None

    def slave_fetch_weight_update(self, start_idx, end_idx):
        """Fetch specific non-zero components from the update vector.
        """
        try:
            sys.stdout.write(' ' * 60  + "\rTransmitting " +
                    str(len(self.weight_update_components)) +
                    " weight updates: " +
                    str(start_idx) + '-' + str(end_idx) + '\r')
            return self.weight_update_components[start_idx:end_idx]

        except BaseException:
            # Print the exception so we know what's happening and return
            # a blank update vector
            traceback.print_exception(*sys.exc_info())
            return None
