#! /usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
import cPickle
import matplotlib.pyplot as plt
import os
import scipy as sp
import scipy.stats as stats
import sys


class Evaluation(object):
    """A class to support evaluation of standard experiments and easy
    estimation of statistical significance.
    """
    def __init__(self, filenames=None, title=''):
        """Initialize with the title name.
        """
        self.title = title
        self.max_field_lens = defaultdict(int)

        # numbers[corpus_name][system_name][metric]
        def dictlist():
            return defaultdict(list)
        def dictdictlist():
            return defaultdict(dictlist)
        self.numbers = defaultdict(dictdictlist)

        if filenames is not None:
            # Combine evaluations from multiple files
            for filename in filenames:
                self.restore(filename)

    def update_max_len(self, field_name, value):
        """Update the maximum length of a field, for proper tab separation.
        """
        value_len = None
        if isinstance(value, float):
            value_len = len('%.2f' % (value,))
        else:
            value_len = len(str(value))

        field_name = field_name.lower()
        self.max_field_lens[field_name] = max(self.max_field_lens[field_name],
                                              len(field_name),
                                              value_len)

    def include(self, corpus='default', system='default', **kwargs):
        """Add numbers in the experiments.
        """
        self.update_max_len('corpus', corpus)
        self.update_max_len('system', system)

        if len(kwargs) == 0:
            print "WARNING: nothing to add"
        else:
            for key, val in kwargs.iteritems():
                # We can't use symbols in keyword args so we include
                # some shortcuts:
                # trailing underscore -> %: the value should be represented
                # as a % instead of a fraction
                # leading_underscore -> #: the value should be represented as a
                # count/sum instead of an average
                if key.startswith('_'):
                    key = '#' + key[1:]
                elif key.endswith('_'):
                    key = key[:-1] + ' %'
                self.numbers[corpus][system][key].append(val)
                self.update_max_len(key, val)

    def add_metrics(self, precision, recall, corpus='default',
            system='default', **kwargs):
        """Add precision and recall for a new instance to the evaluation.
        Also compute F-measure and add it.
        """
        f = 0
        if precision + recall > 0:
            f = (2 * precision * recall) / (precision + recall)

        self.include(corpus=corpus,
                     system=system,
                     p_=precision,
                     r_=recall,
                     f_=f,
                     **kwargs)

    def add_counts(self, correct, incorrect, missed, corpus='default',
            system='default', **kwargs):
        """Compute precision and recall from counts for a new instance
        and add them to the evaluation.
        """
        precision, recall = 0, 0
        if correct > 0:
            precision = correct / (correct + incorrect)
            recall = correct / (correct + missed)

        self.add_metrics(precision=precision,
                         recall=recall,
                         corpus=corpus,
                         system=system,
                         _correct=correct,
                         _incorrect=incorrect,
                         _missed=missed,
                         **kwargs)

    def check_num_instances(self):
        """Ensure that the number of instances seen by each system (and
        therefore each metric or extra field) is identical.
        """
        for corpus in self.numbers:
            num_instances = 0
            for system in self.numbers[corpus]:
                for metric in self.numbers[corpus][system]:
                    if num_instances is 0:
                        num_instances = \
                                len(self.numbers[corpus][system][metric])
                    elif len(self.numbers[corpus][system][metric]) > 0 and \
                            len(self.numbers[corpus][system][metric]) != \
                            num_instances:
                        print "WARNING: inconsistent number of instances",
                        print "for corpus", corpus, "(" + metric + "): saw",
                        print len(self.numbers[corpus][system][metric]),
                        print "but expected", num_instances

    def adjust_tabs(self, field_name, value, space_tabs=True):
        """Adjust the tabs for a table cell to fit a word of the given length.
        """
        value_str = None
        if isinstance(value, float):
            value_str = '%.2f' % (value,)

        else:
            value_str = str(value)

        # In case this new value is larger than the rest of the values seen
        # for this field, update the maximum lengths. Note that this can
        # cause inconsistency in the table.
        #self.update_max_len(field_name, value)

        if space_tabs:
            num_spaces_needed = max(len(field_name),
                                    self.max_field_lens[field_name.lower()])
            num_spaces_taken = len(value_str)
            return value_str + ' ' * (num_spaces_needed - num_spaces_taken)
        else:
            num_tabs_needed = int(self.max_field_lens[field_name.lower()]
                                    / 8) + 1
            num_tabs_taken = int(len(value_str) / 8) + 1
            return value_str + '\t' * (num_tabs_needed - num_tabs_taken)

    def table(self, skip_single_keys=False, space_tabs=True):
        """Print a table summarizing the average numbers.
        """
        self.check_num_instances()

        show_corpus = not skip_single_keys or len(self.numbers) > 1
        show_system = not skip_single_keys or len(self.numbers) < \
                sum(len(self.numbers[corpus]) for corpus in self.numbers)

        # TODO: rewrite all this using string.format for sanity

        header = [[self.adjust_tabs('corpus', 'corpus', space_tabs),
                   self.adjust_tabs('system', 'system', space_tabs)][i]
                   for i,j in enumerate((show_corpus, show_system))
                if j]

        # Check if the standard precision/recall metrics are being used
        # and if micro-averaging is needed
        show_micro = 'correct' in self.max_field_lens
        show_prf = show_micro or 'p %' in self.max_field_lens
        if show_prf:
            if show_micro:
                header += [self.adjust_tabs(name, name + '  ', space_tabs)
                            for name in ('avg', 'p %', 'r %', 'f %')]
            else:
                header += [self.adjust_tabs(name, name + '  ', space_tabs)
                            for name in ('p %', 'r %', 'f %')]

        standard_header_set = set(header)
        rows = []

        prev_corpus = None
        prev_system = None
        for corpus in sorted(self.numbers.iterkeys()):
            for system in sorted(self.numbers[corpus].iterkeys()):
                new_row = []
                if show_corpus:
                    new_row.append(self.adjust_tabs('corpus', corpus,
                                                    space_tabs)
                                if corpus != prev_corpus
                                else self.adjust_tabs('corpus', '',
                                                      space_tabs))
                if show_system:
                    new_row.append(self.adjust_tabs('system', system,
                                                    space_tabs)
                                if corpus != prev_corpus or
                                system != prev_system
                                else self.adjust_tabs('system', '',
                                                      space_tabs))
                prev_corpus = corpus
                prev_system = system

                if show_prf:
                    # Macro averaging + additional values
                    if show_micro:
                        new_row.append('macro')
                    for metric in ('p %', 'r %', 'f %'):
                        macro_avg = sp.average(
                                self.numbers[corpus][system][metric])
                        new_row.append(self.adjust_tabs('x %  ',
                                        '%.2f' % (100*macro_avg,), space_tabs))

                # Additional fields
                for field_name in sorted(self.numbers[corpus][system]):
                    if field_name in ('p %', 'r %', 'f %', '#correct',
                                      '#incorrect', '#missed'):
                        # These were either already added to the header (p,r,f)
                        # or are intentionally dropped from the table
                        # (correct, incorrect, missed)
                        continue

                    # Check if the field name was added to the header
                    if '|'.join(header).find(field_name) == -1:
                        header.append(self.adjust_tabs('00.00', field_name,
                                                       space_tabs))

                    # Find out how many spots to leave blank before this
                    # field
                    padding = ''
                    for header_field in header:
                        if header_field.strip() == field_name:
                            # Found a match, so add the padding and move on
                            new_row.append(padding)
                            break
                        elif header_field in self.numbers[corpus][system] \
                                or header_field in standard_header_set:
                            # No padding needed; these were actually printed
                            padding = ''
                        else:
                            padding += self.adjust_tabs(header_field, '',
                                                        space_tabs)

                    new_row.append(self.adjust_tabs(
                        field_name,
                        self.aggregate_field(corpus, system, field_name),
                        space_tabs))
                rows.append(new_row)

                # Micro averaging
                if not show_micro:
                    continue
                sum_correct = sum(self.numbers[corpus][system]['correct'])
                sum_incorrect = sum(self.numbers[corpus][system]['incorrect'])
                sum_missed = sum(self.numbers[corpus][system]['missed'])

                micro_p, micro_r, micro_f = 0, 0, 0
                if sum_correct > 0:
                    micro_p = 100 * sum_correct / (sum_correct + sum_incorrect)
                    micro_r = 100 * sum_correct / (sum_correct + sum_missed)
                    micro_f = (2 * micro_p  * micro_r) / (micro_p + micro_r)

                micro_row = [[self.adjust_tabs('corpus', '', space_tabs),
                              self.adjust_tabs('system', '', space_tabs)][i]
                        for i,j in enumerate((show_corpus, show_system))
                        if j] + ['micro',
                                 '%.2f' % (micro_p,),
                                 '%.2f' % (micro_r,),
                                 '%.2f' % (micro_f,)]
                rows.append(micro_row)

        colsep = '  ' if space_tabs else '\t'
        return '\n'.join([colsep.join(header)] + [colsep.join(row)
                                                  for row in rows])

    def get_padded_id(self, idx, cell_size):
        return '[' + str(idx) + ']' + ' ' * (cell_size - 3)

    def aggregate_field(self, corpus, system, field_name, *args):
        """Return an aggregate value for a field based on its name.
        """
        if len(args) == 0:
            field_values = self.numbers[corpus][system][field_name]
        else:
            # Apply filters against the other fields
            # TODO broken
            field_values = []
            for n, number in enumerate(
                    self.numbers[corpus][system][field_name]):
                failed_filter = False
                for corpus_name, filter_name, filter_value in args:
                    if self.numbers[corpus_name][system][filter_name][n] != \
                            filter_value:
                        failed_filter = True
                        break
                if not failed_filter:
                    field_values.append(number)

        field_sum = sum(field_values)
        field_len = len(field_values)
        if field_name.startswith('#'):
            return field_sum
        elif field_len == 0:
            return 0
        elif field_name.endswith('%'):
            return 100 * field_sum / field_len
        else:
            return field_sum / field_len

    def filter_field(self, field, *args):
        """Apply filters to the numbers.
        """
        # Calculate padding assuming .3f
        cell_size = 5

        for corpus in self.numbers:
            systems = [system for system in self.numbers[corpus]
                            if field in self.numbers[corpus][system]]
            systems = sorted(systems)

            if len(systems) == 0:
                print "skipping", corpus, " with no systems"
                continue
            elif len(systems) == 1:
                print "skipping", corpus, " with only one system", systems[0]
                continue
            print corpus, ":", field
            for s, system in enumerate(systems):
                print self.get_padded_id(s, cell_size),
                print "%.4f" % self.aggregate_field(corpus, system,
                                                    field, *args),
                print system

    def significance(self, field, parametric=False, separator='   '):
        """Print a table showing the p-values from standard tests of
        significance for all systems in a given field.
        """
        # Calculate padding assuming .3f
        cell_size = 5

        for corpus in sorted(self.numbers):
            systems = [system for system in self.numbers[corpus]
                            if field in self.numbers[corpus][system]]
            systems = sorted(systems)

            if len(systems) == 0:
                print "skipping", corpus, " with no systems"
                continue
            elif len(systems) == 1:
                print "skipping", corpus, " with only one system", systems[0]
                continue
            print corpus, ":", field
            for s, system in enumerate(systems):
                print self.get_padded_id(s, cell_size),
                print "%.4f" % self.aggregate_field(corpus, system, field),
                print system

            print separator.join(['   '] + [self.get_padded_id(s, cell_size)
                                        for s in range(len(systems))])
            for s1, system1 in enumerate(systems):
                row_str = self.get_padded_id(s1, cell_size)
                for s2, system2 in enumerate(systems):
                    pval = self.get_pval(self.numbers[corpus][system1][field],
                                         self.numbers[corpus][system2][field],
                                         parametric=parametric)
                    row_str += separator + '%f' % (pval,)
                print row_str
            print

    def significance_best(self, field, parametric=False, lowest_best=False,
            threshold=0.05):
        """Print a list of the best statistically significant system(s)
        for a given field.
        """
        for corpus in sorted(self.numbers):
            systems = [system for system in self.numbers[corpus]
                            if field in self.numbers[corpus][system]]

            if len(systems) == 0:
                print "skipping", corpus, " with no systems"
                continue
            elif len(systems) == 1:
                print "skipping", corpus, " with only one system", systems[0]
                continue
            print corpus, ":", field

            sorted_systems = sorted(systems, reverse=lowest_best,
                                    key=lambda x: self.aggregate_field(corpus,
                                                                       x,
                                                                       field))
            best = [sorted_systems.pop()]
            print "%.4f" % self.aggregate_field(corpus, best[0], field),
            print best[0]

            # TODO: skips high-scoring models which don't get put into best;
            # sometimes, this is a bad idea
            for s, system in enumerate(reversed(sorted_systems)):
                pvals = [self.get_pval(self.numbers[corpus][system][field],
                                       self.numbers[corpus][best_sys][field],
                                       parametric=parametric)
                         for best_sys in best]
                for pval in pvals:
                    if pval > threshold:
                        # Not significantly different from the best systems,
                        # so we add this to the best systems as well
                        # NOTE: we now don't consider systems that are
                        # statistically similar to these additions
                        # best.append(system)
                        print "        ", pvals
                        print "%.4f" % self.aggregate_field(corpus, system,
                                                            field), system
                        break
            print

    def correlation(self, corpus0, system0, field0, corpus1=None,
            system1=None, field1=None, ranked=False):
        """Get the Pearson's correlation coefficient or Spearman's
        ranked correlation coefficient for the data along with the
        corresponding p-value for a null hypothesis of no correlation.
        """
        if corpus1 is None:
            corpus1 = corpus0
        if system1 is None:
            system1 = system0
        if field1 is None:
            field1 = field0

        numbers0 = self.numbers[corpus0][system0][field0]
        numbers1 = self.numbers[corpus1][system1][field1]
        assert len(numbers0) == len(numbers1)

        if ranked:
            return stats.spearmanr(numbers0, numbers1)
        else:
            return stats.pearsonr(numbers0, numbers1)


    def all_significance(self):
        """Print significance tables for all numeric fields.
        """
        # TODO: pointlessly inefficient, but does it matter?
        field_names = set()
        for corpus in self.numbers:
            for system in self.numbers[corpus]:
                for metric in self.numbers[corpus][system]:
                    firstval = self.numbers[corpus][system][metric]
                    if isinstance(firstval, int) or \
                            isinstance(firstval, float):
                        field_names.add(metric)
                    break

        for field_name in field_names:
            print "METRIC:", field_name
            for parametric in (True, False):
                print ['Wilcoxon\'s test', 'T-test'][int(parametric)]
                self.significance(field_name, parametric=parametric)

    @classmethod
    def get_pval(cls, scores1, scores2, parametric=True):
        """Retrieve p-value from lists of scores under a particular test.
        """
        if parametric:
            # Paired t-test
            tstat, pval = stats.ttest_rel(scores1, scores2)
            return pval
        else:
            # Paired Wilcoxon's signed rank test
            tstat, pval = stats.wilcoxon(scores1, scores2)
            return pval

    def save(self, filename, append=False):
        """Pickle the evaluation. Optionally check for an existing evaluation
        and augment it instead of overwriting.
        """
        if append and os.path.exists(filename):
            self.restore(filename)

        # Convert the defaultdicts into regular dicts for pickling.
        for corpus in self.numbers:
            for system in self.numbers[corpus]:
                self.numbers[corpus][system] = \
                        dict(self.numbers[corpus][system])
            self.numbers[corpus] = dict(self.numbers[corpus])
        self.numbers = dict(self.numbers)

        self.max_field_lens = dict(self.max_field_lens)

        print "Saving", self.title, "evaluation to", filename
        with open(filename, 'wb') as f:
            cPickle.dump(self, f, 2)

    def restore(self, filename):
        """Restore a pickled evaluation file.
        """
        with open(filename) as f:
            other = cPickle.load(f)
            self.merge(other, filename=filename)

    def merge(self, other, filename=None):
        """Merge the results of another experiment with this one.
        """
        for corpus in other.numbers:
            for sys in other.numbers[corpus]:

                # HACK to compare systems with the same system name
                if filename is not None and sys in filename:
                    # Use the suffix from the filename as the system name
                    if filename.endswith('.eval'):
                        filename = filename[:-5]
                    i = filename.find(sys)
                    new_sys = filename[i:]

                if new_sys in self.numbers[corpus]:
                    print "WARNING: ignoring merge collision for", corpus, \
                            new_sys
                    continue
                for metric in other.numbers[corpus][sys]:
                    self.numbers[corpus][new_sys][metric].extend(
                            other.numbers[corpus][sys][metric])

        for field_name, other_max_len in other.max_field_lens.iteritems():
            self.max_field_lens[field_name] = max(other_max_len,
                    self.max_field_lens[field_name])

    def list_systems(self):
        """List systems for all corpora.
        """
        systems = set()
        for corpus in self.numbers.iterkeys():
            systems.update(self.numbers[corpus].keys())
        return systems

    def rename_system(self, new_system, old_system=None):
        """Rename a system for all corpora.
        """
        if old_system is None:
            # Check that there's just a single system
            for corpus in self.numbers.iterkeys():
                systems = self.numbers[corpus].keys()
                if old_system is None:
                    old_system = systems[0]
                elif len(systems) > 1 or systems[0] != old_system:
                    print "ERROR: expected a single system to rename"
                    raise Exception

        assert old_system != new_system

        for corpus in self.numbers.iterkeys():
            self.numbers[corpus][new_system] = self.numbers[corpus][old_system]
            del self.numbers[corpus][old_system]

    def delete_system(self, system):
        """Delete a system for all corpora.
        """
        for corpus in self.numbers.iterkeys():
            if system in self.numbers[corpus]:
                del self.numbers[corpus][system]

    def set_plot_properties(self, system_or_corpus, **kwargs):
        """Assign properties for plotting a system or corpus.
        """
        if not hasattr(self, 'plot_props'):
            self.plot_props = {}
        self.plot_props[system_or_corpus] = kwargs

    @staticmethod
    def unpack_tuple(tup):
        """Unpack a tuple specifying data to plot.
        """
        corpus = tup[0]
        key = tup[1]
        scale = tup[2] if len(tup) >= 3 else 'linear'
        name = tup[3] if len(tup) >= 4 else key.replace('_', ' ')

        return corpus, key, scale, name

    def plot_sorted(self, xtuple, ytuple, avg_x=True, bar=False,
            dump=False):
        """Plot one sorted measure denoted by (corpus, field, "linear"/"log")
        against another as a line graph for each system and corpus.
        """
        xcorpus, xkey, xscale, xname = self.unpack_tuple(xtuple)
        ycorpus, ykey, yscale, yname = self.unpack_tuple(ytuple)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        sort_idxs = None
        for system in sorted(self.numbers[xcorpus].keys(), reverse=True):
            x_vals = sp.array(self.numbers[xcorpus][system][xkey])
            y_vals = sp.array(self.numbers[ycorpus][system][ykey])

            if xkey.endswith('%'):
                x_vals *= 100
            if ykey.endswith('%'):
                y_vals *= 100

            # Have to sort by xkey
            if sort_idxs is None:
                sort_idxs = sp.argsort(x_vals)
            # Not a stable sort, I think
            #else:
            #    assert sort_idxs == sp.argsort(x_vals)

            if avg_x:
                x_plot, y_plot = [], []
                run_len = 0
                for x, y in zip(x_vals[sort_idxs], y_vals[sort_idxs]):
                    if len(x_plot) == 0 or x != x_plot[-1]:
                        x_plot.append(x)
                        y_plot.append(y)
                        run_len = 1
                    else:
                        # update the average for the last point
                        y_plot[-1] = ((y_plot[-1]*run_len) + y) / (run_len + 1)
                        run_len += 1
            else:
                x_plot, y_plot = x_vals[sort_idxs], y_vals[sort_idxs]

            if bar:
                plt.bar(x_plot, y_plot, width=0.6, **self.plot_props[system])
            else:
                plt.plot(x_plot, y_plot, '-',
                        clip_on=False,
                        linewidth=2,
                        **self.plot_props[system])

            if dump:
                print system
                for x, y in zip(x_plot, y_plot):
                    print "(%.2f, %.2f)" % (x,y)

        ax.xaxis.set_label_text(xname, size=17, weight='roman')
        ax.yaxis.set_label_text(yname, size=17, weight='roman')

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.grid(b=True)
        if xscale == 'log':
            ax.xaxis.grid(b=True, which='minor')
            ax.xaxis.grid(b=True, which='major',
                    linestyle='-', color=(.2,.2,.2))
        if yscale == 'log':
            ax.yaxis.grid(b=True, which='minor')
            ax.yaxis.grid(b=True, which='major',
                    linestyle='-', color=(.2,.2,.2))

        for label in ax.xaxis.get_ticklabels():
            label.set_fontsize(17)
        for label in ax.yaxis.get_ticklabels():
            label.set_fontsize(17)

#        plt.legend(borderaxespad=0.3, labelspacing=0.3, loc=1,
#                prop={'size': 17})

#        plt.legend(bbox_to_anchor=(0., 1.01, 1., 1.01), loc=3, ncol=4,
#                mode="expand", borderaxespad=0., prop={'size': 17})

#        plt.axis([xmin, xmax, ymin, ymax + 0.0001])
        plt.show()

    def plot_sorted_ydiff(self, xtuple, ytuple, separate=True, **kwargs):
        """Store the difference between the two systems in each
        table and plot it against another field.
        """
        ycorpus, yfield, yscale, yname = self.unpack_tuple(ytuple)

        random_corpus = e.numbers.keys()[0]
        systems = sorted(e.numbers[random_corpus].keys(), reverse=True)
        assert len(systems) == 2
        sys0, sys1 = systems

        # Percentages are scaled in plot_sorted()
        diff_corpus = 'other'
        diff_field = '_diff_%' if yfield.endswith('%') else '_diff_'

        if separate:
            # For each system-specific table, we store the difference which
            # favors that system so that separate colors can be applied.
            # This means that one system table will have all non-negative
            # values while the other will have all non-positive values.
            e.numbers[diff_corpus][sys0][diff_field] = [
                    x0 - x1 if x0 > x1 else 0
                    for x0, x1 in
                    zip(e.numbers[ycorpus][sys0][yfield],
                        e.numbers[ycorpus][sys1][yfield])]

            e.numbers[diff_corpus][sys1][diff_field] = [
                    x0 - x1 if x0 < x1 else 0
                    for x0, x1 in
                    zip(e.numbers[ycorpus][sys0][yfield],
                        e.numbers[ycorpus][sys1][yfield])]
        else:
            e.numbers[diff_corpus][sys0][diff_field] = [
                    x0 - x1 for x0, x1 in
                    zip(e.numbers[ycorpus][sys0][yfield],
                        e.numbers[ycorpus][sys1][yfield])]

            e.numbers[diff_corpus][sys1][diff_field] = [
                    0 for x0, x1 in
                    zip(e.numbers[ycorpus][sys0][yfield],
                        e.numbers[ycorpus][sys1][yfield])]

        e.plot_sorted(xtuple, (diff_corpus, diff_field, yscale, yname),
                      **kwargs)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        # Single filepath given -- print table
        e = Evaluation(filenames=[sys.argv[1]])
        print e.table(skip_single_keys=True)

        random_corpus = e.numbers.keys()[0]
        sys0 = e.numbers[random_corpus].keys()[0]
        print "Avg time:", sp.mean(
                e.numbers['other'][sys0]['decode_time'])
        print "Median time:", sp.median(
                e.numbers['other'][sys0]['decode_time'])

        if 'convergence %' in e.numbers['other'][sys0]:
            n_iters = [iters for converged, iters in \
                       zip(e.numbers['other'][sys0]['convergence %'],
                           e.numbers['other'][sys0]['iterations'])
                       if converged > 0]
            print "Num converged:", len(n_iters)
            print "Avg num iterations:", sum(n_iters) / len(n_iters)

        # XXX (Temporary) report results for each input size
        for input_size in (2,3,4):
            print "SIZE", input_size,
            print "with",
            print len([x for x in e.numbers['other'][sys0]['inputs']
                       if x == input_size]),
            print "instances"
            for corpus in sorted(e.numbers.iterkeys()):
                if not corpus.startswith('GOLD'):
                    continue
                all_f = [f for f, size in
                         zip(e.numbers[corpus][sys0]['f %'],
                             e.numbers['other'][sys0]['inputs'])
                         if size == input_size]
                print "%s\t%.2f" % (corpus, 100 * sum(all_f) / len(all_f))
            print
#            all_lm = [f for f, size in
#                        zip(e.numbers['other'][sys0]['lm'],
#                            e.numbers['other'][sys0]['inputs'])
#                     if size == input_size]
#            print "%s\t%.2f" % ('LM', sum(all_lm) / len(all_lm))

#    elif len(sys.argv) == 3:
#        # XXX (Temporary) two filepaths given -- plot timing difference
#        e = Evaluation(filenames=sys.argv[1:])
#        random_corpus = e.numbers.keys()[0]
#        sys0, sys1 = e.numbers[random_corpus].keys()
#
#        # Assuming 0 is the exact algorithm and 1 is the approximation
#        if sys1.endswith('ilp'):
#            sys0, sys1 = sys1, sys0
#
#        e.set_plot_properties(sys0, color='black')
#        e.set_plot_properties(sys1, color='red')
#        print "black: ", sys0
#        print "red: ", sys1
#
##        e.plot_sorted(('STATS input', 'length', 'linear'),
##                      ('other', 'decode_time', 'log'),
##                      avg_x=True, bar=False, dump=False) #, separate=False)
#        e.plot_sorted_ydiff(('STATS input', 'length', 'linear'),
#                      ('other', 'decode_time', 'linear'),
#                      avg_x=True, bar=True, dump=False) #, separate=False)
#
##        e.set_plot_properties(sys0, color='black', edgecolor='none')
##        e.set_plot_properties(sys1, color='red', edgecolor='none')
##
##        e.plot_sorted_ydiff(('STATS input', 'length', 'linear'),
##                ('GOLD relgraph', 'f %', 'linear'),
##                avg_x=True, bar=True, dump=False) #, separate=False)

    elif len(sys.argv) > 2:
        # Multiple filepaths given -- run significance test between them
        e = Evaluation(filenames=sys.argv[1:])

        # XXX (Temporary) report results for a particular size
#        input_size = 4
#        for corpus in e.numbers:
#            for system in e.numbers[corpus]:
#                if system == 'other':
#                    continue
#                e.numbers[corpus][system]['f %'] = \
#                        [f for f, size in
#                         zip(e.numbers[corpus][system]['f %'],
#                             e.numbers['other'][system]['inputs'])
#                         if size == input_size]

        e.significance_best('f %', parametric=False, threshold=0.05)
#        e.significance_best('lm', parametric=False, threshold=0.05)
    else:
        print "ERROR: must provide one or more .eval files"
