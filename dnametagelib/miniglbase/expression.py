"""
**Purpose**
    An all-purpose container for transcript/gene expression data.

**To do**

"""

import sys
import os
import csv
import string
import math
import copy
import heapq
import itertools
import functools
import statistics

from operator import itemgetter
from typing import Any, Iterable

import numpy
import scipy
from numpy import array, arange, meshgrid, zeros, linspace, mean, object_, std # this should be deprecated
from scipy.cluster.hierarchy import distance, linkage, dendrogram
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plot
import matplotlib.cm as cm
from scipy.stats import ttest_ind, mannwhitneyu

from . import config, utils
from .base_expression import base_expression
from .draw import draw
from .progress import progressbar
from .errors import AssertionError, ArgumentError
from .genelist import genelist # Name mangling for the win!
from .location import location
from .stats import stats

if config.NETWORKX_AVAIL and config.PYGRAPHVIZ_AVAIL:
    from .network import network

class expression(base_expression):
    def __init__(self,
                 filename: str | None = None,
                 loadable_list: Iterable | None = None,
                 format: Any = None,
                 expn: Any = None,
                 gzip: bool = False,
                 **kargs: Any):
        """
        **Purpose**
            The base container for expression data.

            Please note that:
            Expression analysis in glbase requires easily manipulatable data.
            Examples for normalisation are
            genespring output, or output from R and PMA, Cufflinks, RSEM,
            EDASeq, DESeq, etc...

        **Arguments**
            filename (Required, one of loadable_list or filename)

                the filename of the microarray to load.

            loadable_list (Required, one of loadable_list or filename)
                a genelist-like object I can use to construct the list from. If you use this then
                expn should be a list of keys to extract the expression data from.

            expn (Required)
                If filename:

                Some sort of descriptor telling me where the expression data actually is.
                For example:
                    "column[3:]" (Take each column from column3 onwards, until the end column)
                Or:
                    "column[4::2]" (Take every second item, from column 4 until the end)
                Or:
                    "column[5:8]" (Take column 5 through 7 - 7 is not a typo. The lists are
                        zero-ordered and closed)

                In fact you can even give compound statements:
                    "[column[7], column[17]]" (Take columns 7 and 17)

                The only rules are it must be a valid piece of python code.

                If a loadable_list:

                This should be the name of the keys used to extract the expresion data

            err (Optional)
                Some sort of descriptor for where to get the error data from.
                NOT IMPLEMENTED

            cv_err (Optional)
                Some sort of descriptor for where to get confidence intervals from.
                This should be a tuple, as confidence intervals are 'lo' and 'hi'.
                NOT IMPLEMENTED

            cond_names (Optional)
                A list of the condition names (in order) if glbase is not working them
                out itself.

            name (Optional)
                By default expression will use the filename (removing any .txt, .tsv, etc) from
                the ends of the names

            silent (Optional, default=False)
                Do not output any reports (primarily this is for internal functions)

            gzip (Optional)
                is the filename gzipped?
        """
        # Not correct, it is possible to generate an empty expression object, for example to load a pandas table
        # This also makes it compatible with genelist() which can also be empty.
        #assert loadable_list or filename, 'You must provide one or other of filename or loadable_list'

        if loadable_list:
            base_expression.__init__(self, loadable_list=loadable_list, expn=expn, **kargs)
        elif filename:
            base_expression.__init__(self, filename=filename, expn=expn, format=format, gzip=gzip, **kargs)
        else:
            genelist.__init__(self)

            self.filename = filename
            self._conditions = [] # Provide a dummy conditions temporarily
            self.name = "None"

    def __repr__(self):
        return "glbase.expression"

    def __str_helper(self, index):
        item = []
        for key in self.linearData[index]:
            if key == 'conditions':
                pass
            elif key == 'err':
                pass
            else:
                item.append(f"{key}: {self.linearData[index][key]}") # % (,  for key in self.linearData[index]])

        # Make sure data and err go on the end
        for key in self.linearData[index]:
            if key == 'conditions':
                linc = str([f'{i:.2f}' for i in self.linearData[index][key]]).replace("'", '')
                item.append(f"data: {linc}")
            elif key == 'err':
                linc = str([f'{i:.2f}' for i in self.linearData[index][key]]).replace("'", '')
                item.append(f"error: Data has standard error data avaiable")

        item = ', '.join(item)
        return f"{index}: {item}"

    def __str__(self):
        """
        (Override)
        give a sensible print out.
        """
        if len(self.linearData) > config.NUM_ITEMS_TO_PRINT:
            out = []
            # welcome to perl
            for index in range(config.NUM_ITEMS_TO_PRINT):
                out.append(self.__str_helper(index))

            out.append(f"... truncated, showing {config.NUM_ITEMS_TO_PRINT}/{len(self.linearData)}")

            if config.PRINT_LAST_ITEM:
                out.append(self.__str_helper(len(self.linearData)-1))

            out = '\n'.join(out)

        elif len(self.linearData) == 0:
            out = "This list is empty"

        else: # just print first entry.
            out = []
            out.append(self.__str_helper(0))
            out = "%s\nShowing %s/%s" % ("\n".join(out), len(self.linearData), len(self.linearData))

        return out

    def __getitem__(self, index):
        """
        Confers:

        a = expn["condition_name"]

        and inherits normal genelist slicing behaviour
        """
        if index in self._conditions:
            return self.getDataForCondition(index)
        return base_expression.__getitem__(self, index) # otherwise inherit

    def __getattr__(self, name):
        """
        Confers this idiom:

        expn.pca.genes(...)

        """

        if name == "pca":
            from .manifold_pca import manifold_pca
            self.som = manifold_pca(parent=self, name=self.name)
            return self.som

        elif name == "stats":
            self.stats = stats(self)
            return self.stats

        elif name == 'tsne':
            assert config.SKLEARN_AVAIL, "Asking for som but sklearn not available"
            from .manifold_tsne import manifold_tsne
            self.tsne = manifold_tsne(parent=self, name=self.name)
            return self.tsne

        elif name == 'umap':
            assert config.UMAP_LEARN_AVAIL, "Asking for a UMAP object but umap-learn is not available"
            from .manifold_umap import manifold_umap
            self.umap = manifold_umap(parent=self, name=self.name)
            return self.umap

        raise AttributeError("'%s' object has no attribute '%s'" % (self.__repr__(), name))

    def sort_sum_expression(self, selected_conditions=None):
        """
        sort by the sum of conditions

        **Arguments**
            selected_conditions (Optional, default=None)
                You can send a list of condition names to use for the sum if you want.

                If this is none, then it uses all conditions

        """
        if selected_conditions:
            selected_condition_indeces = [self._conditions.index(i) for i in selected_conditions]
            comparator = lambda x: sum(
                x["conditions"][i] for i in selected_condition_indeces
            )

            self.linearData = sorted(self.linearData, key=comparator)
        else:
            self.linearData = sorted(self.linearData, key=lambda x: sum(x["conditions"]))
        self._optimiseData()
        return True

    # ----------- overrides/extensions ---------------------------------

    def getColumns(self, return_keys=None, strip_expn=False):
        """
        **Purpose**
            return a new expression only containing the columns specified in return_keys (a list)

            This version for expression objects will preserve the conditions and expresion data.

        **Arguments**
            return_keys (Required)
                A list of keys to keep

            strip_expn (Optional, default=False)
                If True then remove the expression and err keys (if present).

                This will return a genelist

        **Returns**
            A new expression object or if strip_expn=True then a genelist
        """
        assert isinstance(return_keys, list), "getColumns: return_keys must be a list"
        not_found = []
        for k in return_keys:
            if k not in list(self.keys()):
                not_found.append(k)
        assert False not in [k in list(self.keys()) for k in return_keys], "key(s): '%s' not found" % (', '.join(not_found),)
        assert len(return_keys) == len(set(return_keys)), 'return_keys list is not unique'

        if strip_expn:
            newl = genelist()
            newl.name = str(self.name)
        else:
            newl = self.shallowcopy()
            newl.linearData = []
            if not "conditions" in return_keys and "conditions" in self.linearData[0]:
                return_keys.append("conditions")
            if "err" not in return_keys and "err" in self.linearData[0]:
                return_keys.append("err")

        for item in self.linearData:
            newd = {} # Could be done with dict comprehension.
            for key in return_keys:
                newd[key] = item[key] # This is wrong? It will give a view?

            newl.linearData.append(newd)
        newl._optimiseData()

        config.log.info("getColumns: got only the columns: %s" % (", ".join(return_keys),))
        return newl

    def merge(self, key=None, *tables):
        """
        **Purpose**
            Merge a bunch of expression tables by key

            This is basically a hstack for the expression data. This is required because map() can have some
            undesired effects in its treatment of expression objects.

            This method though has the disadvantage that your lists must be identical to begin with.

        **Arguments**
            tables (Required)
                A list of expression-objects to merge

            key (Required)
                The key to base the merge on

        **Returns**
            A new expression object. Only 'self' key's are maintained
        """
        assert key, "merge: You must specify a key"
        lls = [len(i) for i in tables]
        assert len(set(lls)) == 1, "merge: the expression objects must be identically sized"

        newgl = self.deepcopy()
        newgl._conditions = sum((gl._conditions for gl in tables), self._conditions)

        for item in newgl:
            others = [i._findDataByKeyLazy(key=key, value=item[key]) for i in tables]
            if None in others:
                raise AssertionError("merge: %s:%s not found in table" % (key, item[key]))
            others = [i["conditions"] for i in others]
            item["conditions"] = sum(others, item["conditions"])
            if "err" in item:
                item["err"] = sum(others, item["err"])

        newgl._optimiseData()
        return newgl

    def sliceConditions(self,
        conditions:Iterable=None,
        _silent=False,
        **kargs):
        """
        **Purpose**

            return a copy of the expression-data, but only containing
            the condition names specified in conditions

            Note that you can also use this method to change the order of the conditions.
            Just slice all of the keys in the order you want them to occur in.

        **Arguments**
            conditions (Required)
                A list, or other iterable of condition names to extract
                from the expression-data. Every condition name must be present
                on the expression-data.

        **Result**
            A new expression-data object with the same settings as the original,
            but containing only the expression-data conditions specified in
            the 'conditions' argument.

        """
        assert conditions, "sliceConditions: You must specify a list of conditions to keep"
        assert not isinstance(conditions, str), "sliceConditions: You must specify an iterable of conditions to keep"
        assert isinstance(conditions, (tuple, list, set, Iterable)), "sliceConditions: You must specify an iterable of conditions to keep"

        assert len(conditions) == len(set(conditions)), 'The provided condition names are not unique'

        conditions = list(conditions) # Some weird bugs if not a list;
        for item in conditions:
            assert item in self._conditions, f"sliceConditions: '{item}' condition not found in this expression data"

        newgl = self.deepcopy()

        newtab = [newgl.serialisedArrayDataDict[name] for name in conditions]

        # err is not stored as a serialisedArrayDataDict, so have to make one here:
        if "err" in self.keys():
            err_table = numpy.array([i["err"] for i in newgl.linearData])

            err_serialisedArrayDataDict = {}
            for index, name in enumerate(self._conditions):
                if name in conditions: # only load those we are going to slice in
                    err_serialisedArrayDataDict[name] = err_table[:,index]

            new_err_tab = numpy.array([err_serialisedArrayDataDict[name] for name in conditions]).T

            # unpack it back into the err key:
            for i, row in enumerate(new_err_tab):
                newgl.linearData[i]["err"] = list(row)

        newgl._conditions = conditions
        newgl.numpy_array_all_data = numpy.array(newtab).T
        newgl._load_numpy_back_into_linearData() # _conditions must be up to date

        newgl._optimiseData()

        if not _silent: config.log.info("sliceConditions: sliced for %s conditions" % (len(newgl[0]["conditions"]),))
        return newgl

    def getDataForCondition(self, condition_name):
        """
        **Purposse**
            get all of the expression-data data for a particular condition
            name, returns a list of all the values.
            The list remains in the same order as the overall list,

            This method returns a view of the data (not a copy)
        """
        #print self.serialisedArrayDataDict.keys()
        assert condition_name in self.getConditionNames(), "getDataForCondition: No condition named '%s' in this expression object" % condition_name

        return self.serialisedArrayDataDict[condition_name]

    def getExpressionTable(self):
        """
        **Purpose**
            Return the entire expression table as a numpy array.
            Note that rows and columns are not labelled.

        **Arguments**
            None
        """
        return numpy.copy(self.numpy_array_all_data)

    def coerce(self, new_type):
        """
        **Purpose**
            Semi-internal/obscure function. Coerces the data in condition into
            the type specified by new type. Primarily this is to convert the
            expression data from/to integers or floats for downstream R problems.

        **Arguments**
            new_type (Required)
                generally int or float

        **Returns**
            None
            THIS IS AN IN-PLACE CONVERSION
        """
        if new_type == int:
            self.numpy_array_all_data = self.numpy_array_all_data.astype(numpy.int64)
            self._load_numpy_back_into_linearData()
        else:
            for item in self.linearData:
                item["conditions"] = [new_type(i) for i in item["conditions"]]
        return None

    def heatmap(self,
        filename:str =None,
        row_label_key:str ="name",
        row_color_threshold=None,
        optimal_ordering=True,
        dpi:int = 300,
        _draw_supplied_cell_labels=None,
        **kargs):
        """
        **Purpose**

            draw a simple heatmap of the current expression-data data.

        **Arguments**
            filename (Required)
                the filename of the image to save. depending upon the current
                drawing settings it will save either a png (default) svg or eps.

            bracket (Optional, default = no bracketing performed)
                bracket the data within a certain range of values.
                For example to bracket to 0 .. 1 you would use the syntax::

                    result = array.heatmap(filename="ma.png", bracket=[0,1])

                Or for something like log2 normalised array data::

                    result = array.heatmap(filename="ma.png", bracket=[-2,2])

                "Bracket' chops off the edges of the data, using this logic::

                    if value > high_bracket then value := high_bracket
                    if value < low_bracket then value := low_bracket

                See normal for a method that modifies the data.

            row_label_key (Optional, default="name")
                A key in your genelist to use to label the rows. Examples would be gene names accesion
                numbers or something else.

            normal (Optional, default = no normalising)
                Unimplemented

            row_cluster (Optional, default = True)
                cluster the rows? True or False

            row_color_threshold (Optional, default=None)
                color_threshold to color the rows clustering dendrogram.

                See also scipy.hierarchy.dendrogram

            col_cluster (Optional, default = True)
                cluster the column conditions, True or False

            log (Optional, defualt=False, True|False of 2..n for log2, log10)
                log the y axis (defaults to e)
                send an integer for the base, e.g. for log10

                log=10

                for log2

                log=2

                for mathematical constant e

                log=True
                log="e"

            row_tree (Optional, default=False)
                provide your own tree for drawing. Should be a valid dendrogram tree.
                Probably the output from tree(), although you could role your own with Scipy.
                row_labels and the data
                will be rearranged based on the tree, so don't rearrange the data yourself.

            col_tree (Optional, default=False)
                provide your own tree for ordering the data by. See row_tree for details.
                This one is applied to the columns.

            highlights (Optional, default=None)
                sometimes the row_labels will be suppressed as there is too many labels on the plot.
                But you still want to highlight a few specific genes/rows on the plot.
                Send a list to highlights that matches entries in the row_names.

            digitize (Optional, default=False)
                change the colourmap (either supplied in cmap or the default) into a 'digitized' version
                that has large blocks of colours, defined by the number you send to discretize.
                In other words, place the expression values into the number of 'digitized' bins

            cmap (Optional, default=matplotlib.cm.RdBu)
                colour map for the heatmaps. Use something like this:

                import matplotlib.cm as cm

                gl.heatmap(..., cmap=cm.afmhot)

            col_norm (Optional, default=False)
                normalise each column of data between 0 .. max => 0.0 .. 1.0

            row_norm (Optional, default=False)
                similar to the defauly output of heatmap.2 in R, rows are normalised 0 .. 1

            row_font_size (Optional, default=guess suitable size)
                the size of the row labels (in points). If set this will also override the hiding of
                labels if there are too many elements.

            col_font_size (Optional, default=6)
                the size of the column labels (in points)

            heat_wid (Optional, default=0.25)
                The width of the heatmap panel. The image goes from 0..1 and the left most
                side of the heatmap begins at 0.3 (making the heatmap span from 0.3 -> 0.55).
                You can expand or shrink this value depending wether you want it a bit larger
                or smaller.

            heat_hei (Optional, default=0.85)
                The height of the heatmap. Heatmap runs from 0.1 to heat_hei, with a maximum of 0.9 (i.e. a total of 1.0)
                value is a fraction of the entire figure size.

            colbar_label (Optional, default="expression")
                the label to place beneath the colour scale bar

            grid (Optional, default=False)
                draw a grid around each cell in the heatmap.

            draw_numbers (Optional, default=False)
                draw the values of the heatmaps in each cell see also draw_numbers_threshold

            draw_numbers_threshold (Optional, default=-9e14)
                draw the values in the cell if > draw_numbers_threshold

            draw_numbers_fmt (Optional, default= '{:.1f}')
                string formatting for the displayed values

                You can also send arbitrary text here, (for example, if you wanted to
                mark significane with a '*' then you could set draw_numbers_fmt='*').

            draw_numbers_font_size (Optional, default=6)
                the font size for the numbers in each cell

            _draw_supplied_cell_labels (Optional, default=False)
                semi-undocumented function to draw text in each cell.

                Please provide a 2D list, with the same dimensions as the heatmap, and this text
                will be drawn in each cell. Useful for tings like drawing a heatmap of expression
                and then overlaying p-values on top of all significant cells.

            imshow (Optional, default=False)
                Embed the heatmap as an image inside a vector file. (Uses matplotlib imshow
                to draw the heatmap part of the figure. Allows very large matrices to
                be saved as an svg, with the heatmap part as a raster image and all other elements
                as vectors).

            sample_label_colbar (Optional, default=None)
                add a colourbar for the samples names. This is designed for when you have too many
                conditions, and just want to show the different samples as colours

                Should be a list of colours in the same order as the condition names

            col_colbar (Optional, default=None)
                add a colourbar for the samples names. This is designed for when you have too many
                conditions, and just want to show the different samples as colours

                Should be a list of colours in the same order as the condition names

            row_colbar (Optional, default=None)
                add a colourbar for the samples names. This is designed for when you have too many
                conditions, and just want to show the different samples as colours

                Should be a list of colours in the same order as the row names.

                Note that unclustered data goes from the bottom to the top!

            optimal_ordering (Optional, default=True)
                An improved ordering for the tree, at some computational and memory cost.
                Can be trouble on very large heatmaps

                See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

        **Result**
            saves an image to the 'filename' location and

            returns a dictionary containing the real_filename and re-ordered labels after clustering (if any).

            returns the 'actual filename' that really gets saved (glbase
            will modify e.g. a '.png' ending to '.svg' or '.eps' etc. depending
            upon the current setings).
        """
        # checks for option here please.
        assert filename, "heatmap: you must specify a filename"
        assert row_label_key in list(self.keys()), 'row_label_key "%s" not found in this genelist' % row_label_key

        data = self.serialisedArrayDataList

        if "log" in kargs:
            data = self.__log_transform_data(self.serialisedArrayDataList, log=kargs["log"])

        # convert it into the serialisedArrayDataDict that _heatmap() expects.
        newdata = {}
        for index, name in enumerate(self._conditions):
            newdata[name] = data[index] # get the particular column

        res = self.draw.heatmap(data=newdata,
            row_names=self[row_label_key],
            col_names=self.getConditionNames(),
            filename=filename,
            row_color_threshold=row_color_threshold,
            optimal_ordering=optimal_ordering, dpi=dpi,
            _draw_supplied_cell_labels=_draw_supplied_cell_labels,
            **kargs)

        config.log.info("heatmap: Saved %s" % res["real_filename"])
        return res

    def filter_low_expressed(self, min_expression, number_of_conditions):
        """
        **Purpose**
            filter genes by a minimum_expression value in at least number_of_conditions

            Basically the command:

            sum([i > min_expression for i in <expression_data>]) >= number_of_conditions

        **Arguments**
            min_expression (Required)
                The minimum expression value required to pass the test

            number_of_conditions (Required)
                The number of conditions that must be greater than min_expression

        **Results**
            Returns a new genelist
        """

        newl = self.shallowcopy()
        newl.linearData = []

        for item in self.linearData:
            if (
                sum(int(i > min_expression) for i in item["conditions"])
                >= number_of_conditions
            ): # passed
                newl.linearData.append(item.copy())

        assert len(newl) > 0, "filter_low_expressed: The number of genes passing the filter was zero!"
        newl._optimiseData()
        config.log.info("filter_low_expression: removed %s items, list now %s items long" % (len(self) - len(newl), len(newl)))
        return newl

    def scatter(self, x_condition_name, y_condition_name, filename=None, genelist=None, key=None,
        label=False, label_fontsize=6, **kargs):
        """
        **Purpose**
            draw an X/Y dot plot or scatter plot, get R^2 correlation etc.

        **Arguments**
            x_condition_name = the name of the er... X condition
            y_condition_name = the name of the er... Y condition

            genelist (Optional)
                If you send a genelist and a key then these items will be emphasised on the dotplot.

            key (Optional, Required if 'genelist' used)
                The key to match between the expression data and the genelist.

            label (Optional, default=False)
                If genelist and key are set, then you can label these spots with the label from
                genelist[key].
                If you want to label all of the items in the list, then just send the original genelist
                and a key to use and all of the spots will be labelled.

            label_fontsize (Optional, default=14)
                labels fontsize

            do_best_fit_line (Optional, default=False)
                draw a best fit line on the scatter

            print_correlation (Optional, default=None)
                You have to spectify the type of correlation to print on the graph.
                valid are:

                r = R (Correlation coefficient)
                r2 = R^2.
                pearson = Pearson

                You need to also set do_best_fit_line=True for this to work

            spot_size (Optional, default=5)
                The size of each dot.

            plot_diag_slope (Optional, default=False)
                Plot a diagonal line across the scatter plot.

            available key-word arguments:
            xlabel, ylabel, title, log (set this to the base to log the data by),
            xlims, ylims, spot_size,

        **Returns**
            the actual filename saved as and a new image in filename.
        """

        assert filename, "no filename specified"
        assert x_condition_name in self.serialisedArrayDataDict, "%s x-axis condition not found" % x_condition_name
        assert y_condition_name in self.serialisedArrayDataDict, "%s y-axis condition not found" % y_condition_name

        x_data = self.getDataForCondition(x_condition_name)
        y_data = self.getDataForCondition(y_condition_name)

        if "xlabel" not in kargs:
            kargs["xlabel"] = x_condition_name
        if "ylabel" not in kargs:
            kargs["ylabel"] = y_condition_name

        if genelist and key:
            if isinstance(genelist, list): # Compatability for passing a simple list;
                gl = Genelist()
                gl.load_list([{key: v} for v in genelist])
                genelist = gl
            matches = genelist.map(genelist=self, key=key) # make sure resulting object is array
            tx = matches.getDataForCondition(x_condition_name)
            ty = matches.getDataForCondition(y_condition_name)
            if label:
                kargs["spot_labels"] = matches[key]

            real_filename = self.draw.nice_scatter(x_data, y_data, filename, spots=(tx, ty), label_fontsize=label_fontsize, **kargs)
        else:
            real_filename = self.draw.nice_scatter(x_data, y_data, filename, **kargs)


        config.log.info(f"scatter: Saved '{real_filename}'")
        return True

    def boxplots(self,
        filename=None,
        cond_order=None,
        box_colors='lightgrey',
        vert_sizer=0.022,
        p_values=None,
        stats_baseline=None,
        stats_test=None,
        stats_multiple_test_correct=True,
        stats_color_significant=False,
        **kargs):
        """
        **Purpose**
            Draw cute vertical boxplots. These can be preferable to the horizontal
            boxplots as they have more space for labels. The disadvantage is that
            they can only realistically present about 20 samples before the flow off
            the top of the figure. Also they tend to overemphasise vertical
            changes.

            Nonetheless, they have their place. This implementation is also useful
            as you can provide your own q-values (presented on the right hand side)
            of you can specify the stats test and a one versus all stats comparison.

        **Arguments**
            filename (Required)
                filename to save the image to

            vert_sizer (Optional, default=0.022)
                the vertical space each boxplot takes up.

            cond_order (Optinoal, default=None)
                optional order for the conditions (bottom to top), otherwise the condition order
                is taken from the order of expression.getConditionNames()

            box_colors (Optional, default='lightgrey')
                either one color, or a list of colors for each box in the boxplot

            p_values (Optional, default=None)
                A list of p-values, or None if you are using the stats_* system
                described below

            stats_baseline (Optional, default=None)
                condition name for all comparisons to be versus.

            stats_test (Optional, default=None)
                Which statistics test to use.
                One of:
                'ttest_ind' (two-sided, equal_var=True)
                'welch' (two-sided, equal_var=False)
                'mannwhitneyu'

            stats_multiple_test_correct (Optional, default=None)
                if False do not correct for multiple testing.
                If True then use Benjamini-Hochberg to correct.
                (Uses fdr_bh in statsmodels.stats.multitest.multipletests)

            stats_color_significant (Optinoal, default=False
                color statistically significant boxes by the color specified with this value
                overrides box_colors

                If there is a | in there then the left value is the down, the right value is the color
                used for up-regulated. e.g. : "blue|red"

        **Returns**
            None
        """
        from statsmodels.stats.multitest import multipletests
        assert stats_test in (None, 'ttest_ind', 'welch', 'mannwhitneyu'), f'stats_test {stats_test} not found'

        assert filename, "must provide a filename"
        if isinstance(box_colors, list):
            assert len(box_colors) == len(self._conditions), 'box_colors must be the same length as the number of conditions in this dataset'

        # Figure out the q-value tests:
        if p_values and stats_test:
            raise AssertionError('stats_test and p_values cannot both be true')
        elif p_values:
            assert isinstance(p_values, list), 'p_values must be a list'
            assert len(p_values) == len(self.serialisedArrayDataList), 'p_values must be the same length as the number of conditions in this dataset'
        elif stats_test:
            assert stats_baseline in self._conditions, 'stats_baseline not found in this expression data sets conditions'
            assert stats_test in ('ttest_ind', 'welch', 'mannwhitneyu'), f'stats_test {stats_test} not found in (ttest_ind, welch, mannwhitneyu)'

            p_values = []
            base_line = self[stats_baseline]
            for c in self._conditions:
                if c == stats_baseline:
                    p = 1.0
                else:
                    if stats_test == 'ttest_ind':      p = ttest_ind(base_line, self[c], equal_var=True, alternative='two-sided')[1]
                    elif stats_test == 'welch':        p = ttest_ind(base_line, self[c], equal_var=False, alternative='two-sided')[1]
                    elif stats_test == 'mannwhitneyu': p = mannwhitneyu(base_line, self[c], alternative='two-sided')[1]
                p_values.append(p)

        if stats_test and stats_multiple_test_correct and p_values:
            p_values = list(multipletests(p_values, method='fdr_bh')[1])

        if not cond_order:
            data_as_list = [self.serialisedArrayDataDict[k] for k in self._conditions]
            data_labels = self._conditions
        else:
            data_as_list = [self.serialisedArrayDataDict[k] for k in cond_order]
            data_labels = cond_order

        if p_values and stats_color_significant:
            box_colors = []
            m = statistics.median(base_line)
            for c, q in zip(self._conditions, p_values):
                if q < 0.01:
                    if '|' in stats_color_significant:
                        if statistics.median(self[c]) < m: box_colors.append(stats_color_significant.split('|')[0])
                        else: box_colors.append(stats_color_significant.split('|')[1])
                    else:
                        box_colors.append(stats_color_significant)
                else:
                    box_colors.append('lightgrey')

        # do plot
        real_filename = self.draw.boxplots_vertical(
            filename=filename,
            data_as_list = data_as_list,
            data_labels = data_labels,
            qs=p_values,
            sizer=vert_sizer,
            vert_height=4, # does nothing?!
            cols=box_colors,
            bot_pad=0.1,
            **kargs)

        config.log.info(f"boxplots_vertical: Saved '{real_filename}'")
        return p_values


    def log(self, base=math.e, pad=0.00001):
        """
        **Purpose**
            log transform the data

            NOTE: THis is one of the few IN-PLACE glbase commands.

        **Arguments**
            base (Optional, default=math.e)
                the base for the log transform.

            pad (Optional, default=1e-6)
                value to pad all values by to log(0) errors.

        **Returns**
            None
        """
        do_log = False

        if base == math.e or isinstance(base, bool):
            do_log = math.e
        elif isinstance(base, int):
            do_log = base
        else:
            do_log = False

        for item in self:
            item["conditions"] = [math.log(v+pad, do_log) for v in item["conditions"]]
        self._optimiseData()
        return None

    def unlog(self, base=None, adjuster=0.00001):
        """
        **Purpose**
            return the raw data to the unlogged form. YOU MUST PROVIDE THE CORRECT BASE

            NOTE: THis is one of the few IN-PLACE glbase commands.

            Also note that continual repeated log() unlog() will cause the data to 'drift' away from its
            actual values.

        **Arguments**
            base (Required)
                the base for the log transform.

        **Returns**
            None
        """
        for item in self:
            item["conditions"] = [base**(v+adjuster) for v in item["conditions"]]
        self._optimiseData()
        return None
