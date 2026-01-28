"""
draw class for glbase

this is a static class containing various generic methods for drawing etc.

**TODO**

There's a change in direction here. Instead of draw containing lots of generic draw functions
instead its more like a set of wrappers around common ways to do matplotlib stuff.
Drawing inside genelists is fine, as long as it follows this paradigm:

fig = self.draw.getfigure(**kargs)

ax = ...

etc ...

self.draw.do_common_args(fig, **kargs)
filename = fig.savefigure(fig, filename)

It would probably be an improvement if the class part was removed.

Instead a series of methods, exposed by draw.method() at the module level would be better.

This makes them more like helpers for matplotlib than a full fledged object.

This could be easily refactored by changing lines like::

class genelist:
    ... init
        self.draw = draw()

        to

        self.draw = draw

For now, until I refactor the code to remove lines like that.
Also I want to rename this file gldraw to remove name clashes.

Then it can go::

    gldraw.heatmap()
    gldraw.scatter()

"""

import sys
import os
import copy
import random
import numpy
import math
import statistics

from typing import Iterable, Any

from numpy import array, arange, mean, max, min, std, float32
from scipy.cluster.hierarchy import distance, linkage, dendrogram
from scipy.spatial.distance import pdist # not in scipy.cluster.hierarchy.distance as you might expect :(
from numpy import polyfit, polyval
from scipy.stats import linregress
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import matplotlib.cm as cm
from cycler import cycler # colours
from matplotlib.colors import ColorConverter, rgb2hex, ListedColormap
import matplotlib.colors as matplotlib_colors
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse, Circle
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
#from .adjustText import adjust_text

from . import config, utils

# This helps AI recognise the text as text:
matplotlib.rcParams['pdf.fonttype']=42

# this is a work around in the implementation of
# scipy.cluster.hierarchy. It does some heavy
# recursion and even with relatively small samples quickly eats the
# available stack.
# I may need to implement this myself later.
# This can deal with ~23,000 x 14 at least.
# No idea on the upper limit.
sys.setrecursionlimit(5000) # 5x larger recursion.

# define static class here.
class draw:
    def __init__(self, bad_arg=None, **kargs):
        """please deprecate me"""
        pass

    def bracket_data(self,
        data,
        min: int,
        max: int):
        """
        brackets the data between min and max (ie. bounds the data with no scaling)

        This should be a helper?
        """
        newd = copy.deepcopy(data)
        for x, row in enumerate(data):
            for y, value in enumerate(row):
                if value < min:
                    newd[x][y] = min
                elif value > max:
                    newd[x][y] = max
        return newd

    def heatmap(self,
        filename: str = None,
        cluster_mode: str = "euclidean",
        row_cluster: bool = True,
        col_cluster: bool = True,
        vmin = 0,
        vmax = None,
        colour_map=cm.RdBu_r,
        col_norm: bool = False,
        row_norm: bool = False,
        heat_wid = 0.25,
        heat_hei = 0.85,
        highlights = None,
        digitize: bool = False,
        border: bool = False,
        draw_numbers: bool = False,
        draw_numbers_threshold = -9e14,
        draw_numbers_fmt = '{:.1f}',
        draw_numbers_font_size = 6,
        grid: bool = False,
        row_color_threshold: bool = None,
        col_names: bool = None,
        row_colbar: bool = None,
        col_colbar: bool = None,
        optimal_ordering: bool = True,
        dpi:int = 300,
        _draw_supplied_cell_labels = False,
        **kargs):
        """
        my own version of heatmap.

        This will draw a dendrogram ... etc...

        See the inplace variants as to how to use.

        row_names is very important as it describes the order of the data.
        cluster_mode = pdist method. = ["euclidean"] ??????!

        **Arguments**
            data (Required)
                the data to use. Should be a 2D array for the heatmap.

            filename (Required)
                The filename to save the heatmap to.

            col_norm (Optional, default=False)
                normalise each column of data between 0 .. max => 0.0 .. 1.0

            row_norm (Optional, default=False)
                similar to the defauly output of heatmap.2 in R, rows are normalised 0 .. 1

            row_tree (Optional, default=False)
                provide your own tree for drawing. Should be a Scipy tree. row_labels and the data
                will be rearranged based on the tree, so don't rearrnge the data yourself.
                i.e. the data should be unclustered. Use tree() to get a suitable tree for loading here

            col_tree (Optional, default=False)
                provide your own tree for ordering the data by. See row_tree for details.
                This one is applied to the columns.

            row_font_size or yticklabel_fontsize (Optional, default=guess suitable size)
                the size of the row labels (in points). If set this will also override the hiding of
                labels if there are too many elements.

            col_font_size or xticklabel_fontsize (Optional, default=6)
                the size of the column labels (in points)

            heat_wid (Optional, default=0.25)
                The width of the heatmap panel. The image goes from 0..1 and the left most
                side of the heatmap begins at 0.3 (making the heatmap span from 0.3 -> 0.55).
                You can expand or shrink this value depending wether you want it a bit larger
                or smaller.

            heat_hei (Optional, default=0.85)
                The height of the heatmap. Heatmap runs from 0.1 to heat_hei, with a maximum of 0.9 (i.e. a total of 1.0)
                value is a fraction of the entire figure size.

            colbar_label (Optional, default=None)
                the label to place beneath the colour scale bar

            highlights (Optional, default=None)
                sometimes the row_labels will be suppressed as there is too many labels on the plot.
                But you still want to highlight a few specific genes/rows on the plot.
                Send a list to highlights that matches entries in the row_names.

            digitize (Optional, default=False)
                change the colourmap (either supplied in cmap or the default) into a 'discretized' version
                that has large blocks of colours, defined by the number you send to discretize.

                Note that disctretize only colorises the comlourmap and the data is still clustered on the underlying numeric data.

                You probably want to use expression.digitize() for that.

            imshow (Optional, default=False)
                optional ability to use images for the heatmap. Currently experimental it is
                not always supported in the vector output files.

            draw_numbers (Optional, default=False)
                draw the values of the heatmaps in each cell see also draw_numbers_threshold

            draw_numbers_threshold (Optional, default=-9e14)
                draw the values in the cell if > draw_numbers_threshold

            draw_numbers_fmt (Optional, default= '{:.1f}')
                string formatting for the displayed values

            draw_numbers_font_size (Optional, default=6)
                the font size for the numbers in each cell

            _draw_supplied_cell_labels (Optional, default=False)
                semi-undocumented function to draw text in each cell.

                Please provide a 2D list, with the same dimensions as the heatmap, and this text
                will be drawn in each cell. Useful for tings like drawing a heatmap of expression
                and then overlaying p-values on top of all significant cells.

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
                See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

        **Returns**
            The actual filename used to save the image.
        """
        assert filename, "heatmap() - no specified filename"

        # preprocess data
        if isinstance(kargs["data"], dict):
            # The data key should be a serialised Dict, I need to make an array.
            data = array([kargs["data"][key] for key in col_names]).T
            # If the lists are not square then this makes a numpy array of lists.
            # Then it will fail below with a strange error.
            # Let's check to make sure its square:
            ls = [len(kargs["data"][key]) for key in col_names]

            if not all(x == ls[0] for x in ls):
                raise Exception("Heatmap data not Square")
        else:
            # the default is a numpy like array object which can be passed right through.
            data = array(kargs["data"], dtype=float32)

        if col_colbar:
            assert len(col_colbar) == data.shape[1], "col_colbar not the same length as data.shape[1]"
        if row_colbar:
            assert len(row_colbar) == data.shape[0], "row_colbar not the same length as data.shape[0]"

        if col_norm:
            for col in range(data.shape[1]):
                data[:,col] /= float(data[:,col].max())

        if row_norm:
            for row in range(data.shape[0]):
                mi = min(data[row,:])
                ma = max(data[row,:])
                data[row,:] = (data[row,:]-mi) / (ma-mi)

        if "square" in kargs and kargs["square"]:
            # make the heatmap square, for e.g. comparison plots
            left_side_tree =          [0.15,    0.15,   0.10,   0.75]
            top_side_tree =           [0.25,    0.90,  0.55,   0.08]
            heatmap_location =        [0.25,    0.15,   0.55,   0.75]
            loc_col_colbar = [] # not supported?
        else:
            # positions of the items in the plot:
            # heat_hei needs to be adjusted. as 0.1 is the bottom edge. User wants the bottom
            # edge to move up rather than the top edge to move down.
            if row_cluster:
                mmheat_hei = 0.93 - heat_hei # this is also the maximal value (heamap edge is against the bottom)

                left_side_tree =         [0.01,  mmheat_hei,   0.186,      heat_hei]
                top_side_tree =          [0.198,   0.932,        heat_wid,   0.044]
                heatmap_location =       [0.198,   mmheat_hei,   heat_wid,   heat_hei]

                if col_colbar: # Slice a little out of the tree
                    top_side_tree =          [0.198,   0.946,  heat_wid,   0.040]
                    loc_col_colbar =         [0.198,   mmheat_hei+heat_hei+0.002,   heat_wid,  0.012]

                if row_colbar: # Slice a little out of the tree
                    left_side_tree =         [0.01,  mmheat_hei,   0.186-0.018, heat_hei]
                    loc_row_colbar =         [0.198-0.016,   mmheat_hei,   0.014,  heat_hei]

            else:
                # If no row cluster take advantage of the extra width available, but shift down to accomodate the scalebar
                mmheat_hei = 0.89 - heat_hei # this is also the maximal value (heamap edge is against the bottom)
                #top_side_tree =          [0.03,   0.852,  heat_wid,   0.044]
                top_side_tree =          [0.03,   0.891,  heat_wid,   0.020]
                heatmap_location =       [0.03,   mmheat_hei,   heat_wid,  heat_hei]
                loc_row_colbar =         [0.03-0.016,   mmheat_hei,   0.014,  heat_hei] # No need to cut the tree, just squeeze i into the left edge

                if col_colbar:
                    top_side_tree =          [0.03,   0.906,  heat_wid,   0.025] # squeeze up the colbar
                    loc_col_colbar =         [0.03,   0.892,   heat_wid,  0.012] #

        scalebar_location = [0.01,  0.98,   0.14,   0.015]

        # set size of the row text depending upon the number of items:
        row_font_size = 0
        if "row_font_size" in kargs:
            row_font_size = kargs["row_font_size"]
        elif "yticklabel_fontsize" in kargs:
            row_font_size = kargs["yticklabel_fontsize"]
        else:
            if "row_names" in kargs and kargs["row_names"]:
                if len(kargs["row_names"]) <= 100:
                    row_font_size = 6
                elif len(kargs["row_names"]) <= 150:
                    row_font_size = 4
                elif len(kargs["row_names"]) <= 200:
                    row_font_size = 3
                elif len(kargs["row_names"]) <= 300:
                    row_font_size = 2
                else:
                    if not highlights: # if highlights, don't kill kargs["row_names"] and don't print warning.
                        config.log.warning("heatmap has too many row labels to be visible. Suppressing row_labels")
                        kargs["row_names"] = None
                        row_font_size = 1
            else:
                row_font_size = 0

        if highlights:
            highlights = list(set(highlights)) # Make it unique, because sometimes duplicates get through and it erroneously reports failure.
            found = [False for i in highlights]
            if row_font_size == 0:
                row_font_size = 5 # IF the above sets to zero, reset to a reasonable value.
            if "row_font_size" in kargs: # but override ir row_font_size is being used.
                row_font_size = kargs["row_font_size"] # override size if highlights == True
                # I suppose this means you could do row_font_size = 0 if you wanted.

            # blank out anything not in row_names:
            new_row_names = []
            for item in kargs["row_names"]:
                if item in highlights:
                    new_row_names.append(item)
                    index = highlights.index(item)

                    found[index] = True
                else:
                    new_row_names.append("")
            kargs["row_names"] = new_row_names

            for i, e in enumerate(found): # check all highlights found:
                if not e:
                    config.log.warning("highlight: '%s' not found" % highlights[i])

        col_font_size = 6
        if "col_font_size" in kargs:
            col_font_size = kargs["col_font_size"]
        elif "xticklabel_fontsize" in kargs:
            col_font_size = kargs["xticklabel_fontsize"]

        if "bracket" in kargs: # done here so clustering is performed on bracketed data
            data = self.bracket_data(data, kargs["bracket"][0], kargs["bracket"][1])
            vmin = kargs["bracket"][0]
            vmax = kargs["bracket"][1]

        if not vmax:
            """
            I must guess the vmax value. I will do this by working out the
            mean then determining a symmetric colour distribution
            """
            try:
                me = statistics.mean(data)
            except (AttributeError, TypeError):
                me = data.mean()

            ma = abs(me - max(data))
            mi = abs(min(data) + me)
            if ma > mi:
                vmin = me - ma
                vmax = me + ma
            else:
                vmin = me - mi
                vmax = me + mi

        if "row_cluster" in kargs:
            row_cluster = kargs["row_cluster"]
        if "col_cluster" in kargs:
            col_cluster = kargs["col_cluster"]
        if not "colbar_label" in kargs:
            kargs["colbar_label"] = ""
        if "cmap" in kargs:
            colour_map = kargs["cmap"]
        if digitize:
            colour_map = cmaps.discretize(colour_map, digitize)

        # a few grace and sanity checks here;
        if len(data) <= 1: row_cluster = False # clustering with a single point?
        if len(data[0]) <= 1: col_cluster = False # ditto.

        if not "aspect" in kargs:
            kargs["aspect"] = "long"
        fig = self.getfigure(**kargs)

        row_order = False
        if row_cluster:
            # ---------------- Left side plot (tree) -------------------
            ax1 = fig.add_subplot(141)

            # from scipy;
            # generate the dendrogram
            if "row_tree" in kargs:
                assert "dendrogram" in kargs["row_tree"], "row_tree appears to be improperly formed ('dendrogram' is missing)"
                Z = kargs["row_tree"]["linkage"]
            else:
                Y = pdist(data, metric=cluster_mode)
                Z = linkage(Y, method='complete', metric=cluster_mode, optimal_ordering=optimal_ordering)

            if row_color_threshold:
                row_color_threshold = row_color_threshold*((Y.max()-Y.min())+Y.min()) # Convert to local threshold.
                a = dendrogram(Z, orientation='left', color_threshold=row_color_threshold, ax=ax1)
                ax1.axvline(row_color_threshold, color="grey", ls=":")
            else:
                a = dendrogram(Z, orientation='left', ax=ax1)

            ax1.set_position(left_side_tree)
            ax1.set_frame_on(False)
            ax1.set_xticklabels("")
            ax1.set_yticklabels("")
            ax1.set_ylabel("")
            # clear the ticks.
            ax1.tick_params(top=False, bottom=False, left=False, right=False)

            # Use the tree to reorder the data.
            row_order = [int(v) for v in a['ivl']]
            # resort the data by order;
            if "row_names" in kargs and kargs["row_names"]: # make it possible to cluster without names
                newd = []
                new_row_names = []
                for index in row_order:
                    newd.append(data[index])
                    new_row_names.append(kargs["row_names"][index])
                data = array(newd)
                kargs["row_names"] = new_row_names
            else: # no row_names, I still want to cluster
                newd = []
                for index in row_order:
                    newd.append(data[index])
                data = array(newd)

            if row_colbar:
                row_colbar = [row_colbar[index] for index in row_order]

        col_order = False
        if col_cluster:
            # ---------------- top side plot (tree) --------------------
            transposed_data = data.T

            ax2 = fig.add_subplot(142)
            ax2.set_frame_on(False)
            ax2.set_position(top_side_tree)
            if "col_tree" in kargs and kargs["col_tree"]:
                assert "dendrogram" in kargs["col_tree"], "col_tree appears to be improperly formed ('dendrogram' is missing)"
                #if kargs["col_names"] and kargs["col_names"]:
                #    assert len(kargs["col_tree"]["Z"]) == len(kargs["col_names"]), "tree is not the same size as the column labels"
                Z = kargs["col_tree"]["linkage"]
            else:
                Y = pdist(transposed_data, metric=cluster_mode)
                Z = linkage(Y, method='complete', metric=cluster_mode, optimal_ordering=optimal_ordering)
            a = dendrogram(Z, orientation='top', ax=ax2)

            ax2.tick_params(top=False, bottom=False, left=False, right=False)

            ax2.set_xticklabels("")
            ax2.set_yticklabels("")

            col_order = [int(v) for v in a["ivl"]]
            # resort the data by order;
            if col_names: # make it possible to cluster without names
                newd = []
                new_col_names = []
                for index in col_order:
                    newd.append(transposed_data[index])
                    new_col_names.append(col_names[index])
                data = array(newd).T # transpose back orientation
                col_names = new_col_names

            if col_colbar:
                col_colbar = [col_colbar[index] for index in col_order]

        # ---------------- Second plot (heatmap) -----------------------
        ax3 = fig.add_subplot(143)
        if 'imshow' in kargs and kargs['imshow']:
            ax3.set_position(heatmap_location) # must be done early for imshow
            hm = ax3.imshow(data, cmap=colour_map, vmin=vmin, vmax=vmax, aspect="auto",
                origin='lower', extent=[0, data.shape[1], 0, data.shape[0]],
                interpolation=config.get_interpolation_mode(filename)) # Yes, it really is nearest. Otherwise it will go to something like bilinear

        else:
            edgecolors = 'none'
            if grid:
                edgecolors = 'black'
            hm = ax3.pcolormesh(data, cmap=colour_map, vmin=vmin, vmax=vmax, antialiased=False, edgecolors=edgecolors, lw=0.4)

        if col_colbar:
            new_colbar = []
            for c in col_colbar:
                if '#' in c:
                    new_colbar.append([utils.hex_to_rgb(c)]) # needs to be tupled?
                else: # must be a named color:
                    new_colbar.append([matplotlib_colors.to_rgb(c)])

            col_colbar = numpy.array(new_colbar)#.transpose(1,0,2)

            ax4 = fig.add_axes(loc_col_colbar)
            if 'imshow' in kargs and kargs['imshow']:
                col_colbar = numpy.array(new_colbar).transpose(1,0,2)
                ax4.imshow(col_colbar, aspect="auto",
                    origin='lower', extent=[0, len(col_colbar),  0, 1],
                    interpolation=config.get_interpolation_mode(filename))

            else:
                col_colbar = numpy.array(new_colbar)
                # unpack the oddly contained data:
                col_colbar = [tuple(i[0]) for i in col_colbar]
                cols = list(set(col_colbar))
                lcmap = ListedColormap(cols)
                col_colbar_as_col_indeces = [cols.index(i) for i in col_colbar]

                ax4.pcolormesh(numpy.array([col_colbar_as_col_indeces,]), cmap=lcmap,
                    vmin=min(col_colbar_as_col_indeces), vmax=max(col_colbar_as_col_indeces),
                    antialiased=False, edgecolors=edgecolors, lw=0.4)

            ax4.set_frame_on(False)
            ax4.tick_params(top=False, bottom=False, left=False, right=False)
            ax4.set_xticklabels("")
            ax4.set_yticklabels("")

        if row_colbar:
            new_colbar = []
            for c in row_colbar:
                if '#' in c:
                    new_colbar.append([utils.hex_to_rgb(c)]) # needs to be tupled?
                else: # must be a named color:
                    new_colbar.append([matplotlib_colors.to_rgb(c)])

            row_colbar = numpy.array(new_colbar)

            ax4 = fig.add_axes(loc_row_colbar)
            if 'imshow' in kargs and kargs['imshow']:
                ax4.imshow(row_colbar, aspect="auto",
                    origin='lower', extent=[0, len(row_colbar),  0, 1],
                    interpolation=config.get_interpolation_mode(filename))
            else:
                # unpack the oddly contained data:
                row_colbar = [tuple(i[0]) for i in row_colbar]
                cols = list(set(row_colbar))
                lcmap = ListedColormap(cols)
                row_colbar_as_col_indeces = [cols.index(i) for i in row_colbar]
                ax4.pcolormesh(numpy.array([row_colbar_as_col_indeces,]).T, cmap=lcmap,
                    vmin=min(row_colbar_as_col_indeces), vmax=max(row_colbar_as_col_indeces),
                    antialiased=False, edgecolors=edgecolors, lw=0.4)

            ax4.set_frame_on(False)
            ax4.tick_params(top=False, bottom=False, left=False, right=False)
            ax4.set_xticklabels("")
            ax4.set_yticklabels("")

        if draw_numbers:
            for x in range(data.shape[0]):
                for y in range(data.shape[1]):
                    if data[x, y] >= draw_numbers_threshold:
                        if '%' in draw_numbers_fmt:
                            ax3.text(y+0.5, x+0.5, draw_numbers_fmt, size=draw_numbers_font_size,
                                ha='center', va='center')
                        else:
                            ax3.text(y+0.5, x+0.5, draw_numbers_fmt.format(data[x, y]), size=draw_numbers_font_size,
                                ha='center', va='center')

        if _draw_supplied_cell_labels:
            assert len(_draw_supplied_cell_labels) == data.shape[0], '_draw_supplied_cell_labels X does not equal shape[1]'
            assert len(_draw_supplied_cell_labels[0]) == data.shape[1], '_draw_supplied_cell_labels Y does not equal shape[0]'

            # This hack will go wrong if col_cluster or row_cluster is True
            # So fix the ordering:
            x_order = row_order if row_order else range(data.shape[0])
            y_order = col_order if col_order else range(data.shape[1])

            print(x_order, y_order)

            for xp, x in zip(range(data.shape[0]), x_order):
                for yp, y in zip(range(data.shape[1]), y_order):
                    val = _draw_supplied_cell_labels[x][y]
                    if draw_numbers_threshold and val < draw_numbers_threshold:
                        ax3.text(yp+0.5, xp+0.5, draw_numbers_fmt.format(val), size=draw_numbers_font_size,
                            ha='center', va='center')


        ax3.set_frame_on(border)
        ax3.set_position(heatmap_location)
        if col_names:
            ax3.set_xticks(arange(len(col_names))+0.5)
            ax3.set_xticklabels(col_names, rotation="vertical")
            ax3.set_xlim([0, len(col_names)])
            if "square" in kargs and kargs["square"]:
                ax3.set_xticklabels(col_names, rotation="vertical")
        else:
            ax3.set_xlim([0,data.shape[1]])

        if "row_names" in kargs and kargs["row_names"]:
            ax3.set_yticks(arange(len(kargs["row_names"]))+0.5)
            ax3.set_ylim([0, len(kargs["row_names"])])
            ax3.set_yticklabels(kargs["row_names"])
        else:
            ax3.set_ylim([0,data.shape[0]])
            ax3.set_yticklabels("")

        ax3.yaxis.tick_right()
        ax3.tick_params(top=False, bottom=False, left=False, right=False)
        [t.set_fontsize(row_font_size) for t in ax3.get_yticklabels()] # generally has to go last.
        [t.set_fontsize(col_font_size) for t in ax3.get_xticklabels()]

        # Make it possible to blank with x/yticklabels
        if "xticklabels" in kargs:
            ax3.set_xticklabels(kargs["xticklabels"])
        if "yticklabels" in kargs:
            ax3.set_yticklabels(kargs["yticklabels"])

        ax0 = fig.add_subplot(144)
        ax0.set_position(scalebar_location)
        ax0.set_frame_on(False)

        cb = fig.colorbar(hm, orientation="horizontal", cax=ax0)
        cb.set_label(kargs["colbar_label"], fontsize=6)
        cb.ax.tick_params(labelsize=4)

        return {
            "real_filename": self.savefigure(fig, filename, dpi=dpi),
            "reordered_cols": col_names,
            "reordered_rows": kargs["row_names"],
            "reordered_data": data
            }

    def heatmap2(self, filename=None, cluster_mode="euclidean", row_cluster=True, col_cluster=True,
        vmin=0, vmax=None, colour_map=cm.plasma, col_norm=False, row_norm=False, heat_wid=0.25,
        imshow=False,
        **kargs):
        """
        **Purpose**
            This version of heatmap is a simplified heatmap. It does not accept colnames, row_names
            and it outputs the heatmap better centred and expanded to fill the available space
            it does not draw a tree and (unlike normal heatmap) draws a black border
            around. Also, the scale-bar is optional, and by default is switched off.

            It is ideal for drawing sequence tag pileup heatmaps. For that is what it was originally made for
            (see track.heatmap())

        **Arguments**
            data (Required)
                the data to use. Should be a 2D array for the heatmap.

            filename (Required)
                The filename to save the heatmap to.

            col_norm (Optional, default=False)
                normalise each column of data between 0 .. max => 0.0 .. 1.0

            row_norm (Optional, default=False)
                similar to the defauly output of heatmap.2 in R, rows are normalised 0 .. 1

            colbar_label (Optional, default="expression")
                the label to place beneath the colour scale bar

            colour_map (Optional, default=afmhot)
                a matplotlib cmap for colour

            imshow (Optional, default=False)
                optional ability to use images for the heatmap. Currently experimental it is
                not always supported in the vector output files.

        **Returns**
            The actual filename used to save the image.
        """
        assert filename, "heatmap() - no specified filename"

        data = array(kargs["data"], dtype=float32) # heatmap2 can only accept a numpy array

        if col_norm:
            for col in range(data.shape[1]):
                data[:,col] /= float(data[:,col].max())

        if row_norm:
            for row in range(data.shape[0]):
                mi = min(data[row,:])
                ma = max(data[row,:])
                data[row,:] = (data[row,:]-mi) / (ma-mi)

        # positions of the items in the plot:
        heatmap_location =  [0.05,   0.01,   0.90,   0.90]
        scalebar_location = [0.05,  0.97,   0.90,   0.02]

        if "bracket" in kargs: # done here so clustering is performed on bracketed data
            data = self.bracket_data(data, kargs["bracket"][0], kargs["bracket"][1])
            vmin = kargs["bracket"][0]
            vmax = kargs["bracket"][1]
        else:
            vmin = data.min()
            vmax = data.max()

        if not "colbar_label" in kargs:
            kargs["colbar_label"] = "density"

        if "cmap" in kargs: colour_map = kargs["cmap"]

        # a few grace and sanity checks here;
        if len(data) <= 1: row_cluster = False # clustering with a single point?
        if len(data[0]) <= 1: col_cluster = False # ditto.

        if "size" not in kargs:
            kargs["size"] = (3,6)
        fig = self.getfigure(**kargs)

        # ---------------- (heatmap) -----------------------
        ax3 = fig.add_subplot(111)

        if imshow:
            ax3.set_position(heatmap_location) # must be done early for imshow
            hm = ax3.imshow(data, cmap=colour_map, vmin=vmin, vmax=vmax, aspect="auto",
                origin='lower', extent=[0, data.shape[1], 0, data.shape[0]],
                interpolation=config.get_interpolation_mode(filename))
        else:
            hm = ax3.pcolormesh(data, cmap=colour_map, vmin=vmin, vmax=vmax, antialiased=False)

        #ax3.set_frame_on(True)
        ax3.set_position(heatmap_location)

        ax3.set_xlim([0,data.shape[1]])

        ax3.set_ylim([0,data.shape[0]])
        ax3.set_yticklabels("")

        ax3.yaxis.tick_right()
        ax3.tick_params(top=False, bottom=False, left=False, right=False)
        [t.set_fontsize(1) for t in ax3.get_yticklabels()] # generally has to go last.
        [t.set_fontsize(1) for t in ax3.get_xticklabels()]

        ax0 = fig.add_subplot(144)
        ax0.set_position(scalebar_location)
        ax0.set_frame_on(False)

        cb = fig.colorbar(hm, orientation="horizontal", cax=ax0, cmap=colour_map)
        cb.set_label(kargs["colbar_label"])
        [label.set_fontsize(5) for label in ax0.get_xticklabels()]

        return self.savefigure(fig, filename)

    def getfigure(self, size=None, aspect=None, **kargs):
        """
        **Purpose**
            setup a valid figure instance based on size.

        **Arguments**
            size or figsize (Optional, default="medium")
                if size is a tuple then that tuple is the specified size in inches (don't ask)
                You can also specify "small", "medium", "large" and "huge". Corrsponding to approximate pixel
                sizes of (with the "normal" aspect)
                small  :
                medium :
                large  :
                huge   :

                If size is specified then it takes preference over aspect and aspect will be ignored.

            aspect (Optional, default="normal")
                the aspect of the image.
                currently only "normal", "long" and "square" are respected).

        **Returns**
            A valid matplotlib figure object
        """
        if "figsize" in kargs and kargs["figsize"]:
            size = kargs["figsize"]

        if not size:
            size = config.draw_size
        elif len(size) == 2: # A tuple or list?
            size_in_in = (size[0], size[1])

        if not aspect:
            aspect = config.draw_aspect

        if len(size) == 2: # overrides aspect/size
            size_in_in = (size[0], size[1])
            return plot.figure(figsize=size_in_in)

        data = {"normal": {"small": (5,4), "medium": (8,6), "large": (12,9), "huge": (16,12)},
                "square": {"small": (4,4), "medium": (7,7), "large": (9,9), "huge": (12,12)},
                "long": {"small": (4,5), "medium": (6,8), "large": (9,12), "huge": (12,16)},
                "wide": {"small": (7,4), "medium": (12,6), "large": (18,9), "huge": (24,12)}
                }
        dpi = {"small": 75, "medium": 150, "large": 300, "huge": 600} # This dpi doesn't actually work here...
        # See savefigure() for the actual specification
        return plot.figure(figsize=data[aspect][size])

    def savefigure(self, fig, filename, size=config.draw_size, bbox_inches=None, dpi=None):
        """
        **Purpose**
            Save the figure
            to filename, modifying the filename based on the current drawing mode
            (if required)
        **Arguments**
            fig
                the figure handle

            filename
                the filename to save the file to

        **Returns**
            the actual filename used to save the image
        """
        if config.draw_mode == 'jupyter':
            fig.show()
            if not filename: # if filename is something, then allow
                return None

            temp_draw_mode = ['pdf']
        else:
            temp_draw_mode = config.draw_mode
            if isinstance(config.draw_mode, str):
                temp_draw_mode = [config.draw_mode] # for simple compat

        for mode in temp_draw_mode:
            assert mode in config.valid_draw_modes, f"'{mode}' is not a supported drawing mode"

            if mode == 'svg':
                matplotlib.rcParams["image.interpolation"] = 'nearest'
            # So that saving supports relative paths.
            
            path, head = os.path.split(filename)
            if "." in filename: # trust Ralf to send a filename without a . in it Now you get your own special exception!
                save_name = "%s.%s" % (".".join(head.split(".")[:-1]), mode)
            else:
                save_name = "%s.%s" % (head, mode)

            fig.savefig(os.path.join(path, save_name), bbox_inches=bbox_inches, dpi=dpi)
            if config.draw_mode != 'jupyter': # Cannot close in jupyter, it will delete the figure
                plot.close(fig) # Saves a huge amount of memory when saving thousands of images

        return save_name

    def do_common_args(self, ax, **kargs):
        """
        **Purpose**
            deal with common arguments to matplotlib (may not always work, depending upon the figure type.

        **Arguments**
            ax
                an matplotlib axes object

            These are based loosly on the matplotlib versions
                xlabel - x-axis label
                ylabel - y-axis label
                title  - title
                xlims - x axis limits
                ylims - y-axis limits
                zlims - z-axis limits (For 3D plots only)
                xticklabels - list (or not) of labels for the x axis
                logx - set the x scale to a log scale argument should equal the base
                logy - set the y scale to a log scale
                legend_size - size of the legend, small, normal, medium
                xticklabel_fontsize - x tick labels fontsizes
                xticklabels - labels to draw on the x axis
                yticklabel_fontsize - y tick labels fontsizes
                yticklabels - labels to draw on the y axis
                vlines - A list of X points to draw a vertical line at
                hlines - A list of Y points to draw a horizontal line at
                ticks_top - True/False, display the axis ticks on the top
                ticks_bottom - True/False, display the axis ticks on the bottom
                ticks_left - True/False, display the axis ticks on the left
                ticks_right - True/False, display the axis ticks on the right
                ticks - True/False, display any ticks at all
                xticks - List of tick positions you want to draw
                yticks - List of tick positions you want to draw
                grid - True/False switch the grid on or off

        **Returns**
            None
        """
        legend = None
        try:
            legend = ax.get_legend()
        except AttributeError:
            pass
        if legend: # None in no legend on this plot
            legend.get_frame().set_alpha(0.5) # make the legend transparent

        if "xlabel" in kargs:
            ax.set_xlabel(kargs["xlabel"])
        if "ylabel" in kargs:
            ax.set_ylabel(kargs["ylabel"])
        if "title" in kargs:
            if "title_fontsize" in kargs:
                ax.set_title(kargs["title"], fontdict={'fontsize': kargs['title_fontsize']})
            else:
                ax.set_title(kargs["title"], fontdict={'fontsize': 6})
        if "xlims" in kargs:
            ax.set_xlim(kargs["xlims"])
        if "ylims" in kargs:
            ax.set_ylim(kargs["ylims"])
        if "zlims" in kargs: # For 3D plots
            ax.set_zlim([kargs["zlim"][0], kargs["zlim"][1]])
        if "logx" in kargs:
            ax.set_xscale("log", base=kargs["logx"])
        if "logy" in kargs:
            ax.set_yscale("log", base=kargs["logy"])
        if "log" in kargs and kargs["log"]:
            ax.set_xscale("log", basex=kargs["log"])
            ax.set_yscale("log", basey=kargs["log"])
        if "legend_size" in kargs:
            [t.set_fontsize(kargs["legend_size"]) for t in legend.get_texts()]
        if "xticklabel_fontsize" in kargs:
            ax.tick_params(axis='x', labelsize=kargs["xticklabel_fontsize"])
        if "yticklabel_fontsize" in kargs:
            ax.tick_params(axis='y', labelsize=kargs["yticklabel_fontsize"])
        if "xticklabels" in kargs:
            ax.set_xticklabels(kargs["xticklabels"])
        if "yticklabels" in kargs:
            ax.set_yticklabels(kargs["yticklabels"])
        if "vlines" in kargs and kargs["vlines"]:
            for l in kargs["vlines"]:
                ax.axvline(l, ls=":", color="grey", lw=0.5)
        if "hlines" in kargs and kargs["hlines"]:
            for l in kargs["hlines"]:
                ax.axhline(l, ls=":", color="grey", lw=0.5)
        if "alines" in kargs and kargs['alines']:
            for quple in kargs['alines']:
                # quples are interleaved, list of x's and list of y's
                ax.plot((quple[0], quple[2]), (quple[1], quple[3]), ls=':', color='grey', lw=0.5)
        if "grid" in kargs and kargs["grid"]:
            ax.grid(kargs["grid"])
        if "ticks" in kargs and not kargs["ticks"]:
            ax.tick_params(top="off", bottom="off", left="off", right="off")
        if "ticks_top" in kargs and not kargs["ticks_top"]:
            ax.tick_params(top="off")
        if "ticks_bottom" in kargs and not kargs["ticks_bottom"]:
            ax.tick_params(bottom="off")
        if "ticks_left" in kargs and not kargs["ticks_left"]:
            ax.tick_params(left="off")
        if "ticks_right" in kargs and not kargs["ticks_right"]:
            ax.tick_params(right="off")
        if 'xticks' in kargs and kargs["xticks"]:
            ax.set_xticks(kargs['xticks'])
        if 'yticks' in kargs and kargs["yticks"]:
            ax.set_yticks(kargs['yticks'])

    def _simple_heatmap(self, filename=None, colour_map = cm.Reds, vmin=0, vmax=None, symmetric=False, **kargs):
        """
        A simplified version of heatmap, with no clustering, and a simpler representation
        Also, you can change the size and aspect of the display.

        **Arguments**
            data
                an array of arrays or equivalent.

            colour_map
                Default is YlOrRd

            bracket
                specify a tuple for the min max values for the heatmap.

            symmetric (Optional, default=False)
                If set to true, find the mean, the max and the min n the data
                and then set the range of colours to span min .. mean .. max, so that
                the gap between mean and max and min and max are identical.
                If set to False (default behaviour) then simply range the colours from min(data) to
                max(data).

            fig_size (Optional, default=(6,6))
                change the figure size aspect.
        """
        # This should be a wrapper around draw.heatmap() to take advantage of heatmaps
        # better code.

        assert filename, "_heatmap() missing filename"

        data = kargs["data"]

        # positions of the items in the plot:
        heatmap_location =  [0.12,   0.01,   0.75,   0.98]
        scalebar_location = [0.01,  0.96,   0.10,   0.03]

        if "bracket" in kargs:
            data = self.bracket_data(data, kargs["bracket"][0], kargs["bracket"][1])
            vmin = kargs["bracket"][0]
            vmax = kargs["bracket"][1]

        if not vmax:
            """
            I must guess the vmax value. I will do this by working out the
            mean then determining a symmetric colour distribution
            """
            if symmetric:
                me = statistics.mean(data)
                ma = abs(me - max(data))
                mi = abs(min(data) + me)
                if ma > mi:
                    vmin = me - ma
                    vmax = me + ma
                else:
                    vmin = me - mi
                    vmax = me + mi
            else:
                vmax = max(data)
                vmin = min(data)

        if not "aspect" in kargs:
            kargs["aspect"] = "long"
        fig = self.getfigure(**kargs)

        # ---------------- Second plot (heatmap) -----------------------
        ax3 = fig.add_subplot(121)
        hm = ax3.pcolormesh(data, cmap=colour_map, vmin=vmin, vmax=vmax, antialiased=False)

        ax3.set_frame_on(True)
        ax3.set_position(heatmap_location)
        ax3.set_xlim([0,data.shape[1]])
        ax3.set_ylim([0,data.shape[0]])
        ax3.set_yticklabels("")
        ax3.set_xticklabels("")
        ax3.yaxis.tick_right()
        ax3.tick_params(top=False, bottom=False, left=False, right=False)
        #[t.set_fontsize(6) for t in ax3.get_yticklabels()] # generally has to go last.
        #[t.set_fontsize(6) for t in ax3.get_xticklabels()]

        ax0 = fig.add_subplot(122)
        ax0.set_position(scalebar_location)
        ax0.set_frame_on(False)

        cb = fig.colorbar(hm, orientation="horizontal", cax=ax0, cmap=colour_map)
        cb.set_label("")
        [label.set_fontsize(5) for label in ax0.get_xticklabels()]

        return self.savefigure(fig, filename)

    def nice_scatter(self, x=None, y=None, filename=None, do_best_fit_line=False,
        print_correlation=False, spot_size=4, plot_diag_slope=False, label_fontsize=6,
        highlights=None,
        **kargs):
        """
        **Purpose**
            Draw a nice simple scatter plot

        **Arguments**
            x, y (Required)
                x and y values

            filename (Required)
                the filename to save as.

            spots (Optional, must be a 2-length tuple containing (x, y) data)
                These spots will be empahsised with whatever spots_cols is or an
                "Orange" colour by default

            spot_labels (Optional, labels to write on the spots)
                A list of labels to write over the spots.

            label_fontsize (Optional, default=14)
            	labels fontsize

            plot_diag_slope (Optional, default=False)
                Plot a diagonal line across the scatter plot.

            do_best_fit_line (Optional, default=False)
                Draw a line of best fit and the

            print_correlation (Optional, default=None)
                You have to spectify the type of correlation to print on the graph.
                valid are:
                    r = R (Correlation coefficient)
                    r2 = R^2.

            spot_size (Optional, default=5)
                The size of each dot.

            Supported keyword arguments:
                xlabel, ylabel, title, logx, logy

        **Returns**
            the real filename, which may get modified depending upon the current drawing mode
            (usually results in a png)
        """
        fig = self.getfigure(**kargs)
        ax = fig.add_subplot(111)

        ax.scatter(x, y, s=spot_size, c="grey", alpha=0.2, edgecolors="none")

        if "spots" in kargs and kargs["spots"]:
            if "spots_cols" in kargs and kargs["spots_cols"]:
                # Will recognise a string or sequence autmagivally.
                ax.scatter(kargs["spots"][0], kargs["spots"][1], s=spot_size*2, c=kargs["spots_cols"], alpha=0.7, edgecolor="none")
            else:
                ax.scatter(kargs["spots"][0], kargs["spots"][1], s=spot_size*2, c="orange", alpha=0.7, edgecolor="none")

            if "spot_labels" in kargs and kargs["spot_labels"]:
                # for the matplot.lib < 100: I want to label everything.
                for i, n in enumerate(kargs["spot_labels"]):
                    ax.annotate(n, (kargs["spots"][0][i], kargs["spots"][1][i]), size=label_fontsize, color="black", ha="center", va="center")

        if print_correlation or do_best_fit_line:
            # linear regression
            (ar, br) = polyfit(x, y, 1)
            xr = polyval([ar,br], x)
            slope, intercept, r_value, p_value, std_err = linregress(x,y)

            # I think this line is actually wrong?
            mx = [min(x), max(x)]
            my = [slope * min(x) + intercept, slope * max(x) + intercept]

            # Only draw if specified:
            if do_best_fit_line:
                ax.plot(mx, my, "r-.", lw=0.5)

            if print_correlation:
                if print_correlation == "r":
                    ax.set_title("R=%.4f" % r_value)
                elif print_correlation == "r2":
                    ax.set_title("R2=%.4f" % (r_value*r_value))
                elif print_correlation == "pearson":
                    ax.set_title("Pearson=%.4f" % scipy.stats.pearsonr(x,y)[0])
        if plot_diag_slope:
            ax.plot([min(x+y), max(x+y)], [min(x+y), max(x+y)], ":", color="grey")

        if highlights:
            for h in highlights:
                # h = [x, y, label]
                ax.text(h[0], h[1], h[2], fontsize=6, ha='center', va='center')

        if "logx" in kargs and kargs["logx"]:
            ax.set_xscale("log", basex=kargs["logx"])
        if "logy" in kargs and kargs["logy"]:
            ax.set_yscale("log", basey=kargs["logy"])

        self.do_common_args(ax, **kargs)

        return self.savefigure(fig, filename)

    def bar_chart(self, filename=None, genelist=None, data=None, cols=None, **kargs):
        """
        **Purpose**
            draw a bar chart with error bars and interpret and package the data coming from a genelist-like object

        **Args**
            filename

            genelist
                a genelist-like object

            data
                the key to look for in the genelist for the data

            labels
                the key to look for in the genelist for labels

            title (Optional)
                the title

            err (Optional)
                the key to look for in the genelist for error bar values.
                This one assumes symmetric values +- around the data

            err_up (Optional)
                the key to look for errorbars going up

            err_dn (Optional)
                the key to look for error bars going down

            errs_are_absolute (Optional, default=False)
                error bars are not +- from the data, but are values that specify where the error
                bars extend to. This needs to be set to True commonly for confidence intervals
                and left as False for standard errors.

            cols (Optional, default=Use a default set from matplotlib)
                the colours to use for the bar charts, there should be one for each bar.

            Other kargs respected by bar_chart:
                aspect
                size
                xlabel - x-axis label
                ylabel - y-axis label
                title  - title
                xlims - x axis limits
                ylims - y-axis limits
                logx - set the x scale to a log scale argument should equal the base
                logy - set the y scale to a log scale

        **Returns**
            The actual_filename used to save the image
        """

        da = []
        err = []
        err_up = []
        err_dn = []
        for i in genelist:
            da.append(i[data])
            if "err" in kargs:
                err.append(i[kargs["errs"]])
            if "err_up" in kargs:
                err_up.append(i[kargs["err_up"]])
            if "err_dn" in kargs:
                err_dn.append(i[kargs["err_dn"]])

        if "errs_are_absolute" in kargs and kargs["errs_are_absolute"]:
            for i, n in enumerate(da): # normalise the values so matplotlib can underastand them
                if err_up:
                    err_up[i] = [a - b for a, b in zip(err_up[i], n)]
                if err_dn:
                    err_dn[i] = [b - a for a, b in zip(err_dn[i], n)]

        if "cond_names" in kargs:
            labs = kargs["cond_names"]
        else: # fake one
            labs = ["" for t in da]

        if not cols:
            # I need to generate a series of colours.
            cmap = cm.get_cmap(cm.Paired, len(labs))
            cols = []
            step = 256 // len(labs)
            for t in range(1, 256, step):
                cols.append(cmap(t))
            #print cols

        fig = self.getfigure(**kargs)
        ax = fig.add_subplot(111)
        ax.set_position([0.3, 0.1, 0.68, 0.8]) # plenty of space for labels

        # Convert to Numpy arrays for type laziness
        da = array(da).T
        err = array(err).T
        err_up = array(err_up).T
        err_dn = array(err_dn).T
        wid = (1.0 / len(da))-0.05
        x = arange(len(da[0]))

        general_args = {"ec": "black", "ecolor": "black"}

        for i, r in enumerate(da):
            if err:
                ax.barh(x+(wid*i), r, wid, xerr=err, label=labs[i], fc=cols[i], **general_args)
            elif "err_up" in kargs and "err_dn" in kargs:
                ax.barh(x+(wid*i), r, wid, xerr=(err_dn[i], err_up[i]), label=labs[i], fc=cols[i], **general_args)
            elif "err_up" in kargs:
                ax.barh(x+(wid*i), r, wid, xerr=err_up[i], label=labs[i], fc=cols[i], **general_args)
            else:
                ax.barh(x+(wid*i), r, wid, label=labs[i], fc=cols[i], **general_args)
            # I'm sure you don't mean just err_dn
        ax.set_ylim([0, x[-1]+(wid*2)])

        if "cond_names" in kargs:
            leg = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
            leg.get_frame().set_alpha(0.5)

        if "labels" in kargs and kargs["labels"]:
            ax.set_yticklabels(genelist[kargs["labels"]], rotation="horizontal")
            ax.set_yticks(x)#+0.5)

        self.do_common_args(ax, **kargs)

        ax.tick_params(top=False, bottom=False, left=False, right=False)
        #[t.set_fontsize(6) for t in ax3.get_yticklabels()] # generally has to go last.
        [t.set_fontsize(6) for t in ax.get_yticklabels()]

        return self.savefigure(fig, filename)

    def unified_scatter(self,
        labels,
        xdata,
        ydata,
        x,
        y,
        mode='PC',
        filename=None,
        spots=True,
        label=False,
        alpha=0.8,
        perc_weights=None,
        spot_cols='grey',
        overplot=None,
        spot_size=40,
        label_font_size=6,
        label_style=None,
        cut=None,
        squish_scales=False,
        only_plot_if_x_in_label=None,
        adjust_labels=False,
        cmap=None,
        cluster_data=None,
        draw_clusters=None,
        cluster_labels=None,
        cluster_centroids=None,
        **kargs):
        '''
        Unified for less bugs, more fun!
        '''
        ret_data = None

        if not "aspect" in kargs:
            kargs["aspect"] = "square"
        if 'figsize' not in kargs:
            kargs['figsize'] = (4,4)

        fig = self.getfigure(**kargs)
        ax = fig.add_subplot(111)

        if only_plot_if_x_in_label:
            newx = []
            newy = []
            newlab = []
            newcols = []
            for i, lab in enumerate(labels):
                if True in [l in lab for l in only_plot_if_x_in_label]:
                    if overplot and lab not in overplot: # Don't do twice
                        newx.append(xdata[i])
                        newy.append(ydata[i])
                        newlab.append(labels[i])
                        newcols.append(spot_cols[i])
            xdata = newx
            ydata = newy
            labels = newlab
            spot_cols = newcols
            #print zip(spot_cols, labels)

        if overplot: # Make sure some spots are on the top
            newx = []
            newy = []
            newcols = []
            for i, lab in enumerate(labels):
                if True in [l in lab for l in overplot]:
                    newx.append(xdata[i])
                    newy.append(ydata[i])
                    newcols.append(spot_cols[i])

        if draw_clusters:
            # unpack the cluster_data for convenience
            ax.set_prop_cycle(cycler(color=plot.get_cmap('tab20c').colors))
            n_clusters = cluster_data.n_clusters

            for labelk in range(n_clusters):
                cluster_center = cluster_centroids[labelk]
                this_x = [xdata[i] for i, l in enumerate(cluster_labels) if l == labelk]
                this_y = [ydata[i] for i, l in enumerate(cluster_labels) if l == labelk]
                ax.scatter(this_x, this_y, s=spot_size+1, alpha=1.0, edgecolors="none", zorder=5)

                #ax.plot(xdata[labelk], ydata[labelk], 'w', markerfacecolor=col, marker='.')

                ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='none', markeredgecolor='black', alpha=0.4, markersize=6, zorder=6)
                ax.text(cluster_center[0], cluster_center[1], labelk, ha='center', zorder=7)

            ax.scatter(xdata, ydata, s=spot_size,
                        alpha=alpha, edgecolors="none",
                        c=spot_cols, cmap=cmap,
                        zorder=2)
        elif spots:
            ax.scatter(xdata, ydata,
                alpha=alpha, edgecolors="none",
                s=spot_size,
                c=spot_cols,
                cmap=cmap,
                zorder=2)
        else:
            # if spots is false then the axis limits are set to 0..1. I will have to send my
            # own semi-sensible limits:
            squish_scales = True

        if overplot:
            ax.scatter(newx, newy, s=spot_size+1, alpha=alpha, edgecolors="none", c=newcols, zorder=5)

        if label:
            texts = []
            for i, lab in enumerate(labels):
                if not spots and isinstance(spot_cols, list):
                    texts.append(ax.text(xdata[i], ydata[i], lab, size=label_font_size, color=spot_cols[i], style=label_style, ha='center'))
                else:
                    texts.append(ax.text(xdata[i], ydata[i], lab, size=label_font_size, style=label_style, color="black"))
            if adjust_labels:
                adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

        # Tighten the axis
        if squish_scales:
            # do_common_args will override these, so don't worry
            dx = (max(xdata) - min(xdata)) * 0.05
            dy = (max(ydata) - min(ydata)) * 0.05
            ax.set_xlim([min(xdata)-dx, max(xdata)+dx])
            ax.set_ylim([min(ydata)-dy, max(ydata)+dy])

        if perc_weights is not None: # perc_weights is often a numpy array
            ax.set_xlabel("%s%s (%.1f%%)" % (mode, x, perc_weights[x-1])) # can be overridden via do_common_args()
            ax.set_ylabel("%s%s (%.1f%%)" % (mode, y, perc_weights[y-1]))
        else:
            ax.set_xlabel("%s%s" % (mode, x)) # can be overridden via do_common_args()
            ax.set_ylabel("%s%s" % (mode, y))

        if "logx" in kargs and kargs["logx"]:
            ax.set_xscale("log", basex=kargs["logx"])
        if "logy" in kargs and kargs["logy"]:
            ax.set_yscale("log", basey=kargs["logy"])

        if cut:
            rect = matplotlib.patches.Rectangle(cut[0:2], cut[2]-cut[0], cut[3]-cut[1], ec="none", alpha=0.2, fc="orange")
            ax.add_patch(rect)

            tdata = []
            for i in range(0, len(xdata)):
                if xdata[i] > cut[0] and xdata[i] < cut[2]:
                    if ydata[i] < cut[1] and ydata[i] > cut[3]:
                        if self.rowwise: # grab the full entry from the parent genelist
                            dat = {"pcx": xdata[i], "pcy": ydata[i]}
                            dat.update(self.parent.linearData[i])
                            tdata.append(dat)
                        else:
                            tdata.append({"name": lab[i], "pcx": xdata[i], "pcy": ydata[i]})
            if tdata:
                ret_data = genelist()
                ret_data.load_list(tdata)

        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        self.do_common_args(ax, **kargs)

        real_filename = self.savefigure(fig, filename)
        config.log.info("scatter: Saved '%s%s' vs '%s%s' scatter to '%s'" % (mode, x, mode, y, real_filename))
        return ret_data

    def dotbarplot(self,
        data,
        filename:str,
        yticktitle='Number',
        draw_stds:bool = True,
        **kargs):
        """
        **Purpose**
            A drawing wrapper for the new style dot-mean-stderr plots.
            Each sample is a circle, the mean is a red line and the std err of the mean
            is shown where available.

            This version also supports heterogenous lengths of data.

        **Arguments**
            data (Required)
                A dictionary of {label: [0, 1, 2, ... n]} values.

                The x category labels will be taken from the dict key.

            order (Optional, default=data.keys())
                order for the conditions to be plotted

            filename (Required)
                filename to save the image to

            yticktitle (Optional default='Number')
                Added here just to emphasise you should really set your own y ticklabel.

        **Returns**
            The filename the data was saved to.

        """
        assert filename, 'You must specify a filename'
        assert isinstance(data, dict), 'data must be a dictionary'

        # ax 2:
        fig = self.getfigure(**kargs)
        fig.subplots_adjust(left=None, bottom=0.35, right=None, top=None, wspace=0.3, hspace=None)
        ax = fig.add_subplot(111)
        barh = list(data.values())
        labs = list(data.keys()) # Dicts are now ordered, so lab order should be fine.

        # Get the means/stdevs:
        means = [numpy.mean(data[k]) for k in data]
        if draw_stds:
            stds = [numpy.std(data[k]) / math.sqrt(len(data[k])) for k in data]
        lengths = [len(data[k]) for k in data] # for working out if it is valid to plot errs or mean

        # convert the data array into a linear list of x and y:
        xs = numpy.arange(0.5, len(labs)+0.5)
        xd = []
        yd = []
        for i, k in enumerate(data):
            for d in data[k]:
                xd.append(xs[i])
                yd.append(d)


        ax.scatter(xd, yd, edgecolors='black', lw=0.5, c='none', s=15)
        if False not in [i>=3 for i in lengths]:
            if draw_stds:
                ax.errorbar(xs, means, yerr=stds, barsabove=True, fmt='none', capsize=4, capthick=0.5, ls='-', color='black', lw=0.5)
            else:
                ax.bar(xs, means, ls='-', color='black', lw=0.5)

        #ax.set_ylim([0, max(yd)+10])
        ax.set_xlim([-0.2, len(labs)+0.2])
        ax.set_xticks(xs)
        ax.set_xticklabels(list(labs))
        ax.set_xticklabels(labs, rotation=45, rotation_mode="anchor", ha="right")
        ax.set_ylabel(yticktitle)

        #if False not in [i>=2 for i in lengths]:
        for i in zip(xs, means):
            l = mlines.Line2D([i[0]-0.3, i[0]+0.3], [i[1], i[1]], c='red', lw=1)
            ax.add_line(l) #ax.scatter(xs, ms, color='red', marker='_')

        self.do_common_args(ax, **kargs)

        real_filename = self.savefigure(fig, filename)
        config.log.info("dotbarplot: Saved dotbarplot to '%s'" % (real_filename))
        return real_filename

    def proportional_bar(self,
        filename,
        data_dict,
        key_order=None,
        title='',
        cols=None,
        **kargs):
        '''
        **Purpose**
            Draw a bar plot, but with proporional bars.

        **Arguments**
            filename (Required)
                filename to save the figure to.

            data_dict (Required)
                {
                'row_name1': {'class1': 0, 'class2': 0},
                'row_name2': {'class1': 0, 'class2': 0},
                }

            key_order (Optional)
                order for the row_names;

            ...

        '''
        #assert filename, 'A filename to save the image to is required'
        assert data_dict, 'data_dict is required'
        assert isinstance(data_dict, dict), 'data_dict is not a dict'

        if not cols:
            cols = plot.rcParams['axes.prop_cycle'].by_key()['color']

        # get all of the classes:
        if not key_order:
            all_keys = [] # preserve order
            for k in data_dict:
                for kk in data_dict[k]:
                    if kk not in all_keys:
                        all_keys.append(kk)
            config.log.info(f'Found {all_keys} keys')
        else:
            all_keys = key_order

        vals = {k: [] for k in all_keys}

        labs = []
        for k in data_dict:
            labs.append(k)
            for kk in all_keys:
                vals[kk].append(float(data_dict[k][kk]))

        scaled = {k: [] for k in all_keys}
        sums = None
        for k in all_keys:
            if sums is None:
                sums = numpy.zeros(len(vals[k]))
            sums += vals[k]

        for k in all_keys:
            vals[k] = numpy.array(vals[k])
            scaled[k] = numpy.array(vals[k])
            scaled[k] /= sums
            scaled[k] *= 100

        plot_hei = (0.8) - (0.04*len(labs))

        plot.rcParams['pdf.fonttype'] = 42
        fig = plot.figure(figsize=[4,3])
        fig.subplots_adjust(left=0.35, right=0.95, bottom=plot_hei,)
        ax = fig.add_subplot(111)
        ax.set_prop_cycle('color', cols)

        ypos = numpy.arange(len(data_dict))

        # data_dict = {'bar_row': {'class': 0, class2': 0}}

        bots = numpy.zeros(len(labs))
        for k in vals:
            ax.barh(ypos, scaled[k], 0.7, label=k, left=bots)
            for y, v, s, b in zip(ypos, vals[k], scaled[k], bots):
                ax.text(b+(s//2), y, '{0:,.0f} ({1:.0f}%)'.format(v, s), ha='center', va='center', fontsize=6)
            bots += scaled[k]

        ax.set_yticks(ypos)
        ax.set_yticklabels(labs)

        ax.set_xlim([-2, 102])
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(['0%', '50%', '100%'])
        ax.set_title(title, size=6)
        ax.grid(False)
        ax.legend()
        plot.legend(loc='upper left', bbox_to_anchor=(0.0, -0.4), prop={'size': 6})
        [t.set_fontsize(6) for t in ax.get_yticklabels()]
        [t.set_fontsize(6) for t in ax.get_xticklabels()]

        self.do_common_args(ax, **kargs)

        real_filename = self.savefigure(fig, filename)
        config.log.info("proportional_bar: Saved '{0}'".format(real_filename))
        return real_filename

    def boxplots_vertical(self,
        filename,
        data_as_list,
        data_labels,
        qs=None,
        p_threshold:float =0.01,
        title=None,
        xlims=None,
        sizer:float =0.022,
        vert_height=4,
        cols='lightgrey',
        bot_pad:float =0.1,
        showmeans=False,
        **kargs):

        assert filename, 'A filename to save the image to is required'

        plot.rcParams['pdf.fonttype'] = 42

        mmheat_hei = 0.1+(sizer*len(data_as_list))

        fig = self.getfigure(**kargs)
        fig.subplots_adjust(left=0.4, right=0.8, top=mmheat_hei, bottom=bot_pad)
        ax = fig.add_subplot(111)
        ax.tick_params(right=True)

        r = ax.boxplot(
            data_as_list,
            showfliers=False,
            whis=True,
            patch_artist=True,
            widths=0.5,
            vert=False,
            showmeans=showmeans,
            meanprops={'marker': 'o', 'markerfacecolor':'black',
                       'markeredgecolor':'black',
                       'markersize':'3'})

        #print([i.get_data() for i in r['medians']])

        plot.setp(r['medians'], color='black', lw=2) # set nicer colours
        if showmeans: plot.setp(r['means'], color='black', lw=2)
        plot.setp(r['boxes'], color='black', lw=0.5)
        plot.setp(r['caps'], color="grey", lw=0.5)
        plot.setp(r['whiskers'], color="grey", lw=0.5)

        ax.set_yticks(numpy.arange(len(data_labels))+1)
        ax.set_yticklabels(data_labels)

        xlim = ax.get_xlim()[1]
        if xlims:
            ax.set_xlim(xlims)
            xlim = xlims[1]

        if qs:
            for i, p in zip(range(0, len(data_as_list)), qs):
                if p < p_threshold:
                    ax.text(xlim+(xlim/12), i+1, '*', ha='left', va='center', fontsize=6,)
                ax.text(xlim+(xlim/8), i+1, f'{p:.1e}', ha='left', va='center', fontsize=6,)

        if isinstance(cols, list):
            for i, k, b in zip(range(0, len(data_as_list)), data_as_list, r['boxes']):
                b.set_facecolor(cols[i])
        else:
            for i, k, b in zip(range(0, len(data_as_list)), data_as_list, r['boxes']):
                b.set_facecolor(cols)

        if title:
            ax.set_title(title, fontsize=6)

        [t.set_fontsize(6) for t in ax.get_yticklabels()]
        [t.set_fontsize(6) for t in ax.get_xticklabels()]

        self.do_common_args(ax, **kargs)

        fig.savefig(filename)

        real_filename = self.savefigure(fig, filename)
        return real_filename
