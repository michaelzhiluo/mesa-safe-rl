import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import pylab
import numpy as np

class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
     
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
     
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
     
        return patch



figlegend = pylab.figure(figsize=(15.5,0.85))
bg = np.array([1, 1, 1])  # background of the legend is white
# colors = ["#7776bc", "#aef78e", "#8ff499", "#66a182", "#b7c335", "#be8d39"]
colors = ["#AA5D1F", "#BA2DC1", "#6C2896", "#D43827", "#4899C5", "#34539C"]
colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
figlegend.legend([0, 1, 2, 3, 4, 5], 
				 ['Unconstrained', 'LR', 'RSPO', 'SQRL', 'RP', 'RCPO'],
           handler_map={
               0: LegendObject(colors[0], colors_faded[0]),
               1: LegendObject(colors[1], colors_faded[1]),
               2: LegendObject(colors[2], colors_faded[2]),
               3: LegendObject(colors[3], colors_faded[3]),
               4: LegendObject(colors[4], colors_faded[4]),
               5: LegendObject(colors[5], colors_faded[5]),
               # 6: LegendObject(colors[6], colors_faded[6]),
               # 7: LegendObject(colors[7], colors_faded[7]),
            }, loc='lower right', fontsize=24, ncol=6)
figlegend.savefig('legend.png')

figlegend = pylab.figure(figsize=(14.4,0.85))
bg = np.array([1, 1, 1])  # background of the legend is white
# colors = ["#f88585", "#830404"]
colors = ["#60CC38", "#349C26"]
colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
figlegend.legend([0, 1], 
				 ['Ours: Recovery RL (MF Recovery)', 'Ours: Recovery RL (MB Recovery)'],
           handler_map={
               0: LegendObject(colors[0], colors_faded[0]),
               1: LegendObject(colors[1], colors_faded[1]),
               # 6: LegendObject(colors[6], colors_faded[6]),
               # 7: LegendObject(colors[7], colors_faded[7]),
            }, loc='lower right', fontsize=24, ncol=2)
figlegend.savefig('legend_ours.png')

