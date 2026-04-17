import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

textwidth_in = 5.90666
textheight_in = 8.5112
fontsize_pt = 10.95

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=fontsize_pt)
plt.rc('font', serif='Computer Modern')
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage{amsfonts}\usepackage{amsmath}'
plt.rcParams['figure.dpi'] = 300
mpl.rcParams['text.usetex'] = True

plt.rcParams.update({
    'axes.titlesize': 13.14,  # Equivalent to LaTeX \large
    'axes.labelsize': 10.95,  # Equivalent to LaTeX \normalsize
    'legend.fontsize': 10.95,  # Same as document body text
    'xtick.labelsize': 9.48,  # Equivalent to LaTeX \small
    'ytick.labelsize': 9.48,  # Equivalent to LaTeX \small
})

small_fig = (0.6*textwidth_in, 0.6*textwidth_in/1.6)
medium_fig = (0.8*textwidth_in, 0.8*textwidth_in/1.6)
big_fig = (textwidth_in, textwidth_in/1.6)
full_page_fig = (textwidth_in, textheight_in)

def custom_fig_size(width_factor, aspect_ratio):
    return (width_factor * textwidth_in, width_factor * textwidth_in/aspect_ratio)

