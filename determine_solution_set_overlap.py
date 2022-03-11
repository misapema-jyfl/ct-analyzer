"""
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm



# Get the limits TODO! From parameters!
limits = {"F":1E-9, "n":[1E10, 1E13], "E":[10,10000]}
E_limits = limits["E"]
n_limits = limits["n"]


def set_limits(df):
    """
    Apply desired limits on the dataframe.
    """

    # # Get the limits
    # limits = {"F":1E-9, "n":[1E10, 1E13], "E":[10,10000]}
    
    # Apply the limits
    df = df[df["F"]<limits["F"]]
    df = df[(df["n"]<limits["n"][1])&(df["n"]>limits["n"][0])]
    df = df[(df["E"]<limits["E"][1])&(df["E"]>limits["E"][0])]

    return df

def get_histogram(df):
	"""
	"""
	N = 1000

	x = df_1["E"]
	y = df_1["n"]

	xmin = 10 # TODO! Get data from parameters file(?)
	xmax = 9999
	ymin = 1E10
	ymax = 1E13
	rng = [[xmin, xmax],[ymin,ymax]]

	heatmap, xedges, yedges = np.histogram2d(x, y, bins=N, range=rng, normed=True)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	heatmap = gaussian_filter(heatmap, sigma=10)

	return heatmap, extent

def determine_max_and_min(array):
	"""
	Determines the maximum and minimum value 
	from an NxN array.
	"""
	# Determine max and min
	maximum_values = []
	minimum_values = []
	for i in range(len(array)):
		maximum_values.append(max(array[i]))
		minimum_values.append(min(array[i]))
	maximum_value = max(maximum_values)
	minimum_value = min(minimum_values)

	return maximum_value, minimum_value

def normalize_array(array):
	"""
	Normalizes an NxN array to unity.
	"""
	maximum_value, minimum_value = determine_max_and_min(array)
	for i in range(len(array)):
		for j in range(len(array)):
			array[i][j] = array[i][j]/maximum_value

	return array

def get_xy_projections(array):
	"""
	Projects the array to its 
	x and y-axes by taking the sum 
	of values in rows/columns
	of the array.
	"""
	x = []
	y = []
	# Projection to the x-axis
	for i in range(len(array)):
		x.append(sum(array[i]))
	# Projection to the y-axis
	for i in range(len(array)):	
		y.append(sum(array.T[i]))

	return np.array(x), np.array(y)


# Import two solution set files and determine their heatmaps
# ------------------------------------------------------------
filepath_1 = "/home/miha/Work/codes/ct-analyzer/test/results/solution_set_k6+.csv"
filepath_2 = "/home/miha/Work/codes/ct-analyzer/test/results/solution_set_k7+.csv"
df_1 = set_limits(pd.read_csv(filepath_1))
df_2 = set_limits(pd.read_csv(filepath_2))
heatmap_1, extent_1 = get_histogram(df_1)
heatmap_2, extent_2 = get_histogram(df_2)


# Normalize the heatmaps
# ----------------------
heatmap_1 = normalize_array(heatmap_1)
heatmap_2 = normalize_array(heatmap_2)


# Multiply one heatmap by the other
# ---------------------------------
heatmap = heatmap_1*heatmap_2


# Normalize
# ---------
normalize_array(heatmap)



# Generate the figure
# -------------------------------------------------------------------
fig = plt.figure()


# Settings for the subplots to add to the figure
# ----------------------------------------------
# Main heatmap
mainx = 0.15
mainy = 0.275
mainw = 0.7
mainh = 0.6

# Cumulation plot (x-projection)
xmargx = mainx
xmargy = mainy + mainh
xmargw = mainw
xmargh = 0.1

# Cumulation plot (y-projection)
ymargx = mainx + mainw
ymargy = mainy
ymargw = 0.1
ymargh = mainh

# Colorbar
cbaxx = mainx
cbaxy = mainy - 0.15
cbaxw = mainw
cbaxh = 0.02

# Energy-axis
Ebins = 300
bool_E_log = True
E_scale = "log"
E_color = "crimson"

# Electron density axis
nbins = 300
bool_n_log = True
n_scale = "log"
n_color = "crimson"
# ----------------------------------------------



# Get the values for the x and y-axis histograms
# ----------------.-----------------------------
E_1 = df_1["E"].values
E_2 = df_2["E"].values
n_1 = df_1["n"].values
n_2 = df_2["n"].values
x = []
[x.append(e) for e in E_1]
[x.append(e) for e in E_2]
y = []
[y.append(e) for e in n_1]
[y.append(e) for e in n_2]

# Plot the heatmap
# ----------------
img = heatmap.T
extent = extent_1
ax1 = fig.add_axes([mainx, mainy, mainw, mainh])
im = plt.imshow(img, origin="lower",
          cmap = cm.gist_heat,
          extent = extent,
          aspect = "auto",
          interpolation = "none")
# ----------------



# Plot x-projection histogram
# ---------------------------
xmarg = fig.add_axes([xmargx, xmargy, xmargw, xmargh])
xmarg.hist(x, bins=Ebins, color=E_color, log=bool_E_log, density=True)
xmarg.set(xlim=(E_limits[0], E_limits[1]))
xmarg.set(xscale=E_scale)
xmarg.spines["left"].set_visible(False)
xmarg.spines["right"].set_visible(False)
xmarg.spines["top"].set_visible(False)
xmarg.spines["bottom"].set_visible(False)
xmarg.yaxis.set_visible(False)
xmarg.xaxis.set_visible(False)
# ---------------------------


# Plot y-projection histogram
# ---------------------------
ymarg = fig.add_axes([ymargx, ymargy, ymargw, ymargh])
ymarg.hist(y, bins=nbins, orientation="horizontal", color=n_color, log=bool_n_log, density=True)
ymarg.set(ylim=(n_limits[0], n_limits[1]))
ymarg.set(yscale=n_scale)
ymarg.spines["top"].set_visible(False)
ymarg.spines["bottom"].set_visible(False)
ymarg.spines["left"].set_visible(False)
ymarg.spines["right"].set_visible(False)
ymarg.yaxis.set_visible(False)
ymarg.xaxis.set_visible(False)
# ---------------------------


# Plot the colorbar
# ---------------------------
cbax = fig.add_axes([cbaxx, cbaxy, cbaxw, cbaxh])
plt.colorbar(im, cax=cbax, orientation="horizontal")
# ---------------------------


# Set axis settings
# ---------------------------
ax1.set(xlabel=r"$\left\langle E_e\right\rangle$ (eV)",
ylabel=r"$n_e$ (cm$^{-3}$)",
xscale=E_scale,
yscale=n_scale,
xlim=(E_limits[0], E_limits[1]),
ylim=(n_limits[0], n_limits[1]))
# ---------------------------


# Marginal axis labels are unnecessary
# ------------------------------------
xmarg.set_yticklabels([])
xmarg.set_xticklabels([])
xmarg.set_yticks([])
ymarg.set_yticklabels([])
ymarg.set_xticklabels([])
ymarg.set_xticks([])
# ------------------------------------



plt.show()