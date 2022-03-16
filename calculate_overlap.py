"""
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import sys
import overlap

# Load the parameters from the .yaml file
# ---------------------------------------
pathToParameterFile = sys.argv[1]
parameters = yaml.safe_load(open(pathToParameterFile)) 

Overlap = overlap.Overlap(parameters)

# Retrieve solution set data
# and generate heatmaps.
# --------------------------
heatmaps = {}
for key, arg in parameters["datasets"].items():
	result = Overlap.get_histogram(pd.read_csv(arg))
	heatmaps[key] = result
	extent = result["extent"]
	print("Non-zero elements in heatmap of {}: {}".format(key, np.count_nonzero(result["heatmap"])))


# Multiply heatmaps by one another
# --------------------------------
N = parameters["bins"]
overlap_heatmap = np.ones((N,N))
for key, arg in heatmaps.items():
	overlap_heatmap = np.multiply(overlap_heatmap, arg["heatmap"]) 

# Set negligible values to zero
# in the overlap set.
# Normalize the heatmap.
# -----------------------------------
Overlap.normalize_array(overlap_heatmap)
for i in range(N):
	for j in range(N):
		if overlap_heatmap[i][j] < parameters["acceptance_threshold"]:
			overlap_heatmap[i][j] = 0
# Overlap.normalize_array(overlap_heatmap)
print("Non-zero elements in overlap: {}".format(np.count_nonzero(overlap_heatmap)))


# Find solutions within the overlap
# ---------------------------------
x, y = [], [] # lists for generating margin histograms
for key, arg in parameters["datasets"].items():
	df = pd.read_csv(arg)
	result = Overlap.find_solutions_in_overlap(df, 
		xedges=heatmaps[key]["xedges"], 
		yedges=heatmaps[key]["yedges"], 
		overlap_heatmap=overlap_heatmap)
	n = result["n"]
	E = result["E"]
	[y.append(n) for n in n]
	[x.append(e) for e in E]

	# Plot the heatmap
	# ----------------
	Overlap.plot_heatmap(histogram_data=heatmaps[key]["heatmap"], 
		x=Overlap.set_limits(df)["E"],
		y=Overlap.set_limits(df)["n"],
		extent=heatmaps[key]["extent"], 
		output_name=key)

print("E_min: {:.0f} eV".format(min(x)))
print("E_max: {:.0f} eV".format(max(x)))
print("n_min: {:.2e} 1/cm3".format(min(y)))
print("n_max: {:.2e} 1/cm3".format(max(y)))

# Plot the overlap heatmap
# ------------------------
Overlap.plot_heatmap(histogram_data=overlap_heatmap, 
	x=x,
	y=y, 
	extent=extent, 
	output_name="overlap")