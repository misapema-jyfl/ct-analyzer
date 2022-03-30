"""
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
import matplotlib



def determine_max_and_min(array):
	"""
	Determines the maximum and minimum value 
	from an NxN array.
	"""
	# return maximum_value, minimum_value
	return max([max(row) for row in array]), min([min(row) for row in array])


class Overlap(object):
	"""docstring for Overlap"""
	def __init__(self, parameters):
		super(Overlap, self).__init__()
		self.parameters = parameters

		for key, arg in self.parameters["limits"].items():
			if type(arg) == str:
				self.parameters["limits"][key] = np.float(arg)
			elif type(arg) == list:
				self.parameters["limits"][key] = [np.float(a) for a in arg]
		self.parameters["acceptance_threshold"] = np.float(self.parameters["acceptance_threshold"])
		# self.parameters["limits"]["E"][0] = np.float(self.parameters["limits"]["E"][0])
		# self.parameters["limits"]["E"][-1] = np.float(self.parameters["limits"]["E"][-1])
		# self.parameters["limits"]["n"][0] = np.float(self.parameters["limits"]["n"][0])
		# self.parameters["limits"]["n"][-1] = np.float(self.parameters["limits"]["n"][-1])
	
	def normalize_array(self, array):
		"""
		Normalizes an NxN array to unity.
		"""
		maximum_value, minimum_value = determine_max_and_min(array)
		for i in range(len(array)):
			for j in range(len(array)):
				array[i][j] = array[i][j]/maximum_value

		return array

	def set_limits(self, df):
	    """
	    Apply desired limits on the dataframe.
	    """

	    limits = self.parameters["limits"]

	    # Apply the limits
	    df = df[df["F"]<limits["F"]]
	    # df = df[(df["n"]<limits["n"][1])&(df["n"]>limits["n"][0])] # Meaningless in this context
	    # df = df[(df["E"]<limits["E"][1])&(df["E"]>limits["E"][0])] 

	    return df

	def get_histogram(self, df):
		"""
		"""
		N = self.parameters["bins"]
		limits = self.parameters["limits"]

		df = self.set_limits(df)

		x = df["E"]
		y = df["n"]

		# xmin = min(x)
		# xmax = max(x)
		# ymin = min(y)
		# ymax = max(y)

		xmin = np.float(limits["E"][0])
		xmax = np.float(limits["E"][-1])
		ymin = np.float(limits["n"][0])
		ymax = np.float(limits["n"][-1])

		rng = [[xmin, xmax],[ymin,ymax]]

		heatmap, xedges, yedges = np.histogram2d(x, y, bins=N, range=rng, normed=True)
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

		heatmap = gaussian_filter(heatmap, sigma=self.parameters["sigma"])

		result = {}
		result["heatmap"] = self.normalize_array(np.array(heatmap))
		result["extent"] = extent
		result["xedges"] = xedges
		result["yedges"] = yedges
		
		return result

	def find_solutions_in_overlap(self, df, xedges, yedges,
	 overlap_heatmap):
		"""
		"""
		n_overlap = []
		E_overlap = []
		for i in range(len(xedges)-1):
			E_lo = xedges[i]
			E_hi = xedges[i+1] 
			for j in range(len(yedges)-1):
				n_lo = yedges[j]
				n_hi = yedges[j+1]
				if overlap_heatmap[i][j] > 0:
					c = (df["E"]>E_lo)&(df["E"]<E_hi)&(df["n"]>n_lo)&(df["n"]<n_hi)
					df_tmp = df[c]
					E = df_tmp["E"]
					n = df_tmp["n"]
					[E_overlap.append(e) for e in E]
					[n_overlap.append(n) for n in n]
					# TODO! Find characteristic values corresponding to 
					# these n,E simultaneously with this.
		result = {}
		result["n"] = n_overlap
		result["E"] = E_overlap
		
		return result


	def set_non_zero_to_one(self, histogram_data):
		"""
		For visualizing the regions where any number 
		of solutions exist.
		"""
		acceptanceThreshold = self.parameters["acceptance_threshold"]
		for i, row in enumerate(histogram_data):
			for j, element in enumerate(row):
				if element < acceptanceThreshold:
					histogram_data[i][j] = 0
				else:
					histogram_data[i][j] = 1
		return histogram_data




	def plot_heatmap(self, histogram_data, x, y, extent, output_name,
		margin_color):
		"""
		"""

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
		E_limits = self.parameters["limits"]["E"]
		Ebins = self.parameters["bins"]
		E_scale = "log"
		# E_color = self.parameters["margin_color"]

		# Electron density axis
		n_limits = self.parameters["limits"]["n"]
		nbins = self.parameters["bins"]
		n_scale = "log"
		# n_color = self.parameters["margin_color"]
		# ----------------------------------------------



		# Plot the heatmap
		# ----------------
		img = histogram_data.T
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
		xmarg.hist(x, bins=Ebins, color=margin_color, density=True)
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
		ymarg.hist(y, bins=nbins, orientation="horizontal", color=margin_color, density=True)
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

		outDir = self.parameters["results_directory"]
		plt.savefig(outDir + "{}.png".format(output_name), format="png", dpi=300)
		plt.savefig(outDir + "{}.eps".format(output_name), format="eps")
		plt.close()