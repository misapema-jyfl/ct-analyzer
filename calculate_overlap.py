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

print("\nOverlap of:")
for key, arg in parameters["datasets"].items():
	print(key)

print("\nParameters:")
print("--------------------------------------")
print("bins: ", parameters["bins"])
print("sigma: ", parameters["sigma"])
print("acceptance_threshold: {:.2e}".format(parameters["acceptance_threshold"]))
print("--------------------------------------\n")


def find_confidence_interval(list_of_values, condition_percentage):
        '''
        Seek the lower and upper limit in a list of values, which enclose 
        between themselves and the median a given percentage of all values 
        in the list. 
        
    
        Parameters
        ----------
        list_of_values : list
            List of the values, for which the analysis will be carried out.
        condition_percentage : float
            The percentage limit.
    
        Returns
        -------
        x_lo : float
            Lower limit.
        median : float
            Median value.
        x_hi : float
            Upper limit.
    
        '''
        xs = list_of_values
        p = condition_percentage
        
        if not xs.size > 0:
            # print("Given list of values is empty!")
            return 0, 0, 0
        
        else:
            median = np.median(xs)
            
            percentages = []
            x_his = []
            for x in xs:
                if x > median:
                    # Select elements between current element and median
                    interval = xs[(xs<=x)&(xs>=median)]
                    # Calculate percentage of all values within current interval
                    percentage = (len(interval))/len(xs)
                    # If this interval satisfies the condition, add it to list
                    if percentage >= p:
                        percentages.append(percentage)
                        x_his.append(x)
            # Find the minimum percentage satisfying the condition
            # along with the corresponding element.
            percentages=np.array(percentages)
            
            if not percentages.size > 0:
                print("There are no elements within the confidence interval!")
                x_hi = 0
            else:    
                result = np.where(percentages==min(percentages))
                idx = result[0][0]
                x_hi = x_his[idx]
            
            # Find value x_lo, for which p fraction of all results are between 
            # it and the median
            percentages = []
            x_los = []
            for x in xs:
                if x < median:
                    # Select elements between current element and median
                    interval = xs[(xs>=x)&(xs<=median)]
                    # Calculate percentage of all values within current interval
                    percentage = (len(interval))/len(xs)
                    # If this interval satisfies the condition, add it to list
                    if percentage >= p:
                        percentages.append(percentage)
                        x_los.append(x)
            # Find the minimum percentage satisfying the condition
            # along with the corresponding element.
            percentages=np.array(percentages)
            if not percentages.size > 0:
                print("There are no elements within the confidence interval!")
                x_lo = 0
            else:    
                result = np.where(percentages==min(percentages))
                idx = result[0][0]
                x_lo = x_los[idx]
        
        return x_lo, median, x_hi


def retrieve_data():
	"""
	Find the data given in parameters file.
	Generate the histograms.
	"""
	# Retrieve solution set data
	# and generate heatmaps.
	# --------------------------
	heatmaps = {}
	for key, arg in parameters["datasets"].items():
		result = Overlap.get_histogram(pd.read_csv(arg))
		heatmaps[key] = result
	return heatmaps

def generate_overlap(Overlap_object):
	"""
	"""
	# Retrieve solution set data
	# and generate heatmaps.
	# --------------------------
	heatmaps = retrieve_data()

	# Get the extent, xedges and yedges.
	# ----------------------------------
	i = 0
	for key, result in heatmaps.items():
		if i == 0:
			extent = result["extent"]
			xedges = result["xedges"]
			yedges = result["yedges"]
			i += 1
		print("Non-zero elements in heatmap of {}: {}"\
			.format(key, np.count_nonzero(result["heatmap"])))


	# Multiply heatmaps by one another
	# to obtain the overlap
	# --------------------------------
	N = parameters["bins"]
	overlap_heatmap = np.ones((N,N))
	for key, arg in heatmaps.items():
		overlap_heatmap = np.multiply(overlap_heatmap, arg["heatmap"]) 

	# Set values below threshold to zero
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

	# Collect the characteristic values from the overlap
	characteristic_value_dict = {}

	print("\n(n,E)-pairs within the overlap:")
	print("--------------------------------------")
	x, y = [], [] # lists for generating margin histograms
	for key, arg in parameters["datasets"].items():
		df = pd.read_csv(arg)
		result = Overlap.find_solutions_in_overlap(
			df, 
			xedges=xedges, 
			yedges=yedges, 
			overlap_heatmap=overlap_heatmap)
		
		n = result["n"]
		E = result["E"]
		tau = result["tau"]
		inz_rate = result["inz_rate"]
		cx_rate = result["cx_rate"]
		eC = result["eC"]
		F = result["F"]

		# Save the overlap dataframe
		# --------------------------
		outDir = parameters["results_directory"]
		df_tmp = pd.DataFrame(result)
		df_tmp.to_csv(outDir + "overlap_df_" + key + ".csv", index=None)

		# Determine the medians, lo_err and hi_err
		# and minimaof the characteristic values 
		# within the overlap.
		# ------------------------------------------------
		df_tmp = pd.DataFrame(columns=["minimum", "lo_err", "median", "hi_err", "maximum"], 
			index=["tau", "inz_time", "cx_time", "eC", "F", "n", "E"])
		for characteristic_key, data in result.items():
			
			data = np.array(data)

			# If the data corresponds to inz_rate or cx_rate,
			# convert it to inz_time and cx_time instead.
			# Note also conversion s -> ms.
			# -----------------------------------------------
			if characteristic_key == "inz_rate" or characteristic_key == "cx_rate":
				# data = np.power(data, (-1)*np.ones(len(data)))
				data = 1e3*data**(-1)
			if characteristic_key == "tau":
				data = 1e3*data
			if characteristic_key == "inz_rate":
				characteristic_key = "inz_time"
			if characteristic_key == "cx_rate":
				characteristic_key = "cx_time"

			lo, median, hi = find_confidence_interval(list_of_values=data, condition_percentage=0.341)
			lo_err = median-lo
			hi_err = hi-median
			df_tmp["lo_err"][characteristic_key] = lo_err
			df_tmp["median"][characteristic_key] = median
			df_tmp["hi_err"][characteristic_key] = hi_err
			df_tmp["minimum"][characteristic_key] = min(data)
			df_tmp["maximum"][characteristic_key] = max(data)

		df_tmp.to_csv(outDir + "overlap_results_" + key + ".csv")

		[y.append(n) for n in n]
		[x.append(e) for e in E]

		print("{}: {}".format(key, str(len(n))))

		# Plot the heatmap of the input solution set
		# ------------------------------------------
		Overlap.plot_heatmap(histogram_data=heatmaps[key]["heatmap"], 
			x=Overlap.set_limits(df)["E"],
			y=Overlap.set_limits(df)["n"],
			extent=heatmaps[key]["extent"], 
			output_name=key,
			margin_color="crimson")
	print("--------------------------------------")

	# Determine the median, lo_err and hi_err
	# of all the (n,E) pairs within the overlap
	df_tmp = pd.DataFrame(columns=["minimum", "lo_err", "median", "hi_err", "maximum"], 
			index=["n", "E"])
	# n
	data = np.array(y)
	lo, median, hi = find_confidence_interval(list_of_values=data, condition_percentage=0.341)
	lo_err = median-lo
	hi_err = hi-median
	df_tmp["lo_err"]["n"] = lo_err
	df_tmp["median"]["n"] = median
	df_tmp["hi_err"]["n"] = hi_err
	df_tmp["minimum"]["n"] = min(data)
	df_tmp["maximum"]["n"] = max(data)	

	# E
	data = np.array(x)
	lo, median, hi = find_confidence_interval(list_of_values=data, condition_percentage=0.341)
	lo_err = median-lo
	hi_err = hi-median
	df_tmp["lo_err"]["E"] = lo_err
	df_tmp["median"]["E"] = median
	df_tmp["hi_err"]["E"] = hi_err
	df_tmp["minimum"]["E"] = min(data)
	df_tmp["maximum"]["E"] = max(data)	

	# Output the result
	df_tmp.to_csv(outDir + "overlap_nE.csv")

	print("\nOverlap boundaries:")
	print("--------------------------------------")
	print("E_min: {:.0f} eV".format(min(x)))
	print("E_max: {:.0f} eV".format(max(x)))
	print("n_min: {:.2e} 1/cm3".format(min(y)))
	print("n_max: {:.2e} 1/cm3".format(max(y)))
	print("--------------------------------------")

	# Plot the overlap heatmap of all input solution sets
	# ---------------------------------------------------
	Overlap.plot_heatmap(histogram_data=overlap_heatmap, 
		x=x,
		y=y, 
		extent=extent, 
		output_name="overlap",
		margin_color="crimson")


def plot_all_non_zero(Overlap_object):
	"""
	"""
	# Retrieve solution set data
	# and generate heatmaps.
	# --------------------------
	heatmaps = retrieve_data()
	# Set heatmap values to 1 where any 
	# solútions exist
	# ---------------------------------
	for key, result in heatmaps.items():
		heatmap = result["heatmap"]
		heatmap = Overlap.set_non_zero_to_one(histogram_data=heatmap)
		df = Overlap.set_limits(pd.read_csv(parameters["datasets"][key]))
		E = df["E"].values
		n = df["n"].values
		Overlap.plot_heatmap(histogram_data=heatmap,
			x=E,
			y=n,
			extent=result["extent"], 
			output_name="all_solutions_{}".format(key),
			margin_color="gray",
			cbar_type="binary")


generate_overlap(Overlap_object=Overlap)
plot_all_non_zero(Overlap_object=Overlap)


print("\nDone!\n")

