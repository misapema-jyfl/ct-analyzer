"""
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter
matplotlib.set_loglevel("critical")

# Set font for plots
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)





class Plotting(object):
    """docstring for PlotSolutionSets"""
    def __init__(self, parameters):
        super(Plotting, self).__init__()
        self.parameters = parameters

    def calculate_cutoff_density(self, f):
        '''
        Calculates the cut-off density for frequency 'f' microwaves.
        Use units of Hz for f.

        Returns the cutoff density in units of 1/cm3.
        '''
        eps0 = 8.8542e-12
        e = 1.6021773e-19
        me = 9.10938356e-31
        cutoff = eps0*me*(2*np.pi*f)**2/e**2
        cutoff *= 1e-6 # Conversion to cm-3
        return cutoff

    def find_confidence_interval(self, list_of_values, condition_percentage):
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

    def determine_max_and_min(self, array):
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

    def normalize_array(self, array):
        """
        Normalizes an NxN array to unity.
        """
        maximum_value, minimum_value = self.determine_max_and_min(array)
        for i in range(len(array)):
            for j in range(len(array)):
                array[i][j] = array[i][j]/maximum_value

        return array

    def get_limits(self):
        """
        """

        # Electron energy limits. 
        # If no constrain is set in plotting parameters
        # use default (maximum) values of [10, 9999].
        E_limits = self.parameters["plotting"]["average_energy_limits"]
        if E_limits[0] == 0:
            E_lo = 10
        else:
            E_lo = np.float(E_limits[0])
        if E_limits[1] == 0:
            E_hi = 9999
        else:
            E_hi = np.float(E_limits[1])
        E_limits = [E_lo, E_hi]


        # Electron density limits. 
        # If no constrain is set in plotting parameters
        # use default values of [1E10, 1E13].
        n_limits = self.parameters["plotting"]["electron_density_limits"]
        if n_limits[0] == 0:
            n_lo = np.float(1E10)
        else:
            n_lo = np.float(n_limits[0])
        
        # If the upper limit is passed as a list,
        # assume that a calculation of the cutoff
        # density is desired.
        if type(n_limits[1]) == list:
            f = float(n_limits[1][1])
            n_hi = self.calculate_cutoff_density(f)
        elif n_limits[1] == 0:
            n_hi = np.float(1E13)
        else:
            n_hi = np.float(n_limits[1])
        n_limits = [n_lo, n_hi]


        # Penalty function upper limit
        F_hi = self.parameters["plotting"]["penalty_function_upper_limit"]
        F_hi = np.float(F_hi)


        limits = {}
        limits["E"] = E_limits
        limits["n"] = n_limits
        limits["F"] = F_hi

        return limits


    def set_limits(self, df):
        """
        Apply desired limits on the dataframe.
        """

        # Get the limits
        limits = self.get_limits()
        
        # Apply the limits
        df = df[df["F"]<limits["F"]]
        df = df[(df["n"]<limits["n"][1])&(df["n"]>limits["n"][0])]
        df = df[(df["E"]<limits["E"][1])&(df["E"]>limits["E"][0])]

        return df


    def get_solution_set(self, charge_state):
        """
        1. Load the solution set data corresponding 
        to given charge state. 
        
        2. Return solution set as a pandas DataFrame().
        """

        #1.
        filename = "solution_set_{}{}+.csv".format(
            self.parameters["injected_species"],
            str(charge_state)
            )
        s = (self.parameters["results_directory"], filename)
        path = "".join(s)

        try:
            df = pd.read_csv(path)
        except:
            print("Couldn't locate solution set file.")
            print("Filepath: {}".format(path))


        #2.
        return df


    def plot_solution_set_heatmap(self, charge_state):
        """
        Plot a heatmap of solution distributions
        for the given charge state.
        """

        df = self.get_solution_set(charge_state)
        df = self.set_limits(df)


        if not len(df) == 0:

            x = df["E"]
            y = df["n"]

            # Calculate the lo_err, median and hi_err
            # for the n and <E> values
            # output to .csv
            # ---------------------------------------
            df_tmp = pd.DataFrame(columns=["minimum", "lo_err", "median", "hi_err", "maximum"], 
                    index=["n", "E"])
            # n
            data = np.array(y)
            lo, median, hi = self.find_confidence_interval(list_of_values=data, condition_percentage=0.341)
            lo_err = median-lo
            hi_err = hi-median
            df_tmp["lo_err"]["n"] = lo_err
            df_tmp["median"]["n"] = median
            df_tmp["hi_err"]["n"] = hi_err
            df_tmp["minimum"]["n"] = min(data)
            df_tmp["maximum"]["n"] = max(data)

            # E
            data = np.array(x)
            lo, median, hi = self.find_confidence_interval(list_of_values=data, condition_percentage=0.341)
            lo_err = median-lo
            hi_err = hi-median
            df_tmp["lo_err"]["E"] = lo_err
            df_tmp["median"]["E"] = median
            df_tmp["hi_err"]["E"] = hi_err
            df_tmp["minimum"]["E"] = min(data)
            df_tmp["maximum"]["E"] = max(data)

            # Output the result
            path = self.parameters["results_directory"]
            element = self.parameters["injected_species"]
            df_tmp.to_csv(path + "result_nE_{}{}+.csv".format(element, str(charge_state)))
            # ---------------------------------------

            # Heatmap data
            # -------------------------------------------------------------------
            
            # Set the plot range (= (n,E)-limits)
            limits = self.get_limits()
            E_limits = limits["E"]
            n_limits = limits["n"]
            
            rng = [ [E_limits[0], E_limits[1]], [n_limits[0], n_limits[1]] ]

            # Distribute (n,E)-pairs into bins
            heatmap, xedges, yedges = np.histogram2d(
                x=x,
                y=y,
                bins=self.parameters["plotting"]["heatmap_bins"],
                range=rng,
                density=True
                )

            

            # Apply Gaussian filter
            # Sigma controls 'coarseness' of heatmap:
            # larger -> smoother
            # smaller -> coarser
            heatmap = gaussian_filter(heatmap, sigma=self.parameters["plotting"]["sigma"]) 

            # Normalize the heatmap
            # ---------------------
            heatmap = self.normalize_array(heatmap)
            
            # A warning about using imshow for plotting a 2d histogram 
            # of x/y values like this: by default, imshow plots the origin
            # in the upper left corner and transposes the image. 
            # What I would do to get the same orientation as a scatter plot
            # is plt.imshow(heatmap.T, extent=extent, origin = 'lower') 
            # – Jamie (Nov 18, 2013 at 13:29)
            img = heatmap.T

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]] 
            # -------------------------------------------------------------------

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
            Ebins = self.parameters["plotting"]["heatmap_bins"]
            bool_E_log = False
            E_scale = "log"
            E_color = "crimson"

            # Electron density axis
            nbins = self.parameters["plotting"]["heatmap_bins"]
            bool_n_log = False
            n_scale = "log"
            n_color = "crimson"
            # ----------------------------------------------

            
            # Plot the heatmap
            # ----------------
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
            xmarg.hist(x, bins=Ebins, color=E_color, log=bool_E_log)
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
            ymarg.hist(y, bins=nbins, orientation="horizontal", color=n_color, log=bool_n_log)
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


            # Save and close figure
            # ---------------------
            path = self.parameters["results_directory"]
            element = self.parameters["injected_species"]
            plt.savefig( path + "solution_set_{}{}+.eps".format(element, str(charge_state)), format="eps")
            plt.savefig( path + "solution_set_{}{}+.png".format(element, str(charge_state)), format="png", dpi=300)
            plt.close()
            # ---------------------
        else:
            print("  Error: No valid solutions in dataset!")
            print("  Characteristic values will be set to 0.")
            print("  Skipping charge state...")


    def plot_solution_sets(self, charge_states):
        """
        Plot the solution set heatmaps for 
        all selected charge states.
        """
        for charge_state in charge_states:
            print("Plotting solution set for charge state {}+...".format(str(charge_state)))
            try:
                self.plot_solution_set_heatmap(charge_state)
            except:
                print("Failed to plot for {}+".format(str(charge_state)))
                print("Check that file exists.")
                print("Continuing...")


    def get_number_of_solutions(self, charge_state, F_upper_limits):
        """
        Get the number of solutions as a function of
        the upper limit of the penalty function.
        """

        # Get the solution set and apply
        # limits to (n,E)-pairs.
        df = self.get_solution_set(charge_state)
        limits = self.get_limits()
        df = df[(df["n"]<limits["n"][1])&(df["n"]>limits["n"][0])]
        df = df[(df["E"]<limits["E"][1])&(df["E"]>limits["E"][0])]

        # Get the number of solutions
        number_of_solutions = []
        for F in F_upper_limits:
            number_of_solutions.append(len(df[df["F"]<F]))

        return number_of_solutions



    def plot_number_of_solutions(self, F_upper_limits, charge_states):
        """
        Plot the number of solutions in a given solution set
        for each charge state into the same figure.
        """
        fig, ax = plt.subplots()
        max_number_of_solutions = 1
        for charge_state in charge_states:
            lbl = "{}+".format(charge_state)
            number_of_solutions = self.get_number_of_solutions(charge_state, F_upper_limits)
            ax.scatter(F_upper_limits, number_of_solutions, label=lbl)
            if max(number_of_solutions) > max_number_of_solutions:
                max_number_of_solutions = max(number_of_solutions)
        
        # Change axis settings
        ax.set_xlabel("Upper limit of penalty function")
        ax.set_ylabel("Number of solutions")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1, top=10*max_number_of_solutions)
        ax.legend(ncol=3)
        fig.tight_layout()

        # Save and close figure
        # ---------------------
        path = self.parameters["results_directory"]
        element = self.parameters["injected_species"]
        plt.savefig( path + "number_of_solutions_vs_F.eps".format(element, str(charge_state)), format="eps")
        plt.savefig( path + "number_of_solutions_vs_F.png".format(element, str(charge_state)), format="png", dpi=300)
        plt.close()
        # ---------------------
        



    


    def plot_characteristic_times(self, charge_states):
        """
        Plot the characteristic times as a function
        of the charge state.
        """
        
        keys = ["confinement_time", "ionization_time", "charge_exchange_time"]

        for key in keys:

            if self.parameters["plotting"]["advanced"]["skip"][key]:
                continue

            lo_errs = []
            hi_errs = []
            medians = []
            minimums = []
            maximums = [] 

            # Create the figure
            fig, ax = plt.subplots()

            # Gather the data
            for charge_state in charge_states:
                # Get the data from the solution set
                # and set the limits.
                df = self.get_solution_set(charge_state)
                df = self.set_limits(df)
                
                if key == "confinement_time":
                    # Get the data from the dataframe
                    data = np.array(df["tau"])*1e3

                    # Settings for the plot
                    lbl = r"$\tau^q$"
                    marker_color = "r"
                    marker = "."
                    # ylabel = r"$\tau^q$ (ms)"
                    ylabel = "Confinement time (ms)"
                    outputname = "characteristic_vs_q_CONF"

                elif key == "ionization_time":
                    # Get the data from the dataframe
                    data = 1e3*np.array(df["inz_rate"])**(-1)

                    # Settings for the plot
                    lbl = r"$[\left\langle\sigma v\right\rangle^{\mathrm{inz}}_{q\to q+1} n_e]^{-1}$"
                    marker_color = "k"
                    marker = "s"
                    # ylabel = r"$[\left\langle\sigma v\right\rangle^{\mathrm{inz}}_{q\to q+1} n_e]^{-1}$ (ms)"
                    ylabel = "Ionisation time (ms)"
                    outputname = "characteristic_vs_q_INZ"

                elif key == "charge_exchange_time":
                    # Get the data from the dataframe
                    data = 1e3*np.array(df["cx_rate"])**(-1)

                    # Settings for the plot
                    lbl = r"$[\left\langle\sigma v\right\rangle^{\mathrm{cx}}_{q\to q-1} n_0]^{-1}$"
                    marker_color = "b"
                    marker = "^"
                    # ylabel = r"$[\left\langle\sigma v\right\rangle^{\mathrm{cx}}_{q\to q-1} n_0]^{-1}$ (ms)"
                    ylabel = "Charge exchange time (ms)"
                    outputname = "characteristic_vs_q_CX"

                elif key == "energy_content":
                    # Get the data from the dataframe
                    data = np.array(df["eC"])

                    # Settings for the plot
                    lbl = r"$n_e\left\langle E_e \right\rangle$"
                    marker_color = "m"
                    marker = "."
                    ylabel = r"$n_e\left\langle E_e \right\rangle$ (eV)"
                    outputname = "characteristic_vs_q_eC"

                # Get the confidence interval
                lo, median, hi = self.find_confidence_interval(data, self.parameters["plotting"]["confidence"])
                lo_err = median-lo
                hi_err = hi-median
                lo_errs.append(lo_err)
                hi_errs.append(hi_err)
                medians.append(median)
                minimums.append(min(data))
                maximums.append(max(data))

            # Plot the data
            ax.errorbar(
                x=np.array(charge_states), 
                y=medians, 
                yerr=[lo_errs, hi_errs],
                fmt="",
                ls="",
                lw=2,
                capsize=8,
                marker=marker,
                color=marker_color,
                label=lbl,
                markersize=13
                )
            if self.parameters["plotting"]["plot_characteristic_minima"]:
                ax.scatter(
                    x=np.array(charge_states),
                    y=minimums,
                    marker=marker,
                    color=marker_color,
                    facecolors="none",
                    s=64
                    )
            N = charge_states[-1] - charge_states[0] + 1
            xticks = [int(x) for x in np.linspace(charge_states[0], charge_states[-1], N)]
            ax.set_xlabel("Charge state")
            ax.set_xticks(xticks)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=charge_states[0]-1, right=charge_states[-1]+1)
            ax.legend()
            fig.tight_layout()
            path = self.parameters["results_directory"]
            fig.savefig(path + outputname + ".png", format="png", dpi=300)
            fig.savefig(path + outputname + ".eps", format="eps")
            plt.close()

            # Output the data to .csv
            df = pd.DataFrame()
            df["charge_state"] = charge_states
            df["minimums"] = minimums
            df["lo_errs"] = lo_errs
            df["medians"] = medians
            df["hi_errs"] = hi_errs
            df["maximums"] = maximums
            df.to_csv(path + outputname + ".csv", index=None)




    def doPlotting(self):
        """
        Generate a heatmap of the solution sets 
        for each available charge state.
        """

        if not self.parameters["plotting"]["advanced"]["override_charge_states"]:
            charge_states = self.parameters["measured_charge_states"][2:-2]
        else:
            charge_states = self.parameters["plotting"]["advanced"]["charge_states"]

        print("Plotting the number of solutions...")
        F_upper_limits = [1E-11, 1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3]
        F_upper_limits = [np.float(F) for F in F_upper_limits]
        self.plot_number_of_solutions(F_upper_limits, charge_states)

        self.plot_solution_sets(charge_states)

        print("Plotting characteristic values...")
        self.plot_characteristic_times(charge_states)




        