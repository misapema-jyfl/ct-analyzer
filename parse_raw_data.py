"""
"""
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

class RawDataParser(object):
    """docstring for RawDataParser"""
    def __init__(self, parameters):
        super(RawDataParser, self).__init__()
        self.parameters = parameters
        self.elementName = parameters["injected_species"].lower()

        # Specify the output file naming conventions for 
        # the one plus data and the n+ data.
        self.onePlusOutConvention = "1+_{}{}+.csv" # First {} takes the species
        self.nPlusOutConvention = "{}{}+.csv" # second {} takes the charge state


    def loadData(self, filename):
        """
        Loads a data file into a pandas dataframe.
        """
        path = self.parameters["parse_raw_data"]["path_to_raw_data"] + filename

        try:
            df = pd.read_csv(path, 
                engine="python",
                sep=self.parameters["parse_raw_data"]["separator"], 
                skipfooter=int(self.parameters["parse_raw_data"]["footer_length"]),
                skiprows=int(self.parameters["parse_raw_data"]["header_length"]),
                header=None,
                names=["t", "i"])

            return df

        except:
            print("Couldn't load file {}".format(filename))
            print("Check path: {}".format(path))
            print("Exiting.")
            sys.exit()


    def parseData(self, charge_state, one_plus_filename, n_plus_filename):
        """
        Parse a given pair of data.

        1. The one plus signal (from: one_plus_filename.csv)
        is used to set t=0 for the n+ signal (from: n_plus_filename)
        
        2. The t<0 data from the n+ signal is used to 
        determine (and remove) the background from the n+ signal.
        I.e. the n+ signal corresponding to no 1+ injection.

        3. Injected 1+ signal is normalized to one.
        """
        
        # Open the data files
        dfOne = self.loadData(one_plus_filename)
        dfN = self.loadData(n_plus_filename)
        
        # Get the time series data
        t_1, i_1 = dfOne["t"], dfOne["i"]
        t_n, i_n = dfN["t"], dfN["i"]*self.parameters["parse_raw_data"]["conversion_factor"] # Conversion to A
        
        # Remove time offset, i.e. set 1+ rise onset as t=0
        t_off = t_1[ i_1 > 0.1*max(i_1) ].values[0]
        t_1 = t_1 - t_off
        t_n = t_n - t_off
        
        # Remove background from n+ signal
        bg = np.average(i_n[t_n<0])
        i_n = i_n-bg
        
        # Remove background from 1+ signal
        bg = np.average(i_1[t_1<0])
        i_1 = i_1-bg
        
        # Normalize 1+ command signal
        i_1 = max(i_n)*i_1/max(i_1)

        
        # Pack parsed data to DataFrame and save to .csv
        df_1 = pd.DataFrame([t_1,i_1]).transpose()
        df_n = pd.DataFrame([t_n,i_n]).transpose()
        
        # Set the output directory
        outDir = self.parameters["results_directory"]
        
        # Set the parsed 1+ output name
        s = (outDir, self.onePlusOutConvention.format(self.elementName, charge_state))
        onePlusOutName = "".join(s)
        
        # Set the parsed n+ output name
        s = (outDir, self.nPlusOutConvention.format(self.elementName, charge_state))
        nPlusOutName = "".join(s)
        
        df_1.to_csv(onePlusOutName, index=None)
        df_n.to_csv(nPlusOutName, index=None)
        
        
        # Return the data frames for possible plotting
        return df_1, df_n


    def doParse(self):
        """
        """

        chargeStates = self.parameters["measured_charge_states"]
        
        # Use the naming convention to generate the filenames.
        # (cf. parameters.yaml)
        one_convention = self.parameters["parse_raw_data"]["one_plus_naming_convention"]
        n_convention = self.parameters["parse_raw_data"]["n_plus_naming_convention"]
        variables = self.parameters["parse_raw_data"]["filename_variables"]
        one_plus_filenames = [str(one_convention.format(str(variable))) for variable in variables]
        n_plus_filenames = [str(n_convention.format(str(variable))) for variable in variables]


        # Check that figures have a directory to save to
        s = (self.parameters["results_directory"], "input_data_plots/")
        figuresDirectory = "".join(s)
        if not os.path.isdir(figuresDirectory):
            os.mkdir(figuresDirectory)

        # Parse all given data using the parseData() function.
        # Make plots of the parsed data for checking the output.
        iterList = zip(chargeStates, one_plus_filenames, n_plus_filenames)
        for cState, onePlusFilename, nPlusFilename in iterList:
            
            # Make plots
            fig, ax = plt.subplots()
            
            # Do parsing
            try:            
                df_1, df_n = self.parseData(cState, onePlusFilename, nPlusFilename)
            except:
                print("Something went wrong with parsing charge state: {}+"\
                        .format(cState))
                print("Check input data file.")
                print("Exiting.")
                sys.exit()

            # Plot for sanity check
            ax.plot(df_1["t"], df_1["i"])
            ax.plot(df_n["t"], df_n["i"])
            
            # Determine save to path and save figure
            s = (figuresDirectory, self.elementName,"-", str(cState),"+", ".png")
            savePath = "".join(s)
            fig.savefig(savePath, dpi=300)





