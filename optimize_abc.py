"""
"""
from scipy import interpolate
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set the following loglevel
# to suppress unnecessary warnings
# from saving transparent artists as .eps
matplotlib.set_loglevel("critical")

# Adjust the plot font size.
font = {"family":"normal",
        "weight":"normal",
        "size": 15}

matplotlib.rc("font", **font)

def rk4(t0, y0, tf, f, h, **kwargs):
    '''
    Fourth order Runge-Kutta integrator for the arbitrary function
    f=f(t,y,**kwargs), where kwargs are the variables necessary to
    completely define the right-hand-side of the differential equation
    dy/dt = f(t,y; **kwargs)

    Parameters
    ----------
    t0 : float
        Moment in time to begin solution.
    y0 : float
        Initial value at t = t0.
    tf : float
        Final time.
    f : function
        Function which defines the DE to solve.
    h : float
        Time step of the method (s).
    **kwargs : dict
        Keyword arguments to completely define the right-hand-side of
        the DE.

    Returns
    -------
    T : array
        Time array.
    Y : array
        Function value array.

    '''
    
    # h = d["h"] # Time step (units of s) for the RK4 integrator. 
    Y = []
    T = []
    yn = y0
    tn = t0
    
    while tn <= tf-h:
        #
        # Update time T and solution Y
        #
        T.append(tn)
        Y.append(np.float(yn))
        #
        # Calculate new coefficients k
        #
        k1 = h*f(tn, yn, **kwargs)
        k2 = h*f(tn+0.5*h, yn + 0.5*k1, **kwargs)
        k3 = h*f(tn+0.5*h, yn + 0.5*k2, **kwargs)
        k4 = h*f(tn+h, yn + k3, **kwargs)
        #
        # Calculate next solution value and time 
        #
        yn += (1/6)*(k1 + 2*k2 + 2*k3 + k4)                  # Update yn
        tn += h                                              # Update tn
    
    return np.array(T),np.array(Y)


def RHS(t,y,a,b,c,I_lo,I_hi):
    '''
    Right-Hand-Side of the DE, which determines the time-evolution
    of the ion extraction current.

    Takes the a,b,c parameters and the interpolate objects
    for I_lo and I_hi (i.e. measurement data for I^q-1 & I^q+1) 
    as arguments.

    Returns the instantaneous value of dy/dt = RHS.

    Use with rk4 to solve for y(t).
    '''

    return a*I_lo(t) - b*y + c*I_hi(t)


class OptimizeABC(object):
    """docstring for OptimizeABC"""
    def __init__(self, parameters):
        super(OptimizeABC, self).__init__()
        self.parameters = parameters


        # Generate paths to the parsed n+ data files
        # to use in interpolation.
        self.generate_parsed_filepaths()


        # Generate the interpolate objects 
        # out of the n+ signals 
        # for fitting.
        self.generate_interpolate_objects()


    def generate_parsed_filepaths(self):
        """
        The parsed n+ signals have a hardcoded 
        naming convention, which is determined 
        in parse_raw_data.py.
        Generate here the list of filepaths
        to the parsed data files.
        These paths will be used by the script
        to locate said data files.

        The data files themselves are created
        by the parse_raw_data.py-routine. 
        The naming conventions here and there
        must, therefore, match.
        """
        element = self.parameters["injected_species"].lower()
        nPlusOutConvention = "{}{}+.csv"
        self.nPlusFilepathDict = {}
        for charge_state in self.parameters["measured_charge_states"]:
            n_plus_filename = nPlusOutConvention.format(element, charge_state)
            nPlusPath = self.parameters["results_directory"] + n_plus_filename
            self.nPlusFilepathDict[charge_state] = nPlusPath


    def generate_interpolate_objects(self):
        """
        Create interpolate objects out of the 
        n+ signals so that they can be fed to the
        rk4 integrator.

        Add said interpolate objects into 
        a dictionary for ease of access.
        """
        self.interpolateObjectDict = {}

        # Generate paths to the parsed n+ data files
        # to use in interpolation.
        nPlusFilepathDict = self.nPlusFilepathDict

        # Read the data for each charge state
        # and create an interpolation object,
        # which is then stored into the 
        # corresponding dictionary.
        for charge_state in self.parameters["measured_charge_states"]:
            path = nPlusFilepathDict[charge_state]
            data = pd.read_csv(path)
            t, i = data["t"], data["i"]
            interpolateObject = interpolate.interp1d(t,i,kind="cubic")
            self.interpolateObjectDict[charge_state] = interpolateObject


    def determine_interpolation_limits(self, charge_state):
        '''
        Determines the interpolation limits 
        for three consecutive transients,
        around the central charge state (charge_state)
        with t_i set to 0 s, always.
        '''    
        data_lo = pd.read_csv(self.nPlusFilepathDict[charge_state-1])
        t_lo, i_lo = data_lo["t"], data_lo["i"]
        data_mid = pd.read_csv(self.nPlusFilepathDict[charge_state])
        t_mid, i_mid = data_mid["t"], data_mid["i"]
        data_hi = pd.read_csv(self.nPlusFilepathDict[charge_state+1])
        t_hi, i_hi = data_hi["t"], data_hi["i"]

        t_i = 0
        t_f = min( max(t_lo), max(t_hi), max(t_mid) )
    
        return t_i, t_f


    def fitting_function(self, t, a, b, c, charge_state):
        """
        """

        # Choose the interpolate objects (q-1 and q+1 signals)
        I_lo = self.interpolateObjectDict[charge_state-1]
        I_hi = self.interpolateObjectDict[charge_state+1]

        t_i, t_f = self.determine_interpolation_limits(charge_state)

        T,Y = rk4(
            t0=t_i,
            y0=0, 
            tf=t_f, 
            f=RHS, 
            h=self.parameters["optimize_abc"]["h"],
            a=a,
            b=b,
            c=c,
            I_lo=I_lo,
            I_hi=I_hi
            )

        y = interpolate.interp1d(T,Y,kind="cubic",fill_value="extrapolate")
        
        return y(t)


    def doOptimizeABC(self):
        """
        """

        # Create a dataframe for the output
        result_dataframe = pd.DataFrame()

        # Perform the optimization via the RK4-method
        # on all possible charge states.
        for charge_state in self.parameters["measured_charge_states"][1:-1]:

            # Print status
            print("Running abc optimization for charge state {}+".format(str(charge_state)))

            
            # Get the data to fit to
            data = pd.read_csv(self.nPlusFilepathDict[charge_state])
            t, i = data["t"], data["i"]
            t_i, t_f = self.determine_interpolation_limits(charge_state)
            xdata, ydata = t[(t>t_i)&(t<t_f)], i[(t>t_i)&(t<t_f)]

            # Determine the noise level
            noise = np.std(i[t<t_i])*np.ones(len(xdata))

            # Fix the charge state for the curve_fit
            f = lambda t, a, b, c: self.fitting_function(t,a,b,c,charge_state=charge_state)

            # Do the curve fit
            lo_bnds = [0,0,0]
            hi_bnds = [np.inf,np.inf,np.inf]
            bnds = (lo_bnds, hi_bnds)
            p0 = [1000,1000,100]
            popt,pcov = curve_fit(f=f,
                                  xdata=xdata,
                                  ydata=ydata,
                                  p0=p0,
                                  sigma=noise,
                                  absolute_sigma=True,
                                  method="trf",
                                  bounds=bnds)  
            
            # Calculate the chi2
            chi2 = np.sum( np.square( (ydata - f(xdata, *popt))/noise ) )
            
            # Calculate the (reduced) chi2
            chi2_reduced = chi2 / (len(ydata)-3)

            # Unpack the result to lists
            [a, b, c] = popt
            [da, db, dc] = np.sqrt(np.diag(pcov))

            # Add lists to result_dataframe
            result_dataframe[charge_state]=[a,b,c,da,db,dc,chi2_reduced]


            # Save the fit in a DataFrame
            T = np.linspace(t_i, t_f, num=len(t))
            Y = f(t=T, a=a, b=b, c=c)

            fit_dataframe = pd.DataFrame()
            fit_dataframe["t"] = T
            fit_dataframe["i"] = Y
            
            # Output the fit data to .csv
            h = self.parameters["optimize_abc"]["h"]
            outputFileName = "out_fit_h{:.0e}_q{}+.csv".format(h, str(charge_state))
            fit_dataframe.to_csv(self.parameters["results_directory"]  + outputFileName)

            # Make a plot of the final fit
            outName = "fig_fit_h{:.0e}_q{}+".format(h, str(charge_state))     
            fig, ax = plt.subplots()
            ax.plot(t*1e3,i,c="k", label="Measured")
            ax.plot(T*1e3,Y,c="r", ls="--", label="Fitted")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(r"Current ($e$A)")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.legend()
            fig.tight_layout()    
            fig.savefig(self.parameters["results_directory"] + outName + ".eps", format="eps", transparent=False)
            fig.savefig(self.parameters["results_directory"] + outName + ".png", format="png", dpi=300)
            plt.close(fig)



        result_dataframe.index=["a", "b", "c", "da", "db", "dc", "chi2"]
        result_dataframe.to_csv(self.parameters["results_directory"] + "abc.csv")



            





