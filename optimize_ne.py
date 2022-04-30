import pandas as pd
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import multiprocessing
import sys
from os import path
import random



@njit()
def generate_E_array(N, lower, upper):
    """
    Generate an array of average energies
    with each point randomly displaced.
    """
    array = np.logspace(np.log10(lower), np.log10(upper), N)
    for i in range(N):
        if not i == 0 and not i == len(array)-1:
            array[i] += random.uniform(-array[i]+array[i-1], array[i+1]-array[i])
    return array


def calculate_tau(n, a, b, c_l, rc_lo, rc):
    """
    Calculate the confinement time

    Expects pre-biased rate coefficient values.
    """
    return (b - n*rc - (a*c_l/(n*rc_lo)))**(-1)

def penalty_function(n, q, a, a_h, b, b_h, c_l, c, rc_l, rc, rc_h):
    """
    Expects the rate coefficient values to be pre-biased
    for the monte carlo iteration.

    Give n in base 10.
    """

    LHS = (q/(q+1))*(a_h/(n*rc))
    tau = calculate_tau(n, a=a, b=b, c_l=c_l, rc_lo=rc_l, rc=rc)
    tau_h = calculate_tau(n, a=a_h, b=b_h, c_l=c, rc_lo=rc, rc=rc_h)
    RHS = tau/tau_h
    # print(100*np.abs(LHS-RHS)/LHS)
    return 100*np.abs(LHS-RHS)/LHS

def minimize_penalty_function(q, a, a_h, b, b_h, c_l, c, rc_l, rc, rc_h, n_lo, n_hi):
        """
        """

        result = minimize_scalar(
            fun=penalty_function,
            args=(q, a, a_h, b, b_h, c_l, c, rc_l, rc, rc_h),
            bounds=[n_lo, n_hi],
            method="bounded"
            )
        # result = minimize(
        #     fun = penalty_function,
        #     args = (q, a, a_h, b, b_h, c_l, c, rc_l, rc, rc_h),
        #     bounds = [(n_lo, n_hi)],
        #     method = "SLSQP",
        #     x0 = (n_lo + n_hi)*.5
        #     )

        return result


class OptimizeNE(object):
    """docstring for OptimizeNE"""
    def __init__(self, parameters):
        super(OptimizeNE, self).__init__()
        self.parameters = parameters

        # Electron density limits
        self.n_lo = float(parameters["optimize_nE"]["electron_density_limits"][0])
        if type(parameters["optimize_nE"]["electron_density_limits"][1]) == list:
            f = float(parameters["optimize_nE"]["electron_density_limits"][1][1])
            self.n_hi = self.calculate_cutoff_density(f)
        else:
            self.n_hi = float(parameters["optimize_nE"]["electron_density_limits"][1])

        # Energy limits
        self.E_lo = self.parameters["optimize_nE"]["average_energy_limits"][0]
        self.E_hi = self.parameters["optimize_nE"]["average_energy_limits"][1]

        # Load the abc parameters into a dataframe.
        abc_path = parameters["results_directory"] + "abc.csv"
        self.abc = pd.read_csv(abc_path,index_col=0)

        # Pre-load the (interpolation) functions for the (inz) rate coefficients
        self.rate_coefficient_dict = {}
        for charge_state in parameters["measured_charge_states"]:
            self.rate_coefficient_dict[charge_state] = self.get_rate_coefficient_function(charge_state)

        # Create a dictionary for the biases.
        # This will be updated as the script progresses.
        # Each MC iteration generates a new set of biases.
        # Accessed though "self.bias_dict[charge_state]"
        self.bias_dict = {}
        self.generate_rate_coefficient_biases()

        # To keep track of current MC iteration.
        # Needed to access the biases.
        self.iteration = 0


    def calculate_cutoff_density(self,f):
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


    def get_rate_coefficient_function(self, charge_state):
        """
        Use this function to pre-load the (interpolation) functions
        with which to calculate the necessary rate coefficients
        during the optimization. This speeds up the code performance,
        when you don't have to load the function over and over again.
        """

        # Find the interpolation functions within the
        # rate coefficient data directory
        # -------------------------------------------    
        s1 = "".join(
            (
            self.parameters["working_directory"],
            "rate_coefficient_data/",
            self.parameters["injected_species"],
            "/"
            )) 

        s2 = "".join(
            (
            self.parameters["optimize_nE"]["rate_coefficient_method"].upper(),
            "_",
            self.parameters["injected_species"],
            "_",
            str(charge_state),
            "+.npy"
            ))

        fpath = "".join((s1, s2))
        if not path.exists(fpath):
            print("Couldn't find file {}".format(fpath))
            print("Exiting...")
            sys.exit()

        

        # Load the interpolation function
        # -------------------------------
        f = np.load(fpath, allow_pickle=True).item()
        
        return f



    def generate_rate_coefficient_biases(self):
        """
        Generates a dictionary which contains 
        a list of biases for each charge state
        which are used to simulate the effect of
        the experimental uncertainty of the 
        rate coefficients (originally from experimental
        uncertainty of the cross section data).
        """

        # Load the experimental uncertainties 
        # into a dictionary for easy call-up
        # during the script runtime.
        uncertainty_dict = {}
        s = (self.parameters["working_directory"],
            "rate_coefficient_data/",
            self.parameters["injected_species"],
            "/uncertainty_",
            self.parameters["injected_species"],
            ".csv"
            )
        fpath = "".join(s)
        df = pd.read_csv(fpath)
        for charge_state in self.parameters["measured_charge_states"]:
            c = (df["state"]==charge_state)
            uncertainty = df[c]["UNC"].values[0]
            uncertainty_dict[charge_state] = uncertainty

        # Generate a list of biases for 
        # each measured charge state.
        # Store in a dictionary object.
        self.bias_dict = {}
        N = self.parameters["optimize_nE"]["number_of_MC_iterations"]
        for charge_state in self.parameters["measured_charge_states"][1:-1]:
            unc = uncertainty_dict[charge_state]
            # Create a list of biases by a uniform random selection
            # within the bounds dictacted by the experimental uncertainty.
            biases = np.random.uniform(low=-unc/100, high=unc/100, size=N) + 1
            self.bias_dict[charge_state] = biases

        # Output the bias dictionary as .csv
        df = pd.DataFrame(self.bias_dict)
        s = (self.parameters["results_directory"],
            "biases.csv")
        fpath = "".join(s)
        df.to_csv(fpath)


    def rate_coefficient(self, charge_state, average_energy):
        """
        Average energy of the EED is given in eV.
        """
        # Note conversion m3 -> cm3
        return self.rate_coefficient_dict[charge_state](average_energy)*1e6 


    def MC_iteration(self, q):
        """
        """
        
        res = {}

        # Load abc parameters
        # -------------------
        a = self.abc[str(q)]["a"]
        a_h = self.abc[str(q+1)]["a"]
        b = self.abc[str(q)]["b"]
        b_h = self.abc[str(q+1)]["b"]
        c_l = self.abc[str(q-1)]["c"]
        c = self.abc[str(q)]["c"]

        # Generate the energy array
        # -------------------------
        N = self.parameters["optimize_nE"]["number_of_E"]
        E_array = generate_E_array(N=N, lower=self.E_lo, upper=self.E_hi)

        # Calculate the rate coefficient values at each energy
        # Apply the bias coefficients at this stage.
        # ----------------------------------------------------
        bias_l = self.bias_dict[q-1][self.iteration]
        bias = self.bias_dict[q][self.iteration]
        bias_h = self.bias_dict[q+1][self.iteration]
        rc_l_array = np.array([self.rate_coefficient(q-1, e)*bias_l for e in E_array])
        rc_array = np.array([self.rate_coefficient(q, e)*bias for e in E_array])
        rc_h_array = np.array([self.rate_coefficient(q+1, e)*bias_h for e in E_array])

        # Limits
        # ------
        n_lo = self.n_lo
        n_hi = self.n_hi


        # Minimize penalty function at each E.
        # If solution is found and it is 
        # physical, append to return lists.
        # ------------------------------------
        n_list = []
        E_list = []
        tau_list = []
        inz_rate_list = []
        cx_rate_list = []
        F_list = []
        ec_list = []
        iteration_list = []
        for i in range(N):

            E = E_array[i]
            rc_l = rc_l_array[i]
            rc = rc_array[i]
            rc_h = rc_h_array[i]
            result = minimize_penalty_function(q, a, a_h, b, b_h, c_l, c, rc_l, rc, rc_h, n_lo, n_hi)

            # Check result success, 
            # pack values into result dictionary 'res'
            # and return the result.
            if result.success == True:

                n = result.x # Density corresponding to penalty function minimum
                # print("{:.2e}".format(n))

                # Compute values corresponding to obtained (n,E)-pair
                # ---------------------------------------------------
                F = result.fun
                tau = calculate_tau(n, a=a, b=b, c_l=c_l, rc_lo=rc_l, rc=rc)
                tau_h = calculate_tau(n, a=a_h, b=b_h, c_l=c, rc_lo=rc, rc=rc_h)
                inz_rate = rc*n
                cx_rate = b - inz_rate - 1/tau
                energy_content = E*n

                # Check physicality (i.e. that all values are positive)
                if tau > 0 and tau_h > 0 and inz_rate > 0 and cx_rate > 0 and energy_content > 0:
                    
                    # If solution is physical
                    # it is accepted as a point in the solution set.

                    n_list.append(n)
                    E_list.append(E)
                    tau_list.append(tau)
                    inz_rate_list.append(inz_rate)
                    cx_rate_list.append(cx_rate)
                    ec_list.append(energy_content)
                    F_list.append(F)
                    iteration_list.append(self.iteration)
        
        # Create the result dictionary
        # ----------------------------
        res["n"] = n_list
        res["E"] = E_list
        res["tau"] = tau_list
        res["inz_rate"] = inz_rate_list
        res["cx_rate"] = cx_rate_list
        res["eC"] = ec_list
        res["F"] = F_list
        res["iteration"] = iteration_list

        # Increment the MC iterations
        # to move to next bias.
        # ---------------------------
        self.iteration += 1

        return res


    def doOptimizeNE(self):
        """
        For each charge state possible:
        
        1. Repeat the process of finding the solution set
        for a number of Monte Carlo iterations.
        
        2. Save all solution sets thus obtained.
        """
        
        for charge_state in self.parameters["measured_charge_states"][2:-2]:
            sys.stdout.write("\033[K")
            print("Starting on charge state {}+".format(str(charge_state)))

            # Lists to gather the solutions
            ne = []
            Ee = []
            tau = []
            inz_rate = []
            cx_rate = []
            eC = []
            F = []
            i = [] 

            # Iteration tracks the biasing factors to use. Set to 0 at start.
            # ---------------------------------------------------------------
            self.iteration = 0
            for _ in range(self.parameters["optimize_nE"]["number_of_MC_iterations"]):
                
                sys.stdout.write("\033[K")
                print("Processed MC iteration #{} out of {}".format(self.iteration + 1,
                    self.parameters["optimize_nE"]["number_of_MC_iterations"]))
                sys.stdout.write("\033[F")

                result = self.MC_iteration(q=charge_state)

                # Append the output lists
                [ne.append(el) for el in result["n"]]
                [Ee.append(el) for el in result["E"]]
                [tau.append(el) for el in result["tau"]]
                [inz_rate.append(el) for el in result["inz_rate"]]
                [cx_rate.append(el) for el in result["cx_rate"]]
                [F.append(el) for el in result["F"]]
                [eC.append(el) for el in result["eC"]]
                [i.append(el) for el in result["iteration"]]

            # Create the output dataframe
            output = pd.DataFrame()
            output["n"] = ne
            output["E"] = Ee
            output["tau"] = tau
            output["inz_rate"] = inz_rate
            output["cx_rate"] = cx_rate
            output["eC"] = eC
            output["F"] = F
            output["iteration"] = i

            # Save to file
            name = "solution_set_{}{}+.csv".format(
                self.parameters["injected_species"],
                str(charge_state)
                )
            fpath = self.parameters["results_directory"] + name
            output.to_csv(fpath)



    