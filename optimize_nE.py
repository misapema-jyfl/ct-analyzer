"""
The code expects rate coefficient interpolation functions in units of m3/s.

Output particle densities in units of 1/cm3. Energies in units of eV.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import multiprocessing
import sys
from os import path

class OptimizeNE(object):
    """docstring for Optimizer"""
    def __init__(self, parameters):
        super(OptimizeNE, self).__init__()
        self.parameters = parameters

        # Load the abc parameters into a dataframe
        # for quick access.
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
        self.generate_rate_coefficient_biases()

        # To keep track of current MC iteration.
        # Needed to access the biases.
        self.iteration = 0

    def get_rate_coefficient_function(self, charge_state):
        """
        Use this function to pre-load the (interpolation) functions
        with which to calculate the necessary rate coefficients
        during the optimization. This speeds up the code performance,
        when you don't have to load the function over and over again.

        TODO! Voronov formula (?) 
        """

        # Find the interpolation functions within the
        # rate coefficient data directory
            
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


    def calculate_confinement_time(self, charge_state, average_energy, electron_density):
        """
        Calculate the confinement time of given charge state
        based on the abc parameters.
        """

        # Rename for brevity
        n = electron_density

        # Retrieve the a, b, c values
        a = self.abc[str(charge_state)]["a"]
        b = self.abc[str(charge_state)]["b"]
        c_l = self.abc[str(charge_state-1)]["c"]

        # Calculate necessary rate coefficients
        bias = self.bias_dict[charge_state-1][self.iteration]
        rc_lo = self.rate_coefficient(charge_state-1, average_energy)*bias
        bias = self.bias_dict[charge_state][self.iteration]
        rc = self.rate_coefficient(charge_state, average_energy)*bias

        # Calculate confinement time 
        tau = (b - n*rc - (a*c_l/(n*rc_lo)))**(-1)

        return tau


    def calculate_cx_rate(self, charge_state, average_energy, electron_density):
        """
        Calculate the charge exchange rate of given charge state
        based on the abc parameters.
        """
        b = self.abc[str(charge_state)]["b"]
        tau = self.calculate_confinement_time(
            charge_state, 
            average_energy, 
            electron_density
            )
        # Ionization rate (rc*ne) for the calculation
        bias = self.bias_dict[charge_state][self.iteration]
        ionisation_rate = bias*self.rate_coefficient(charge_state, average_energy)*electron_density

        # Calculate the charge exchange rate
        cx_rate = b - ionisation_rate - 1/tau

        return cx_rate


    def penalty_function(self, average_energy, charge_state, electron_density):
        """
        Penalty function to minimize.
        """
        a_h = self.abc[str(charge_state+1)]["a"]

        # Calculate the ionization rate coefficient
        bias = self.bias_dict[charge_state][self.iteration]
        rc = self.rate_coefficient(charge_state, average_energy)*bias

        LHS = (charge_state/(charge_state+1))*(a_h/(electron_density*rc))
        tau = self.calculate_confinement_time(charge_state, average_energy, electron_density)
        tau_h = self.calculate_confinement_time(charge_state+1, average_energy, electron_density)
        RHS = tau/tau_h

        return 100*np.abs(LHS-RHS)/LHS

    def minimize_penalty_function(self, args):
        """
        Minimize the penalty function for a given electron density
        (and charge state) by varying the average energy of the EED.

        Returns the values corresponding to the found (n,E)-pair
        as a dictionary (if minimization is successful).
        """
        charge_state, electron_density = args[0], args[1]
        
        res = {}

        # Initial guess for average energy
        E_limits = self.parameters["optimize_nE"]["average_energy_limits"]
        E0 = np.random.uniform(low=E_limits[0], high=E_limits[1])

        # Bound the search by average energy limits
        bnds = [(E_limits[0], E_limits[1])]

        # Minimize the penalty function
        result = minimize(
            fun = self.penalty_function,
            x0=E0,
            args=(charge_state, electron_density),
            method="SLSQP",
            bounds=bnds
            )

        # Check result success, 
        # pack values into result dictionary 'res'
        # and return the result.
        if result.success == True:

            E = result.x[0] # Energy corresponding to penalty function minimum for given electron density.

            # Compute values corresponding to obtained (n,E)-pair
            F = self.penalty_function(E, charge_state, electron_density)
            tau = self.calculate_confinement_time(charge_state, E, electron_density)
            tau_h = self.calculate_confinement_time(charge_state+1, E, electron_density)
            bias = self.bias_dict[charge_state][self.iteration]
            inz_rate = bias*self.rate_coefficient(charge_state, E)*electron_density
            cx_rate = self.calculate_cx_rate(charge_state, E, electron_density)
            energy_content = E*electron_density

            # Check physicality (i.e. that all values are positive)
            if tau > 0 and tau_h > 0 and inz_rate > 0 and cx_rate > 0 and energy_content > 0:
                
                res["success"]=True
                res["tau"]=tau
                res["inz_rate"]=inz_rate
                res["cx_rate"]=cx_rate
                res["eC"]=energy_content
                res["F"]=F
                res["ne"]=electron_density
                res["Ee"]=E

                return res

            else:
                res["success"]=False
        else:
            res["success"]=False

        return res

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
        

    def generate_electron_density_array(self):
        """
        Generates the list of electron densities
        over which to iterate the minimization
        of the penalty function.

        The array is created as a logspaced array,
        whose values are then each displaced by a
        random amount (discounting first and last element
        of the array).

        A new electron density array is generated in 
        each Monte Carlo iteration.
        """

        # Pull the lower and upper limits from the parameters
        n_limits = self.parameters["optimize_nE"]["electron_density_limits"]

        # If the upper limit is passed as a list,
        # assume that a calculation of the cutoff
        # density is desired.
        if type(n_limits[1]) == list:
            f = float(n_limits[1][1])
            n_limits[1] = self.calculate_cutoff_density(f)

        # Create the array
        N = self.parameters["optimize_nE"]["number_of_density_iterations"]
        n_limits = [float(n) for n in n_limits]
        array = np.logspace(np.log10(n_limits[0]),np.log10(n_limits[1]),N)
        
        # Displace each element by a random amount
        # towards either the next or previous 
        # element in the array.
        random_factors = np.random.uniform(low=-.5, high=.5, size=N)
        tmp = array
        for i in range(N):
            if i == 0:
                continue
            if i == N-1:
                continue

            r = random_factors[i]
            # Displace towards previous element if r < 0,
            # and towards the next if r > 0.
            if r < 0:
                array[i] = tmp[i] + r*(tmp[i]-tmp[i-1])
            if r >= 0:
                array[i] = tmp[i] + r*(tmp[i+1]-tmp[i])

        return array


    def find_solution_set(self, charge_state):
        """
        Find the set of acceptable solutions (n,E),
        by iterating over an array of electron densities
        and minimizing the penalty function as a function 
        of the average energy of the EED for each 
        value of the electron density.

        Output the characteristic times/rates
        calculated in each acceptable (n,E)-pair.
        """
        
        # Lists to gather the solutions
        ne = []
        Ee = []
        tau = []
        inz_rate = []
        cx_rate = []
        eC = []
        F = []
        i = [] # To track, to which iteration the solution set belongs.

        # Multiprocessing for the penalty function minimization:
        array = self.generate_electron_density_array()
        c = [(charge_state, a) for a in array]
        try:
            if not self.parameters["optimize_nE"]["max_workers"] == -1:
                pool = multiprocessing.Pool(self.parameters["optimize_nE"]["max_workers"])
            else:
                pool = multiprocessing.Pool()
            results = pool.map(self.minimize_penalty_function, c)
        finally:
            pool.close()
            pool.join()

        # Append results to lists
        for res in results:
            if res["success"]==True:
                    ne.append(res["ne"])
                    Ee.append(res["Ee"])
                    tau.append(res["tau"])
                    inz_rate.append(res["inz_rate"])
                    cx_rate.append(res["cx_rate"])
                    F.append(res["F"])
                    eC.append(res["eC"])
                    i.append(self.iteration)
        
        # Return the solution set as a dictionary
        result = {}
        result["ne"]=ne
        result["Ee"]=Ee
        result["F"]=F
        result["tau"]=tau
        result["inz_rate"]=inz_rate
        result["cx_rate"]=cx_rate
        result["eC"]=eC
        result["iteration"]=i

        return result


    def doOptimizeNE(self):
        """
        For each charge state possible:
        
        1. Repeat the process of finding the solution set
        for a number of Monte Carlo iterations.
        
        2. Save all solution sets thus obtained.

        TODO!
        3. Return the pre-generated biasing factors.
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
            i = [] # To track, to which iteration the solution set belongs.

            # Recall that the iteration tracks the biasing factors to use.
            self.iteration = 0
            for _ in range(self.parameters["optimize_nE"]["number_of_MC_iterations"]):
                sys.stdout.write("\033[K")
                print("Processed MC iteration #{} out of {}".format(self.iteration + 1,
                    self.parameters["optimize_nE"]["number_of_MC_iterations"]))
                sys.stdout.write("\033[F")

                result = self.find_solution_set(charge_state=charge_state)

                # Append the output lists
                [ne.append(el) for el in result["ne"]]
                [Ee.append(el) for el in result["Ee"]]
                [tau.append(el) for el in result["tau"]]
                [inz_rate.append(el) for el in result["inz_rate"]]
                [cx_rate.append(el) for el in result["cx_rate"]]
                [F.append(el) for el in result["F"]]
                [eC.append(el) for el in result["eC"]]
                [i.append(el) for el in result["iteration"]]

                self.iteration += 1

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









