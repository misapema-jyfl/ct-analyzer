"""
Particle densities in units of 1/cm3.
Energies in units of eV.
"""
import pandas as pd

class OptimizeNE(object):
	"""docstring for Optimizer"""
	def __init__(self, parameters):
		super(Optimizer, self).__init__()
		self.parameters = parameters

		# Load the abc parameters into a dataframe
		# for quick access.
		abc_path = parameters["results_directory"] + "abc.csv"
		self.abc = pd.read_csv(abc_path)

		# Pre-load the (interpolation) functions for the (inz) rate coefficients
		self.rate_coefficient_dict = {}
		for charge_state in parameters["measured_charge_states"]:
			self.rate_coefficient_dict[charge_state] = self.get_rate_coefficient_function(charge_state)


	def calculate_confinement_time(self, charge_state, average_energy, electron_density):
		"""
		Calculate the confinement time of given charge state
		based on the abc parameters.
		"""

		# Rename for brevity
		n = electron_density

		# Retrieve the a, b, c values
		a = self.abc[charge_state]["a"]
        b = self.abc[charge_state]["b"]
        c_l = self.abc[charge_state-1]["c"]

        # Calculate necessary rate coefficients
        rc_lo = self.rate_coefficient[charge_state-1](average_energy)
        rc = self.rate_coefficient[charge_state](average_energy)

        # Calculate confinement time 
        tau = (b - n*rc - (a*c_l/(n*rc_lo)))**(-1)

        return tau


    def get_rate_coefficient_function(self, charge_state):
    	"""
    	Use this function to pre-load the (interpolation) functions
    	with which to calculate the necessary rate coefficients
    	during the optimization. This speeds up the code performance,
    	when you don't have to load the function over and over again.

    	TODO! Voronov formula.
    	"""

    	# If the chosen EED is MB, use the interpolation functions.
    	if self.parameters["optimize_nE"]["rate_coefficient_method"] == "MB":
    		# Find the interpolation functions within the
    		# rate coefficient data directory
    		s = (self.parameters["working_directory"],
    			"rate_coefficient_data/",
    			self.parameters["injected_species"],
    			"/interpMB_",
    			self.parameters["injected_species"],
    			"_"
    			str(charge_state),
    			"+.npy")
    		path = "".join(s)
    		# Load the interpolation function
    		f = np.load(path, allow_pickle=True).item()
    		
    		return f

    def rate_coefficient(self, charge_state, average_energy):
    	"""
    	Average energy of the EED is given in eV.
    	"""
    	# Note conversion m3 -> cm3
    	return = self.rate_coefficient_dict[charge_state](average_energy)*1e6 
