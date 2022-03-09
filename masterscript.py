"""
Example:
    >>> python3 masterScript.py parameters.yaml

TODO! 
- Use the time library to print time elapsed for each sub-routine.
- n&E optimization
- plotting (might want to create a separate script for this. 
Will avoid clutter, and enable more features.)
"""

import sys
import yaml
import pandas as pd
import parameter_tester
import parse_raw_data
import optimize_abc
import optimize_nE


# Load the parameters from the .yaml file
pathToParameterFile = sys.argv[1]
parameters = yaml.safe_load(open(pathToParameterFile)) 



# Use the parameter_tester to check the validity
# of the given parameters file.
try:
	ParameterTester = parameter_tester.ParameterTester(parameters)
	errorCount, warningCount = ParameterTester.doTests()
except:
	print("Couldn't test parameters .yaml file.")
	print("Check that file exists.")
	print("Exiting...")
	sys.exit()

# If there were no errors, run the script.
if errorCount == 0:

	# Parse the raw data
	if parameters["parse_raw_data"]["do"]:
		try:
			print("Parsing raw data...")
			Parser = parse_raw_data.RawDataParser(parameters)
			Parser.doParse()
			print("Finished parsing!\n")
		except:
			print("Failed to parse raw data.")
			print("Exiting...")
			sys.exit()

	# Run the abc optimization
	if parameters["optimize_abc"]["do"]:
		try:
			print("Beginning abc optimization...")
			OptABC = optimize_abc.OptimizeABC(parameters)
			OptABC.doOptimizeABC()
			print("Finished abc optimization!\n")
		except:
			print("Failed to optimize the abc parameters.")
			print("Exiting...")
			sys.exit()

	# Run the (n,E) optimization
	if parameters["optimize_nE"]["do"]:
		# try:
		# 	print("Beginning (n,E)-optimization...")
		# 	OptNE = optimize_nE.OptimizeNE(parameters)
		# 	OptNE.doOptimizeNE()
		# 	print("Finished (n,E)-optimization!\n")
		# except:
		# 	print("Failed to optimize (n,E).")
		# 	print("Exiting...")
		# 	sys.exit()
		print("Beginning (n,E)-optimization...\n")
		OptNE = optimize_nE.OptimizeNE(parameters)
		OptNE.doOptimizeNE()
		print("\nFinished (n,E)-optimization!\n")

	print("Thank you for flying with us!")