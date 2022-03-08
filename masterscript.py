"""
Example:
    >>> python3 masterScript.py parameters.yaml
"""

import sys
import yaml
import pandas as pd

import parameter_tester
import parse_raw_data
import optimize_abc

pathToParameterFile = sys.argv[1]
parameters = yaml.safe_load(open(pathToParameterFile)) 


# Use the parameter_tester to check the validity
# of the given parameters file.
ParameterTester = parameter_tester.ParameterTester(parameters)
errorCount, warningCount = ParameterTester.doTests()

# If there were no errors, run the script.
if errorCount == 0:

	# Parse the raw data
	try:
		print("Parsing raw data...")
		Parser = parse_raw_data.RawDataParser(parameters)
		Parser.doParse()
		print("Finished parsing!\n")
	except:
		print("Failed to parse raw data.")
		sys.exit()

	# Run the abc optimization
	try:
		print("Beginning abc optimization...")
		OptABC = optimize_abc.OptimizeABC(parameters)
		OptABC.doOptimizeABC()
		print("Finished abc optimization!\n")
	except:
		print("Failed to optimize the abc parameters.")
		sys.exit()
	


