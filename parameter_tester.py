"""
Checks that all parameters given to the
masterscript are valid.

The parameters given to ParameterTester are the 
ones specified by the .yaml file.

TÒDO!
- Create a separate script which 
runs the parameter tests by calling this 
script, with the desired .yaml file as 
its input. 
"""

import os

class ParameterTester(object):
	"""docstring for ParameterTester"""
	def __init__(self, parameters):
		super(ParameterTester, self).__init__()
		self.parameters = parameters

	def doTests(self):
		"""
		TODO:
		Check that all directories are actual directories.
		Check that integer values are given where they are required.
		Check that lists are lists.
		Check that injected species is lower()
		Make sure that rolling average is not negative, or greater than 10(0).
		  Might want to warn user if it is bigger than 5.
		"""
		errorCount = 0
		warningCount = 0
		print("\nTesting parameter validity...")
		
		# TODO! Run sub-routines.
		# Check all directories.
		# ...
		print("TODO! Check that all directories are actual directories.")
		print("TODO! Check that integer values are given where they are required.")
		print("TODO! Check that lists are lists.")
		print("TODO! Check that floats are floats.")
		print("TODO! Check that doubles are doubles.")
		print("TODO! Check that injected species is lower()")
		print("TODO! Make sure that rolling average is not negative, or greater than 10(0).")
		
		print("Process exited with {} errors and {} warnings.\n"\
			.format(errorCount,warningCount))


		return errorCount, warningCount