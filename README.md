# README #

The ct-analyzer and its components are thoroughly discussed in the thesis by M Luntinen (in preparation; to be published) titled: "Consecutive Transients Method for plasma diagnostics of Electron Cyclotron Resonance Ion Sources".

# ct-analyzer

A numerical code for determining plasma characteristic values from 
short pulse mode 1+ injection induced extraction current transients
in a CB-ECRIS.

## Compatibility

This code has been developed and tested on Linux Ubuntu (20.04). The code may have compatibility issues which prevent it from running on a Windows/Mac OS.

# Running the code #

The ct-analyzer has two main components: The ct-analyzer itself, which consists of the subroutines

- parse_raw_data
- optimize_abc
- optimize_nE
- plot_results
- masterscript

and an additional subroutine which allows taking an overlap of solution sets output by the ct-analyzer.

## ct-analyzer

Fill the file `parameters.yaml` as appropriate, and run:
`python3 masterScript.py parameters.yaml`

## overlap of solution sets

Fill the file `overlap_parameters.yaml` and run:
`python3 calculate_overlap.py overlap_parameters.yaml`