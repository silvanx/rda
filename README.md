- [Overview](#overview)
- [Structure](#structure)
- [Pipeline](#pipeline)

# Overview
Reorganization of the files I used to analyze the data from the recordings and an attempt to streamline the process. What follows in the TODO section is just a list of functions/scripts I need to port over from all over the place. Some of it was written in Python, some in Matlab, some calculated by hand.

# Structure
Module `ratdata` contains four submodules:
- `data_manager` with functions for saving and retrieval of the data from the database
- `ingest` with functions for extracting relevant data from the recording files
- `process` for filtering, power computing and analysis
- `plot` for producing plots

# Pipeline
The data is processed in Jupyter notebooks. The numbers in ingest_ files matter, the numbers in compute_ and plot_ files don't (although you should run compute_ before the corresponding plot_).
- ingest_
  - add_rats_to_database
  - add_recording_files_to_database
  - connect_recordings_with_baseline
  - add_gui_amplitude_to_database
- compute_
  - beta_power: saved in the database, in RecordingPower table
  - stim_teed: saved in a file, specified in the notebook
  - oof: slight misnomer; 1/f activity is calculated in beta_power; this file creates a text file that is later used for plotting
  - mean_stim_amplitude_from_gui_recording: Prints out the result
- plot_
  - peak_locations_from_blinded_spectra: Peak locations for each rat, based on the manual peak finding (results in an Excel spreadsheet specified in the notebook)
  - peak_locations_auto: Peak locations for each rat, based on automatic peak detection
  - blinded_power_spectra: Power spectra in random order with file names replaced by a number; the corresponding numbers and filenames are saved to a txt file key.txt
  - immunocytochemistry: Results from immunocytochemistry testing (from an Excel spreadsheet)
  - relative_beta_power_over_time: Change in relative beta over time in baseline recordings and relative beta during a single recording day; data taken from database
  - behaviour_data: Behaviour results from csv files
  - oof: 1/f activity from the text file created in compute_oof
  - beta_power: Needs to be split into pieces; first, saves the relative beta power change measured in 6-OHDA and sham rats (in two separate csv files) and then it plots it
  - stim_teed: plots teed in 1s from text file