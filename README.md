- [Overview](#overview)
  - [Data analysis pipeline](#data-analysis-pipeline)
- [Structure](#structure)
- [Todo](#todo)
  - [Data management and storage](#data-management-and-storage)
  - [Data analysis](#data-analysis)
  - [Data visualisation](#data-visualisation)

# Overview
Reorganization of the files I used to analyze the data from the recordings and an attempt to streamline the process. What follows in the TODO section is just a list of functions/scripts I need to port over from all over the place. Some of it was written in Python, some in Matlab, some calculated by hand.

## Data analysis pipeline
1. Ingest
2. Join and align files
3. Reject artifacts
4. Compute biomarker value (and save to database)
5. Calculate power consumption

# Structure
Module `ratdata` contains four submodules:
- `data_manager` with functions for saving and retrieval of the data from the database
- `ingest` with functions for extracting relevant data from the recording files
- `process` for filtering, power computing and analysis
- `plot` for producing plots

# Todo
## Data management and storage
- [x] Find all recording files in a directory
- [x] Find unprocessed recording files in a directory
- [x] Read new recording files into a database
- [ ] Identify a new recording type
- [ ] Identify a rat
- [x] Find related baseline recording
- [ ] Find related amplitude recording
- [ ] Find related DSP recording

## Data analysis
- [x] Reading data from MCE matlab file
- [x] Extract recording type from filename
- [x] Reading data from GUI (bin)
- [x] Reading data from GUI (amplitude)
- [x] Removing missing samples from the GUI-recorded data
- [x] Compute absolute beta power from data
- [x] Compute relative beta power from data
- [x] Compute beta change wrt to baseline
- [ ] Compute TEED
  - [x] Open-loop
  - [x] ON/OFF
  - [ ] Proportional
- [ ] Calculate spectra based on activity
- [ ] Calculate average stim amplitude from proportional control data

## Data visualisation
- [ ] Plot power spectra for all the channels
  - [ ] Monopolar
  - [ ] Bipolar
- [ ] Plot accelerometry data
- [ ] Compare baseline PSD over time