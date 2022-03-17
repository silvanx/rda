- [Overview](#overview)
  - [Data analysis pipeline](#data-analysis-pipeline)
- [Todo](#todo)
  - [Data management and storage](#data-management-and-storage)
  - [Data analysis](#data-analysis)
  - [Data visualisation](#data-visualisation)

# Overview
I am trying to organize the files I used to analyze the data from the recordings and streamline the process. What follows is just a list of functions/scripts I need to port over from all over the place. Some of it was written in Python, some in Matlab, some calculated by hand. Time to streamline the process.

## Data analysis pipeline
1. Ingest
2. Join and align files
3. Reject artifacts
4. Compute biomarker
5. Calculate power consumption

# Todo
## Data management and storage
- [x] Find all recording files in a directory
- [x] Find unprocessed recording files in a directory
- [x] Read new recording files into a database
- [ ] Identify a new recording type
- [ ] Identify a rat
- [ ] Find related baseline recording
- [ ] Find related amplitude recording
- [ ] Find related DSP recording

## Data analysis
- [x] Reading data from MCE matlab file
- [x] Extract recording type from filename
- [ ] Reading data from MCE HDF5 file
- [ ] Reading data from GUI (csv)
- [x] Reading data from GUI (bin)
- [x] Reading data from GUI (amplitude)
- [x] Removing missing samples from the GUI-recorded data
- [x] Compute absolute beta power from data
- [x] Compute relative beta power from data
- [x] Compute beta change wrt to baseline
- [ ] Compute TEED
  - [x] Open-loop
  - [ ] ON/OFF
  - [ ] Proportional
- [ ] Calculate spectra based on activity
- [ ] Calculate average stim amplitude from proportional control data

## Data visualisation
- [ ] Plot power spectra for all the channels
  - [ ] Monopolar
  - [ ] Bipolar
- [ ] Plot accelerometry data
- [ ] Compare baseline PSD over time