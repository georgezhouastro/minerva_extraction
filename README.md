# minerva_extraction

Pipeline for 1-D spectrum extraction from MINERVA-Australis spectra

## Quick use
python main.py /path/date/dataset

### Process
Apply bias/dark/flat calibration

Trace & extract with background subtraction

Apply wavelength solution

Inputs
 /path/date/dataset containing science FITS frames [Must contain an iodine exposure] 

Additional options readin in config file (YAML) 

Currently set only for T4 and T5. 
