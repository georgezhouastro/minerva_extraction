#!/usr/bin/env python3
import os
import glob
from collections import defaultdict

import numpy as np
from astropy.io import fits

def make_master_flats(directory='.'):
    # 1) Find all FIT files
    fits_files = glob.glob(os.path.join(directory, '*.FIT')) + \
                 glob.glob(os.path.join(directory, '*.fit'))
    
    # 2) Group by the first two dash-separated fields of the filename
    groups = defaultdict(list)
    for path in fits_files:
        fname = os.path.basename(path)
        parts = fname.split('-')
        if len(parts) < 2:
            continue
        key = f"{parts[0]}-{parts[1]}"
        groups[key].append(path)
    
    # 3) For each group, stack and average
    for key, files in groups.items():
        print(f"Processing {key}: {len(files)} files")
        stack = []
        for fn in files:
            with fits.open(fn) as hdul:
                data = hdul[0].data.astype(np.float64)
                stack.append(data)
        master_image = np.nanmean(stack, axis=0)
        
        # 4) Grab header from the first file
        with fits.open(files[0]) as hdul0:
            hdr = hdul0[0].header
        
        # 5) Write out master
        outname = os.path.join(directory, f"{key}-master.FIT")
        hdu = fits.PrimaryHDU(master_image, header=hdr)
        hdu.writeto(outname, overwrite=True)
        print(f"  → Wrote {outname}\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Build master flats by grouping on filename prefix"
    )
    p.add_argument("dir", nargs="?", default=".", help="Directory of FITS files")
    args = p.parse_args()
    make_master_flats(args.dir)
