"""
Spectral extraction based on trace x y coordinates in config file. 

2025-07-08

Box extracion is performed over y-block of size extraction-width

Background subtraction can be enable via unit='fwhm' -> then background will be a linear polyfit over 
column over pixels that are 3xfwhm away from centroid

Otherwise set as 'zeros' for no background subtraction. 

"""
import os,sys
import yaml
import numpy as np
from astropy.io import fits
import argparse
import matplotlib.pyplot as plt
from trace import rough_extract

def extract_spectrum(order, centroid, fwhm, unit='fwhm', background_mask=3.0, bkorder=1, extraction_width=5):
    if np.median(fwhm) < 1:
        fwhm = np.ones(len(order))
    if np.nanmax(fwhm)*background_mask > extraction_width:
        fwhm = (extraction_width/(background_mask+1))*np.ones(len(order))
    """
    Extract flux along a curved order cutout (no error arrays).
    """
    nx = order.shape[1]
    y = np.arange(order.shape[0])
    output_flux = np.full(nx, np.nan)

    for i in range(nx-1):
        try:
            if centroid[i] != 0:
                col = order[:, i]
                mask = ~np.isnan(col)
                if mask.sum() == 0 or centroid[i] != centroid[i]:
                    continue
                # define background mask
                if unit == 'fwhm':
                    bkmask = np.abs(y - centroid[i]) > (fwhm[i] * background_mask)
                else:
                    bkmask = np.abs(y - centroid[i]) > background_mask
                bkmask &= mask

                # estimate background
                if bkorder >= 1:
                    spatial = y[bkmask]
                    bkg_flux = col[bkmask]
                    coeff = np.polyfit(spatial, bkg_flux, bkorder)
                    background = np.polyval(coeff, y)
                else:
                    background = np.nanmedian(col[bkmask]) * np.ones_like(y)


                if unit == 'zeros':
                    background = np.zeros(len(y))

                # integrate flux minus background
                optimal_extraction_weights = 1*np.exp(-0.5*(y-centroid[i])**2/fwhm**2)
                #output_flux[i] = np.nansum(col - background) ### just a natural sum over the box
                output_flux[i] = np.nansum((col - background)*optimal_extraction_weights) ### just a natural sum over the box
            else:
                output_flux[i] = np.nan
        except TypeError:
            output_flux[i] = np.nan
    return output_flux[:-1]

def extract_band(data, y0, width):
    """
    Mask `data` so that for each column x only rows y with
    |y − y0[x]| ≤ width remain; everything else is set to np.nan.

    Parameters
    ----------
    data : 2D array, shape (ny, nx)
    y0   : 1D array-like of length nx — central y-position per column
    width: scalar (can be float or int) — half-width of the band

    Returns
    -------
    cut : 2D array, same shape as data, with masked-away pixels set to np.nan
    """
    data = np.asarray(data)
    ny, nx = data.shape

    y0 = np.asarray(y0)
    # if too short, pad with y0[-1]; if too long, truncate
    if y0.size < nx:
        pad_len = nx - y0.size
        y0 = np.concatenate([y0, np.full(pad_len, y0[-1])])
    elif y0.size > nx:
        y0 = y0[:nx]

    # Round width if needed
    w = int(np.round(width))

    # Create a column vector of row indices [0,1,...,ny-1]ᵀ of shape (ny, 1)
    rows = np.arange(ny)[:, None]                # → shape (ny, 1)

    # Broadcast against y0[None, :] → shape (ny, nx)
    mask = np.abs(rows - y0[None, :]) <= w

    # Apply mask
    cut = data.copy()
    cut[~mask] = np.nan

    return cut

def main(cfg,fits_file=None, output_dir=None, bkmasterfile=None, shifts=None):

    if fits_file is None:
        fits_file = cfg['fits_file']
    trace_files = cfg['trace_files']  # list of NPZ trace paths
    trace_names = cfg['trace_names']

    unit = cfg.get('unit', 'fwhm')
    background_mask = cfg.get('background_mask', 3.0)
    extraction_width = cfg.get('extraction_width', 10.0)
    bkorder = cfg.get('bkorder', 1)

    if output_dir is None:
        output_fits = cfg.get('output', 'extracted_spectra.fits')
    else:
        output_fits = os.path.join(output_dir,os.path.basename(fits_file))

    # open input FITS and copy header
    hdul_in = fits.open(fits_file)
    data = hdul_in[0].data.astype(float)

    if not bkmasterfile is None:
        bk = fits.open(bkmasterfile)
        bk = bk[0].data.astype(float)
        data -= bk


    hdr = hdul_in[0].header

    # prepare output HDUList
    hdul_out = fits.HDUList([fits.PrimaryHDU(data=None, header=hdr)])

    # Process first trace file to create one HDU per order
    first_tf = trace_files[0]
    first_label = trace_names[0]
    traces0 = np.load(first_tf, allow_pickle=True)['traces']
    for idx, tr in enumerate(traces0):
        # curved extraction
        #cut, _ = rough_extract(data, tr['order_center'], int(np.round(extraction_width)))
        # extract flux
        if not shifts is None:
            tr['y'] += shifts[:-1]

        y0 = tr['y']
        cut = extract_band(data, y0, extraction_width)
        flux = extract_spectrum(cut, tr['y'], tr['fwhm'],
                                unit=unit, background_mask=background_mask,
                                bkorder=bkorder, extraction_width=extraction_width/2)


        # build columns
        col_x = fits.Column(name=f"{first_label}_X", format='J', array=tr['x'])
        col_flux = fits.Column(name=f"{first_label}_FLUX", format='D', array=flux)
        try:
            col_wave = fits.Column(name=f"{first_label}_WAVE", format='D', array=tr['wave'])
        except IndexError:
            print("Wave index not found")
            pass

        cols = fits.ColDefs([col_x, col_flux, col_wave])
        hdu = fits.BinTableHDU.from_columns(cols, name=f"Order_{idx}")
        hdul_out.append(hdu)

    # Process remaining trace files, appending columns to existing HDUs
    for tf, label in zip(trace_files[1:], trace_names[1:]):
        traces = np.load(tf, allow_pickle=True)['traces']
        for idx, tr in enumerate(traces):
            # curved extraction
            #cut, _ = rough_extract(data, tr['order_center'], int(np.round(extraction_width)))
            if not shifts is None:
                tr['y'] += shifts[:-1]
            y0 = tr['y']
            cut = extract_band(data, y0, extraction_width)
            # extract flux
            flux = extract_spectrum(cut, tr['y'], tr['fwhm'],
                                    unit=unit, background_mask=background_mask,
                                    bkorder=bkorder, extraction_width=extraction_width/2)[1:]
            # new columns
            col_x = fits.Column(name=f"{label}_X", format='J', array=tr['x'])
            col_flux = fits.Column(name=f"{label}_FLUX", format='D', array=flux)
            try:
                col_wave = fits.Column(name=f"{label}_WAVE", format='D', array=tr['wave'])
            except IndexError:
                print("Wave index not found")
                pass
            
            # append to existing HDU
            old_hdu = hdul_out[idx+1]
            new_cols = old_hdu.columns + fits.ColDefs([col_x, col_flux, col_wave])
            new_hdu = fits.BinTableHDU.from_columns(new_cols, name=old_hdu.name)
            hdul_out[idx+1] = new_hdu

    # Write output FITS
    hdul_out.writeto(output_fits, overwrite=True)
    print(f"Wrote extracted spectra to {output_fits}")

if __name__ == '__main__':

    # load configuration
    p = argparse.ArgumentParser(description='Extract spectra from curved traces via YAML config')
    p.add_argument('config', help='YAML configuration file')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)


    main(cfg)
