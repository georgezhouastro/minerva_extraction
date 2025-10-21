import os,sys
from astropy.io import fits
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

def median_column(img, i, window=5):
    """
    Return the median‐filtered column at index i of img, 
    averaging up to `window` columns centered on i.
    """
    half = window // 2
    # Clip the window so it stays within [0, nx)
    start = max(i - half, 0)
    end   = min(i + half + 1, img.shape[1])  # +1 because slice end is exclusive

    # img[:, start:end] has shape (ny, n_cols_in_window)
    # take the median along axis=1 (i.e. down each row)
    return np.median(img[:, start:end], axis=1)
def sigma_clipped_polyfit(x, y, degree=1, sigma=3, max_iters=5):
    """
    Fit a polynomial of given degree to (x,y), iteratively
    rejecting points more than `sigma`·σ from the current fit.
    Returns (coeffs, model_y).
    """
    # start with all points
    mask = np.isfinite(y)

    for _ in range(max_iters):
        # fit to the “good” points
        coeffs = np.polyfit(x[mask], y[mask], degree)
        model = np.polyval(coeffs, x)

        # compute residuals & their dispersion
        resid = y - model
        std = np.std(resid[mask])

        # update mask
        new_mask = np.abs(resid) < sigma * std
        # if nothing changed, stop early
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    # final fit on clipped data
    coeffs = np.polyfit(x[mask], y[mask], degree)
    model = np.polyval(coeffs, x)
    return coeffs, model


def two_stage_sigma_clipped_polyfit(x, y, degree=1, sigma=3, max_iters=5):
    """
    1) Fit a degree-0 (constant) model to y vs x, sigma-clip outliers.
    2) On the clipped dataset, do an iterative sigma-clipped fit of degree `degree`.
    Returns:
      c0         : constant fit coefficient (array of length 1)
      coeffs     : final polynomial coeffs of length (degree+1)
      model_full : full-model values (degree-n) at every x
    """
    # --- Stage 1: constant fit & one-time clip ---
    c0 = np.polyfit(x, y, 0)                # degree-0
    m0 = np.polyval(c0, x)
    resid0 = y - m0
    std0 = np.median(abs(resid0))
    #std0 = np.std(resid0)
    mask = np.abs(resid0) < sigma * std0   # keep within ±σ*sigma

    # --- Stage 2: iterative degree-n fit on clipped data ---
    for _ in range(max_iters):
        coeffs = np.polyfit(x[mask], y[mask], degree)
        model = np.polyval(coeffs, x)
        resid = y - model
        std = np.median(abs(resid))
        #std = np.std(resid[mask])
        new_mask = np.abs(resid) < sigma * std
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    # final fit
    coeffs = np.polyfit(x[mask], y[mask], degree)
    model_full = np.polyval(coeffs, x)

    return c0, coeffs, model_full

def column_shifts(img1, img2):
    """
    Compute per-column vertical shift (in pixels) between img1 and img2.
    Positive shift means img2 is shifted downward relative to img1.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    ny, nx = img1.shape
    shifts = np.full(nx, np.nan)

    for i in range(nx):
        col1 = median_column(img1, i, window=5)
        col2 = median_column(img2, i, window=5)
        #col1 = img1[:, i]
        #col2 = img2[:, i]
        # mask NaNs
        good = np.isfinite(col1) & np.isfinite(col2)
        if good.sum() < 2:
            continue  # not enough data to correlate

        c1 = col1[good] - np.mean(col1[good])
        c2 = col2[good] - np.mean(col2[good])

        corr = correlate(c2, c1, mode='full')
        lags = np.arange(-len(c1) + 1, len(c1))
        
        mask = abs(lags) < 10
        mask *= corr > 0.5*np.nanmax(corr[mask])
        fit = np.polyfit(lags[mask],corr[mask],2)
        shifts[i] = -0.5*fit[1]/fit[0]
        
        #shifts[i] = lags[np.argmax(corr)]

    return shifts

def main(inputfits,calibrationfits,outdir):
    # 1. Read FITS
    with fits.open(inputfits) as hd1, fits.open(calibrationfits) as hd2:
        img1 = hd1[0].data.astype(float)
        img2 = hd2[0].data.astype(float)

    # 2. Compute shifts
    shifts = column_shifts(img1, img2)
    x = np.arange(len(shifts))
    #shiftsmodel = np.polyfit(x,shifts,1)
    #shiftsmodel = np.polyval(shiftsmodel,x)

    # coeffs, shiftsmodel = sigma_clipped_polyfit(x, shifts,
    #                                         degree=0,
    #                                         sigma=2,
    #                                         max_iters=5)
    
    c0, lin_coeffs, shiftsmodel = two_stage_sigma_clipped_polyfit(
        x, shifts,
        degree=1,    # your desired final polynomial degree
        sigma=2,
        max_iters=5
    )
    # 3. (Optional) Plot the shifts vs. column index
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(len(shifts)), shifts, marker='o', linestyle='-')
    plt.plot(np.arange(len(shifts)),shiftsmodel)
    plt.xlabel('Column index (x)')
    plt.ylabel('Vertical shift (pixels)')
    plt.title('Column-by-column vertical offset between images')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(outdir,"vertical_shift.png"))

    return shifts,shiftsmodel


if __name__ == "__main__":
    inputfits = sys.argv[1]
    calibrationfits = sys.argv[2]
    outdir = sys.argv[3]
    main(inputfits,calibrationfits,outdir)
