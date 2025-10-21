import sys
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import optimize

# Make sure computePSF and trace() are defined or imported in the same module
# e.g., from your_existing_trace_module import computePSF, trace
def poly2D(x, y, coeffs):
    order = int(np.sqrt(len(coeffs))) - 1
    z = np.zeros_like(x)
    index = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            z += coeffs[index] * (x ** i) * (y ** j)
            index += 1
    return z


# Define the function for curve_fit
def fit_func(data, *coeffs):
    x, y = data
    return poly2D(x, y, coeffs)

    
def sigma_clipped_polyfit(x, y, degree, sigma=3.0, max_iter=5):
    """
    Perform a sigma-clipped polynomial fit to data.

    Parameters:
    x : array_like
        x-coordinates of the data.
    y : array_like
        y-coordinates of the data.
    degree : int
        Degree of the polynomial to fit.
    sigma : float, optional
        Number of standard deviations to use for clipping.
    max_iter : int, optional
        Maximum number of iterations for the sigma clipping.

    Returns:
    p : ndarray
        Polynomial coefficients, highest power first.
    """

    # Initial fit
    x = np.array(x)
    y = np.array(y)
    mask = x == x
    mask *= y == y
    x = x[mask]
    y = y[mask]

    p = np.polyfit(x, y, degree)
    
    for _ in range(max_iter):
        # Evaluate the polynomial
        y_fit = np.polyval(p, x)
        
        # Calculate residuals
        residuals = y - y_fit
        
        # Compute the standard deviation of the residuals
        std = np.std(residuals)
        
        # Identify inliers
        mask = np.abs(residuals) < sigma * std
        
        # Update x and y to only use inliers
        x_inliers = x[mask]
        y_inliers = y[mask]

        
        # Refit the polynomial
        p = np.polyfit(x_inliers, y_inliers, degree)
        
    return p

def psf(x0,x):
    """
    Gaussian PSF model
    """ 

    psfmodel = x0[2]*np.exp(-(x-x0[0])**2/x0[1]**2)+x0[3]*x+x0[4]
    return psfmodel

def computePSF(prof):
    """
    Fit of PSF model
    """ 

    x = np.arange(len(prof))

    xmax = x[np.nanargmax(prof)]
    
    def minfunc(x0):
        if abs(x0[0]-xmax) < 2 and x0[1] > 0 and x0[1] < 5:
            psfmodel = psf(x0,x)
            diff = np.nansum((psfmodel-prof)**2)
            return diff
        else:
            return np.inf
    
    x0 = [xmax,1.,np.nanmax(prof),0,0]
    x0 = optimize.fmin(minfunc,x0,disp=0)

    return x0

def trace(order,blocksize=10,debug=False):
    """
    Trace along stellar signal by a Gaussian fit along the spatial axis
    Wavelength axis is binned by width of blocksize pixels

    Operates on a 2D image of an order
    """

    
    i = 0
    gauss_fit_params = []
    while i < len(order[0]):
        prof = np.nanmedian(order[:,i:i+blocksize],axis=1)
        detection = np.nanmax(prof) / np.nanmedian(abs(np.diff(prof)))
        #print(detection)
        if detection > 50:
            x0 = computePSF(prof)
            x0 = [i]+list(x0)
            gauss_fit_params.append(x0)
        else:
            gauss_fit_params.append(np.ones(6)*np.nan)

        i += blocksize

    gauss_fit_params= np.array(gauss_fit_params)
    mask = gauss_fit_params[:,0]==gauss_fit_params[:,0]
    
    centroid = sigma_clipped_polyfit(gauss_fit_params[:,0][mask][10:-10],gauss_fit_params[:,1][mask][10:-10],2)
    centroid = np.polyval(centroid,np.arange(len(order[0])))
    invalidmask = np.arange(len(order[0])) > min(gauss_fit_params[:,0][mask])
    invalidmask *= np.arange(len(order[0])) < max(gauss_fit_params[:,0][mask])
    centroid[np.invert(invalidmask)] = 0
    
    fwhmextraction = sigma_clipped_polyfit(gauss_fit_params[:,0][mask][10:-10],gauss_fit_params[:,2][mask][10:-10],0)
    fwhmextraction = np.polyval(fwhmextraction,np.arange(len(order[0])))

    if debug:
        plt.subplot(311)
        plt.plot(gauss_fit_params[:,0],gauss_fit_params[:,1])
        plt.plot(np.arange(len(order[0])),centroid)
        plt.ylabel("Y")
        plt.subplot(312)
        plt.plot(gauss_fit_params[:,0],gauss_fit_params[:,2])
        plt.plot(np.arange(len(order[0])),fwhmextraction)
        plt.ylabel("FWHM")
        plt.subplot(313)
        plt.plot(gauss_fit_params[:,0],gauss_fit_params[:,3])
        plt.ylabel("Height")
        plt.show()

        plt.imshow(order,aspect='auto')
        plt.plot(np.arange(len(order[0])),centroid)
        plt.show()


    return gauss_fit_params[:,0],centroid,fwhmextraction,gauss_fit_params



def find_orders(image, centre_frac=0.1, n_orders=None, prominence=0.1, debug=True):
    """
    Identify the approximate Y-centroids of echelle orders by looking
    at the median spatial profile in the central columns.

    Parameters
    ----------
    image : 2D ndarray
        Input echelle frame (ny, nx).
    centre_frac : float
        Fraction of X-axis around center to use for profile.
    n_orders : int or None
        Number of brightest orders to keep. If None, keep all peaks.
    prominence : float
        Minimum peak prominence as a fraction of max(profile).

    Returns
    -------
    peaks : 1D ndarray
        Sorted Y-indices of order centers.
    """
    ny, nx = image.shape
    cx = nx // 2
    half = int(nx * centre_frac / 2)
    x_start, x_end = max(cx - half, 0), min(cx + half, nx)
    vert_profile = np.nanmedian(image[:, x_start:x_end], axis=1)

    # detect peaks in the vertical profile
    threshold = prominence * np.nanmax(vert_profile)
    peaks, __ = find_peaks(vert_profile, prominence=threshold)
    peaks = peaks[peaks>30]

    if n_orders is not None and len(peaks) > n_orders:
        # select the top-n_orders by profile height
        heights = vert_profile[peaks]
        top_idx = np.argsort(heights)[-n_orders:]
        peaks = peaks[top_idx]



    if debug:
        plt.plot(vert_profile[peaks], peaks, 'rx', ms=8)
        plt.plot(vert_profile, np.arange(ny), '-', lw=1)
        plt.show()


    return np.sort(peaks)

def rough_extract(image, center_y, blocksize):
    """
    Create a full-frame curved extraction: at each X, keep only pixels
    within ±blocksize of the local center, fill others with NaN.
    Returns 'cut' of same shape as image, and the center trace y_center.
    """
    ny, nx = image.shape
    half = blocksize
    y_center = np.full(nx, np.nan)
    cx = nx // 2
    y_center[cx] = center_y
    # trace left
    for x in range(cx-1, -1, -1):
        prev = int(round(y_center[x+1]))
        lo = max(prev-half, 0); hi = min(prev+half+1, ny)
        segment = image[lo:hi, x]
        if segment.size and not np.all(np.isnan(segment)):
            rel = np.nanargmax(segment)
            y_center[x] = lo + rel
    # trace right
    for x in range(cx+1, nx):
        prev = int(round(y_center[x-1]))
        lo = max(prev-half, 0); hi = min(prev+half+1, ny)
        segment = image[lo:hi, x]
        if segment.size and not np.all(np.isnan(segment)):
            rel = np.nanargmax(segment)
            y_center[x] = lo + rel
    # build curved cutout
    cut = np.full_like(image, np.nan)
    for x in range(nx):
        yc = y_center[x]
        if np.isnan(yc): continue
        yc = int(round(yc))
        lo = max(yc-half, 0); hi = min(yc+half+1, ny)
        cut[lo:hi, x] = image[lo:hi, x]
    return cut, y_center

def trace_orders(fits_path, blocksize=10, centre_frac=0.1, n_orders=None, debug=False, wavelengthfile=None,orderoffset=None):
    """
    Automatically trace all echelle orders in a FITS frame, with optional diagnostics.
    Uses a rough curved extraction before Gaussian fitting.
    """
    # Load image
    hdul = fits.open(fits_path)
    data = hdul[0].data.astype(float)
    hdul.close()

    # Detect order centers
    centers = find_orders(data, centre_frac=centre_frac,
                          n_orders=n_orders, prominence=0.05,
                          debug=debug)
    traces = []
    ny, nx = data.shape

    if not wavelengthfile is None and not orderoffset is None:
        import pickle
        wavelengthfile = pickle.load(open(wavelengthfile,'rb'))[1:]
        centers = centers[orderoffset:]
        centers = centers[:len(wavelengthfile)]

    for idx, y0 in enumerate(centers):
        # rough curved extract
        order_data, rough_center = rough_extract(data, y0, blocksize)

        # optional debug: plot rough center on full frame
        if debug:
            print(rough_center)
            xs = np.arange(nx)

            plt.figure(figsize=(10,6))
            plt.subplot(211)
            plt.imshow(data, aspect='auto', origin='lower')
            plt.plot(xs, rough_center, '-', color='r', lw=1, label=f'Rough {idx}')
            plt.subplot(212)
            plt.imshow(order_data, aspect='auto', origin='lower')
            plt.show()
            


        # run gaussian-based trace on curved image
        x_idx, centroid, fwhm, params = trace(order_data,
                                               blocksize=blocksize,
                                               debug=debug)
        # centroid is relative to cut image; shift by rough_center midpoint
        #y_trace = centroid + rough_center.mean() - (order_data.shape[0]/2)




        if wavelengthfile is None:
            traces.append({
                'order_center': y0,
                'rough_x': x_idx,
                'rough_center': rough_center,
                'x': np.arange(len(centroid)),
                'y': centroid,
                'fwhm': fwhm
            })

        else:
            traces.append({
                'order_center': y0,
                'rough_x': x_idx,
                'rough_center': rough_center,
                'x': np.arange(len(centroid))[1:],
                'y': centroid[1:],
                'fwhm': fwhm[1:],
                'wave':wavelengthfile[idx]
            })

    if True:
        # overlay final fitted trace
        plt.figure(figsize=(10,6))
        plt.imshow(data, aspect='auto', origin='lower')
        for itr, tr in enumerate(traces):
            print(itr,tr['x'],tr['y'])
            plt.plot(tr['x'], tr['y'], '-', lw=1.5, color='r')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.title('Rough and Gaussian traces')
        plt.show()


    return traces


if __name__ == '__main__':
    import argparse
    import numpy as _np

    parser = argparse.ArgumentParser(
        description='Trace all orders in an echelle FITS frame.')
    parser.add_argument('fits_file', help='Path to the echelle FITS image')
    parser.add_argument('--output', help='Path to the output trace file')
    parser.add_argument('--blocksize', type=int, default=10,
                        help='Bin width in X for Gaussian fitting')
    parser.add_argument('--centre_frac', type=float, default=0.1,
                        help='Fraction of X-axis for order detection')
    parser.add_argument('--n_orders', type=int, default=None,
                        help='Number of orders to detect (brightest)')
    parser.add_argument('--wavelengthfile', type=str, default="wavelengthfile.pkl",
                        help='Input wavelength file')
    parser.add_argument('--orderoffset', type=int, default=0,
                        help='Order offset between detection and input wavelength (default 2 as of 20250707')    
    parser.add_argument('--debug', action='store_true',
                        help='Plot diagnostics during tracing')
    args = parser.parse_args()

    traces = trace_orders(args.fits_file,
                          blocksize=args.blocksize,
                          centre_frac=args.centre_frac,
                          n_orders=args.n_orders,
                          debug=args.debug,
                          wavelengthfile = args.wavelengthfile,
                          orderoffset = args.orderoffset
)

    # # save results for later use
    _np.savez(args.output, traces=traces)
    print(f'Traced {len(traces)} orders and saved to echelle_traces.npz')
