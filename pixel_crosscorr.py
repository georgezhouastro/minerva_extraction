"""
Script to cross-correlate echelle spectra in pixel space chunk-by-chunk,
fit a 2D polynomial to the pixel offsets, and optionally produce diagnostic plots.
"""
import yaml
import argparse
import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
c = 299792.

import emcee


def findccpeak(drv,cc):

    v0 = drv[np.argmax(cc)]
    velmask = abs(drv - v0) < 5
    velmask *= cc > 0.3*np.nanmax(cc)
    err = 0.001 * np.nanstd(cc[~velmask])

    def lnlike(x0):
        f = x0[2] * np.exp(-1*(drv[velmask] - x0[0])**2/(2*x0[1]**2))
        residual = f - cc[velmask]
        prob = -0.5 * np.nansum(residual**2 / err**2)

        if prob == prob:
            return prob
        else:
            prob = -1*inf

    # MCMC fit
    nwalkers = 100
    p0 = []
    while len(p0) < nwalkers:
        goodwalker = False
        pi = [np.random.normal(v0, 1.),
              np.random.normal(2., 0.1),
              np.random.normal(max(cc) - np.median(cc),
              0.1 * (max(cc) - np.median(cc)))]

        if abs(lnlike(pi)) < np.inf:
            goodwalker = True
            p0.append(pi)
        else:
            print("Bad walker")

    ndim = len(p0[0])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, threads=1)
    pos, _, _ = sampler.run_mcmc(p0, 200)
    sampler.reset()
    sampler.run_mcmc(pos, 200)

    chain = sampler.flatchain
    return np.nanmedian(chain[:, 0])


def crosscorrpixel(x, flux, template_x, templ_flux, pad_factor=2, kind="linear"):
    mask1 = np.isfinite(flux)
    mask2 = np.isfinite(templ_flux)
    x1, f1 = x[mask1], flux[mask1]
    x2, f2 = template_x[mask2], templ_flux[mask2]
    if len(x1) < 2 or len(x2) < 2:
        return None, None, None, 0
    xmin, xmax = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    dx = min(np.median(np.diff(x1)), np.median(np.diff(x2)))
    grid = np.arange(xmin, xmax, dx)
    if len(grid) < 2:
        return None, None, None, 0
    interp1_func = interp1d(x1, f1, kind=kind, bounds_error=False, fill_value=0.)
    interp2_func = interp1d(x2, f2, kind=kind, bounds_error=False, fill_value=0.)
    g1, g2 = interp1_func(grid), interp2_func(grid)
    n = len(grid)
    m = int(2**np.ceil(np.log2(n))) * pad_factor
    pad1 = np.zeros(m); pad2 = np.zeros(m)
    pad1[:n], pad2[:n] = g1, g2
    cc = np.fft.ifft(np.fft.fft(pad1) * np.conj(np.fft.fft(pad2)))
    cc_shifted = np.fft.fftshift(cc).real
    lags = np.arange(-m//2, m//2)
    peak_idx = np.argmax(cc_shifted)
    peak = cc_shifted[peak_idx]
    #offset = int(lags[peak_idx])

    mask = cc_shifted > 0.5*np.nanmax(cc_shifted)

    if True: ### bottom chopped Gaussian fit
        offset = findccpeak(lags,cc_shifted)

    if False: ### Quadratic peak fit
        offset = np.polyfit(lags[mask],cc_shifted[mask],2)
        offset = -0.5*offset[1]/offset[0]

    noise = np.nanstd(cc_shifted[np.invert(mask)])
    snr = (peak / noise) 

    return offset, cc_shifted, lags, snr


def fit_2d_poly(order_list, pos_list, offset_list):
    O = np.array(order_list)
    P = np.array(pos_list)
    Y = np.array(offset_list)
    X = np.vstack([
        np.ones_like(O), O, P, O**2, O*P, P**2
    ]).T
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coeffs


def predict_offset(order, x, coeffs):
    a, b, c, d, e, f = coeffs
    return a + b*order + c*x + d*order**2 + e*order*x + f*x**2


def main(config,science,chunk_size=400,snr_threshold=10,sigma_clip=3,max_iter=5,diagnostics=True,out_dir='.',diag_dir='.'):

    cal = config['iodinetemplate']
    cal_hdul, sci_hdul = fits.open(cal), fits.open(science)
    results, diagnostics = [], []

    for order in range(1, len(cal_hdul)):
        sci_flux = sci_hdul[order].data['Cal_FLUX']
        cal_flux = cal_hdul[order].data['Cal_FLUX']
        x_full = np.arange(len(sci_flux))
        mask_sci = np.isfinite(sci_flux)
        p1 = np.polyfit(x_full[mask_sci], sci_flux[mask_sci], 4)
        f1 = sci_flux / np.polyval(p1, x_full) - 1
        mask_cal = np.isfinite(cal_flux)
        p2 = np.polyfit(x_full[mask_cal], cal_flux[mask_cal], 4)
        f2 = cal_flux / np.polyval(p2, x_full) - 1

        for start in range(0, len(x_full), chunk_size):
            end = start + chunk_size
            xs = x_full[start:end]
            f1c, f2c = f1[start:end], f2[start:end]
            if len(xs) < 10: continue
            offset, cc, lags, snr = crosscorrpixel(xs, f1c, xs, f2c)
            ok = (offset is not None and snr >= snr_threshold)
            diagnostics.append((xs, cc, lags, ok))
            if ok: results.append((order, xs.mean(), offset))

    if not results:
        print("No valid peaks.")
        return

    # iterative sigma-clipped 2D fit
    O, P, Z = map(np.array, zip(*results))
    mask = np.ones_like(Z, dtype=bool)
    coeffs = None
    for i in range(max_iter):
        coeffs = fit_2d_poly(O[mask], P[mask], Z[mask])
        Z_pred = predict_offset(O, P, coeffs)
        resid = Z - Z_pred
        sigma = np.std(resid[mask])
        new_mask = np.abs(resid) < sigma_clip * sigma
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    print(f"2D poly coeffs after {i+1} iterations:", coeffs)

    # diagnostics
    if diagnostics:
        # grid of CCFs
        n = len(diagnostics)
        cols = 5
        rows = int(np.ceil(n/cols))
        fig = plt.figure(figsize=(cols*3, rows*2))
        gs = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.4)
        for idx, (xs, cc, lags, ok) in enumerate(diagnostics):
            ax = fig.add_subplot(gs[idx])
            ax.plot(lags, cc, color='C0' if ok else 'C1')
            ax.axvline(lags[np.argmax(cc)], linestyle='--', color='k')
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('Chunk CCFs (blue=ok, red=rej)')
        fig.savefig(os.path.join(diag_dir, os.path.basename(science)+'ccf_grid.png'), dpi=150)
        plt.close(fig)

        # 3D surface + points
        fig = plt.figure(figsize=(8,6))
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.scatter(P[mask], O[mask], Z[mask], c='k', marker='o', label='inliers')
        ax3d.scatter(P[~mask], O[~mask], Z[~mask], c='r', marker='x', label='outliers')
        Og, Pg = np.meshgrid(
            np.linspace(O.min(), O.max(), 50),
            np.linspace(P.min(), P.max(), 50)
        )
        Zg = predict_offset(Og, Pg, coeffs)
        ax3d.plot_surface(Pg, Og, Zg, alpha=0.5, cmap='viridis')
        ax3d.set_xlabel('Pixel pos'); ax3d.set_ylabel('Order'); ax3d.set_zlabel('Offset')
        ax3d.set_title('Sigma-clipped 2D Fit Surface')
        ax3d.set_zlim(-0.5,0.5)
        ax3d.legend()
        fig.savefig(os.path.join(diag_dir, os.path.basename(science)+'offset_surface.png'), dpi=150)
        #plt.show()
        plt.close(fig)

    cal_hdul.close()


    ### Apply wavelength correction to science frame

    trace_names = config['trace_names']
    trace_vel_offsets = config['trace_vel_offsets']

    for order in range(1, len(sci_hdul)):
        for label, vel_offset in zip(trace_names, trace_vel_offsets):
            sci_wave = sci_hdul[order].data[f"{label}_WAVE"]
            offsets = predict_offset(order, np.arange(len(sci_wave)), coeffs)
            dispersion = np.gradient(sci_wave)
            # wavelength shift per pixel
            sci_wave -= dispersion * offsets        
            sci_wave += (vel_offset/c)*sci_wave
            sci_hdul[order].data[f"{label}_WAVE"] = sci_wave


    sci_hdul[0].header['I2_CORR'] = np.nanmedian(Z.flatten())
    sci_hdul.writeto(os.path.join(out_dir, os.path.basename(science)), overwrite=True)

    sci_hdul.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Chunked pixel cross-correlation with 2D fit and diagnostics")
    parser.add_argument('--config', help='YAML configuration file', default='config.yaml')
    parser.add_argument("--science", required=True)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--snr-threshold", type=float, default=10.0)
    parser.add_argument("--sigma-clip", type=float, default=3.0, help="Sigma threshold for iterative clipping")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations for sigma clipping")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--diag-dir", default=".")
    args = parser.parse_args()

    if args.diagnostics:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.diag_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config,args.science,args.chunk_size,args.snr_threshold,args.sigma_clip,args.max_iter,args.diagnostics,args.out_dir,args.diag_dir)
