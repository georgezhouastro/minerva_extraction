import argparse
import os
import shutil
import glob
import numpy as np
from astropy.io import fits

def process_one(path, raw_dir, dry_run=False):
    # Open FITS (do not memmap; we’ll move file after reading)
    with fits.open(path, memmap=False) as hdul:
        # Find the first 2D image HDU (usually PrimaryHDU)
        hdu_idx = None
        for i, h in enumerate(hdul):
            if isinstance(h, (fits.PrimaryHDU, fits.ImageHDU)) and h.data is not None:
                if h.data.ndim == 2:
                    hdu_idx = i
                    break
        if hdu_idx is None:
            print(f"  [skip] No 2D image HDU in {os.path.basename(path)}")
            return

        hdr = hdul[hdu_idx].header
        data = hdul[hdu_idx].data

        # Check exact header geometry
        naxis1 = hdr.get('NAXIS1', None)
        naxis2 = hdr.get('NAXIS2', None)
        if naxis1 != 2200 or naxis2 != 2052:
            print(f"  [skip] Header size mismatch (NAXIS1={naxis1}, NAXIS2={naxis2}) in {os.path.basename(path)}")
            return

        if data.shape != (naxis2, naxis1):
            print(f"  [warn] Data shape {data.shape} != header ({naxis2},{naxis1}) in {os.path.basename(path)}; proceeding with data shape.")

        # Convert to float for safe subtraction
        img = np.asarray(data, dtype=np.float32)

        # Define regions: active [0:2048), overscan [2048:2200)
        active = img[:, :2048]
        overscan = img[:, 2048:2200]

        # Median per row over overscan region; keep dims for broadcasting
        row_medians = np.median(overscan, axis=1, keepdims=True)

        # Subtract from active region
        active_corr = active - row_medians

        # Trim to 2048 columns (drop overscan)
        trimmed = active_corr  # shape (2052, 2048)

        # Prepare output header: copy and update NAXIS1
        out_hdr = hdr.copy()
        out_hdr['NAXIS1'] = 2048  # Axis 1 after trim
        out_hdr['HISTORY'] = 'Overscan subtracted (median per row of cols 2048..2199) and overscan trimmed.'
        out_hdr['HISTORY'] = 'Script: overscan_subtract_trim.py'

        # Keep original dtype? To avoid clipping/quantization, we’ll write float32.
        # If you *must* keep integer type, replace dtype=np.float32 with data.dtype below (beware of clipping).
        out_data = trimmed.astype(np.float32)

        # Build a new HDUList with updated data/header in the same HDU index
        new_hdul = fits.HDUList()
        for i, h in enumerate(hdul):
            if i == hdu_idx:
                # Replace this HDU with corrected image
                if i == 0 and isinstance(h, fits.PrimaryHDU):
                    new_hdul.append(fits.PrimaryHDU(data=out_data, header=out_hdr))
                else:
                    new_hdul.append(fits.ImageHDU(data=out_data, header=out_hdr, name=h.name))
            else:
                # Keep other HDUs untouched
                # Note: if there are image extensions that referenced the original size,
                #       they remain as-is.
                new_hdul.append(hdu.copy())

    # Write out: move original to raw/, then write corrected to original path
    dirname = os.path.dirname(path)
    fname = os.path.basename(path)
    raw_path = os.path.join(dirname, 'raw')
    os.makedirs(raw_path, exist_ok=True)

    src_backup = os.path.join(raw_path, fname)
    if dry_run:
        print(f"  [dry-run] Would move original -> {src_backup}")
        print(f"  [dry-run] Would write corrected -> {path}")
        return

    # Move original file
    shutil.move(path, src_backup)

    # Write corrected file to original location (overwrite path)
    # Use overwrite=True to ensure write succeeds even if file exists (shouldn’t after move).
    new_hdul.writeto(path, overwrite=True, output_verify='fix')
    print(f"  [ok] {fname}: moved original to raw/ and wrote corrected file.")



def main(argv=None):
    parser = argparse.ArgumentParser(description="Subtract and trim overscan for FIT/FITS images.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder to scan (default: current directory)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing/moving files")
    args = parser.parse_args(argv)   # pass argv list if given, else sys.argv[1:]

    folder = os.path.abspath(args.folder)

    patterns = ["*.FIT"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))

    if not files:
        print("No FIT/FITS files found.")
        return

    print(f"Found {len(files)} files. Processing...")
    for path in sorted(files):
        try:
            process_one(path, raw_dir=os.path.join(os.path.dirname(path), "raw"), dry_run=args.dry_run)
        except Exception as e:
            print(f"  [error] {os.path.basename(path)}: {e}")

if __name__ == "__main__":
    main()
