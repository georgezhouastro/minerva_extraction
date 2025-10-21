import os,sys
import yaml
import numpy as np
import argparse
import glob
import extraction
import pixel_crosscorr
import background_subtraction
import time
import psutil
import fixheader
import overscansubtract
from multiprocessing import Pool, cpu_count, TimeoutError

MIN_FREE_MEM = 10 * 1024**3  # 10 GiB in bytes
CHECK_INTERVAL = 5           # seconds between memory checks


def runreduction(fits,outputfolder,tempfolder,cfg,bkmasterfile,shiftsmodel):

    try:

        outputname = os.path.join(outputfolder,os.path.basename(fits))
        tempname = os.path.join(tempfolder,os.path.basename(fits))
        #if not os.path.exists(outputname) or cfg['clobber']:
        #if not os.path.exists(outputname) or cfg['clobber']:
        #if True:
        if not os.path.exists(outputname):
            print("Running extraction on",fits)
            bkindividualfile = os.path.join(tempfolder,os.path.basename(fits)+'_background.fits')
            background_subtraction.extract_background(fits, bkindividualfile, bkindividualfile+'.mask.png',sigma_clip=3)

            #extraction.main(cfg,fits_file=fits, output_dir=outputfolder, bkmasterfile=bkmasterfile,shifts=shiftsmodel)
            extraction.main(cfg,fits_file=fits, output_dir=tempfolder, bkmasterfile=bkindividualfile,shifts=shiftsmodel)
            pixel_crosscorr.main(cfg,tempname,chunk_size=400,snr_threshold=10,sigma_clip=3,max_iter=5,diagnostics=True,out_dir=outputfolder,diag_dir=tempfolder)

            fixheader.fixheader(outputname)

        else:
            print("Already reduced",fits)
    except Exception as e:
        print(f"[ERROR] processing {fits}: {e}", file=sys.stderr)


def wait_for_memory(threshold=MIN_FREE_MEM, interval=CHECK_INTERVAL):
    """Block until available RAM ≥ threshold."""
    while True:
        avail = psutil.virtual_memory().available
        if avail >= threshold:
            return
        print(f"[MEM LOW] Available: {avail/1024**3:.1f} GiB < {threshold/1024**3:.1f} GiB; waiting…")
        time.sleep(interval)

def parallel_extract(tasks, nproc=5, timeout=600):
    """
    Run runreduction(f, …) on each task, but only launch a new worker
    when there's ≥MIN_FREE_MEM free RAM.
    """
    results = []
    with Pool(processes=nproc) as pool:
        # dispatch each task one at a time, waiting for memory
        for args in tasks:
            wait_for_memory()
            results.append(pool.apply_async(runreduction, args))
        # now collect / enforce timeout
        try:
            for r in results:
                r.get(timeout=timeout)
        except TimeoutError:
            print(f"[TIMEOUT] did not finish within {timeout}s", file=sys.stderr)
            pool.terminate()
        else:
            print("All extractions completed within time")




def main(inputfolder,cfg):

    ### setup reduction directories
    tempfolder = os.path.join(inputfolder, "temp")
    os.makedirs(tempfolder, exist_ok=True)
    outputfolder = os.path.join(inputfolder, "output")
    os.makedirs(outputfolder, exist_ok=True)

    import overscansubtract
    overscansubtract.main([inputfolder])

    fitslist = glob.glob(os.path.join(inputfolder, "*.FIT"))

    # find the iodine calibration file (or None if not found)
    iodinecalibration = next(
        (f for f in fitslist if "Iodine" in os.path.basename(f)),
        None
    )

    # remove it from fitslist if we found one
    if iodinecalibration:
        fitslist.remove(iodinecalibration)
    else:
        raise FileNotFoundError("ERROR: No Iodine calibration file found for the night")

    ### perform vertical cc between iodine calibration on the night 
    ### and master iodine to find fiber offset

    import measure_column_shifts
    shifts,shiftsmodel = measure_column_shifts.main(iodinecalibration,cfg['rawiodinefits'],tempfolder)

    ### make a background file
    bkmasterfile = os.path.join(tempfolder,'background.fits')
    background_subtraction.extract_background(iodinecalibration, bkmasterfile, bkmasterfile+'.mask.png',sigma_clip=3)


    print("multithreading",cfg['multithread'])
    if not cfg['multithread']:
        print("Running single thread extraction")
        ### run the main extraction loop
        for fits in fitslist:
            runreduction(fits,outputfolder,tempfolder,cfg,bkmasterfile,shiftsmodel)

    else:


        print("Running multi thread extraction")
        # nproc = max(1, cpu_count() - 1)
        nproc = 10
        timeout = 600  # seconds for entire batch
        tasks = [
            (f, outputfolder, tempfolder, cfg, bkmasterfile, shiftsmodel)
            for f in fitslist
        ]

        parallel_extract(tasks, nproc=nproc, timeout=timeout)















        # print("Running multi thread extraction")

        # from multiprocessing import Pool, cpu_count, TimeoutError

        # #nproc    = max(1, cpu_count() - 1)
        # nproc = 5
        # timeout  = 600  # total seconds for entire batch
        # tasks = [
        #     (f, outputfolder, tempfolder, cfg, bkmasterfile, shiftsmodel)
        #     for f in fitslist
        # ]

        # with Pool(processes=nproc) as pool:
        #     async_map = pool.starmap_async(runreduction, tasks)
        #     try:
        #         async_map.get(timeout=timeout)
        #     except TimeoutError:
        #         print(f"[TIMEOUT] Extraction did not finish within {timeout}s", file=sys.stderr)
        #         pool.terminate()
        #     else:
        #         print("All extractions completed within time")


        # # ———> replace serial loop with parallel map <———
        # from multiprocessing import Pool, cpu_count
        # nproc = max(1, cpu_count() - 1)
        # tasks = [
        #     (f, outputfolder, tempfolder, cfg, bkmasterfile, shiftsmodel)
        #     for f in fitslist
        # ]
        # with Pool(processes=nproc) as pool:
        #     pool.starmap(runreduction, tasks)


if __name__ == "__main__":
    
    # load configuration
    p = argparse.ArgumentParser(description='Run MinervaAustralis Extraction. Last updated 20250708')
    p.add_argument('inputfolder', help='Input folder')
    p.add_argument('--config', help='YAML configuration file', default='config.yaml')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(args.inputfolder,cfg)
