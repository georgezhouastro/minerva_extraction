from astropy.io import fits
import os,sys

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas 
targetsjson = pandas.read_json("../minerva_rvs/targets.json")

def fixheader(fitsfile):
        
    # Open in update mode so we can write back new header cards
    with fits.open(fitsfile, mode='update') as hdul:
        hdr = hdul[0].header
        try:
            hdr['OBJECT']
            hdr['RA']
            hdr['DEC']
            hdr['EXPSTART']
            hdr['EXPEND']
            hdr['DATETIME']
            hdr['EXPLENGT']
        except KeyError:
            objectname = os.path.basename(fitsfile)
            objectname = str.split(objectname,"_")[1:]
            objectname = '_'.join(objectname)
            objectname = str.replace(objectname,".FIT","")
            #if "_" in objectname:
            #    objectname = str.split(objectname,"_")[0]

            hdr['OBJECT'] = " "+objectname
            mask = targetsjson['name'] == objectname

            if sum(mask) == 1:
                ra = str(targetsjson[mask]['ra'].iloc[0])
                dec = str(targetsjson[mask]['dec'].iloc[0])
                vmag= float(targetsjson[mask]['Vmag'].iloc[0])

                hdr['RA'] = ra
                hdr['DEC'] = dec
                hdr['VMAG'] = vmag

            else:
                print("object not found")


            # 1) Local timezone for Queensland (no DST)
            local_tz = ZoneInfo("Australia/Brisbane")

            # 2) Parse the start time from DATE-OBS (ISO format)
            dt_start_local = datetime.fromisoformat(hdr["DATE-OBS"])
            dt_start_local = dt_start_local.replace(tzinfo=local_tz)

            # 3) Parse the stop time from the TIME card ("HH:MM:SS.sss to HH:MM:SS.sss")
            time_range = hdr["TIME"]
            t0_str, t1_str = [t.strip() for t in time_range.split("to", 1)]
            date_str = hdr.get("DATE", dt_start_local.date().isoformat())
            dt_end_local = datetime.fromisoformat(f"{date_str}T{t1_str}")
            dt_end_local = dt_end_local.replace(tzinfo=local_tz)

            # handle exposures crossing UTC midnight
            if dt_end_local < dt_start_local:
                dt_end_local += timedelta(days=1)

            # 4) Convert both to UTC
            dt_start_utc = dt_start_local.astimezone(timezone.utc)
            dt_end_utc   = dt_end_local.astimezone(timezone.utc)

            # 5) Format as ISO with millisecond precision + 'Z'
            def fmt(t):
                return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            expstart = " " +fmt(dt_start_utc)
            expend   = " " +fmt(dt_end_utc)

            # 6) Compute exposure length in minutes (float → round to nearest int)
            delta_min = (dt_end_local - dt_start_local).total_seconds() / 60.0
            explengt  = str(int(round(delta_min)))

            # 7) Write new header cards
            hdr["EXPSTART"] = (expstart, "Exposure start (UTC)")
            hdr["EXPEND"]   = (expend,   "Exposure end   (UTC)")
            hdr["EXPLENGT"] = (explengt, "Exposure length (min)")

            # changes are auto‑written on close()


if __name__ == "__main__":
    
    fixheader("/media/koi368/minerva_spec/2025/202507/20250723/output/20250724T025804_TOI222_01.FIT")
