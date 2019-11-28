import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.time import Time
import logging
import warnings

from ctapipe.coordinates import MissingFrameAttributeWarning
from ctapipe.coordinates import CameraFrame
from ctapipe.instrument import TelescopeDescription

from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

log = logging.getLogger(__name__)


def horizontal_to_camera_cta_simtel(df):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MissingFrameAttributeWarning)

        alt_pointing = u.Quantity(df.pointing_altitude.to_numpy(), u.rad, copy=False)
        az_pointing = u.Quantity(df.pointing_azimuth.to_numpy(), u.rad, copy=False)
        fl = u.Quantity(df.focal_length.to_numpy(), u.m, copy=False)
        mc_alt = u.Quantity(df.mc_alt.to_numpy(), u.deg, copy=False)
        mc_az = u.Quantity(df.mc_az.to_numpy(), u.deg, copy=False)

        altaz = AltAz()
        
        tel_pointing = SkyCoord(
            alt=alt_pointing,
            az=az_pointing,
            frame=altaz,
        )
        camera_frame = CameraFrame(
            focal_length=fl,
            telescope_pointing=tel_pointing,
        )
        
        source_altaz = SkyCoord(
            az=mc_az,
            alt=mc_alt,
            frame=altaz,
        )
        
        cam_coords = source_altaz.transform_to(camera_frame)
        return cam_coords.x.to_value(u.m), cam_coords.y.to_value(u.m)



def camera_to_horizontal_cta_simtel(df, x_key='source_x_prediction', y_key='source_y_prediction'):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=MissingFrameAttributeWarning)

        alt_pointing = u.Quantity(df.pointing_altitude.to_numpy(), u.rad, copy=False)
        az_pointing = u.Quantity(df.pointing_azimuth.to_numpy(), u.rad, copy=False)
        x = u.Quantity(df[x_key].to_numpy(), u.m, copy=False)
        y = u.Quantity(df[y_key].to_numpy(), u.m, copy=False)
        fl = u.Quantity(df.focal_length.to_numpy(), u.m, copy=False)
        
        altaz = AltAz()

        tel_pointing = SkyCoord(
            alt=alt_pointing,
            az=az_pointing,
            frame=altaz,
        )

        frame = CameraFrame(
            focal_length = fl,
            telescope_pointing=tel_pointing,
        )


        cam_coords = SkyCoord(
                x=x,
                y=y,
                frame=frame,
        )

        source_altaz = cam_coords.transform_to(altaz)

        # rad verwenden? 
        return source_altaz.alt.to_value(u.deg), source_altaz.az.to_value(u.deg)



'thanks SO: https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby'
def temp_func(func, name, group, **kwargs):
    #print([(x, type(x)) for x in kwargs])
    return func(group, **kwargs), name 

def apply_parallel(dfGrouped, func, n_jobs=1, **kwargs):
    retLst, top_index = zip(*Parallel(n_jobs=n_jobs)(delayed(temp_func)(func, name, group, **kwargs) for name, group in tqdm(dfGrouped)))
    return pd.concat(retLst, keys=top_index, sort=False)