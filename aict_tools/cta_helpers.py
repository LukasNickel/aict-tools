import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.time import Time
import logging

from astropy.coordinates.representation import REPRESENTATION_CLASSES
REPRESENTATION_CLASSES.pop("planar")
from ctapipe.coordinates import CameraFrame
from ctapipe.instrument import TelescopeDescription


log = logging.getLogger(__name__)


## maybe use multiprocessing for the array event loop?
def horizontal_to_camera_cta_simtel(df, config, model_config):
    source_x = []
    source_y = []
    # using these as a placeholder
    # necessary to use the coord trafos but not actually used?
    obstime = Time('2013-11-01T03:00')
    location = EarthLocation.of_site('Roque de los Muchachos')

    array_ids = df[config.array_event_column].unique()
    id_to_tel = config.id_to_tel
    id_to_cam = config.id_to_cam
    for array_id in array_ids:
        tel_event_rows = np.where(df.array_event_id.values == array_id)[0]
        for tel_event in tel_event_rows:  # tel_event is the index of the row of the actual tel_event
            # construct SkyCoord
            alt_pointing = df.iloc[tel_event].pointing_altitude * u.rad
            az_pointing = df.iloc[tel_event].pointing_azimuth * u.rad
            tel_pointing = SkyCoord(
                alt=alt_pointing,
                az=az_pointing,
                frame=AltAz(
                    obstime=obstime,
                    location=location)
            )
            # construct camera frame
            focal_length = df.iloc[tel_event].focal_length * u.m
            rotation = 0 * u.deg # always?
            camera_frame = CameraFrame(
                focal_length=focal_length,
                rotation=rotation,
                telescope_pointing=tel_pointing,
            )
            # construct altaz frame
            mc_alt = df.iloc[tel_event][model_config.source_zd_column] * u.deg  #zd == alt for now (only a name anyway)
            mc_az = df.iloc[tel_event][model_config.source_az_column] * u.deg
            altaz = AltAz(
                    az=mc_az,
                    alt=mc_alt,
                    location=location,
                    obstime=obstime,
            )
            cam_coords = altaz.transform_to(camera_frame)
            source_x.append(cam_coords.x.value)
            source_y.append(cam_coords.y.value)
    return np.array(source_x), np.array(source_y)
