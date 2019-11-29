import pandas as pd
import ctapipe as cta
import numpy as np
import matplotlib.pyplot as plt
import logging 
import click
from astropy.stats import sigma_clipped_stats
from ..cta_helpers import camera_to_horizontal_cta_simtel
import itertools
from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u

logger = logging.getLogger(__name__)
from ..cta_helpers import apply_parallel
from ..io import append_column_to_hdf5, HDFColumnAppender, read_telescope_data, get_column_names_in_file, remove_column_from_file, drop_prediction_column
from ..apply import predict_disp
from ..configuration import AICTConfig





def pairwise_nearest_disp(group, weights=None):
    predictions_alt = []
    predictions_az = []
    #print(group.keys())

    tel_combinations = list(itertools.combinations(group.index, 2))
    for combination in tel_combinations:
        #print(combination)
        candidate_1 = group[['source_alt_prediction', 'source_az_prediction']].loc[combination[0]]
        candidate_2 = group[['source_alt_prediction_2', 'source_az_prediction_2']].loc[combination[0]]
        candidate_3 = group[['source_alt_prediction', 'source_az_prediction']].loc[combination[1]]
        candidate_4 = group[['source_alt_prediction_2', 'source_az_prediction_2']].loc[combination[1]]
        candidates = np.array([candidate_1, candidate_2, candidate_3, candidate_4])

        disp_combinations = itertools.combinations(range(4), 2)
        min_distance = 1 * u.deg  # 0.22?
        result_alt = np.nan
        result_az = np.nan
        for pair in disp_combinations:
            distance = angular_separation(
                candidates[pair[0]][1] * u.deg,
                candidates[pair[0]][0] * u.deg,
                candidates[pair[1]][1] * u.deg,
                candidates[pair[1]][0] * u.deg)
            if distance < min_distance:
                min_distance = distance
                alt_mean = np.mean([candidates[pair[0]][0], candidates[pair[1]][0]])
                az_mean = np.mean([candidates[pair[0]][1], candidates[pair[1]][1]])
                result_alt = alt_mean
                result_az = az_mean

        predictions_alt.append(result_alt)
        predictions_az.append(result_az)
    
    result_df = pd.DataFrame()
    result_df['alt_pairwise_disp'] = [np.nanmean(predictions_alt)]
    result_df['az_pairwise_disp'] = [np.nanmean(predictions_az)]

    return result_df


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
def main(configuration_path, data_path, n_jobs):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp
    columns = [
        'run_id',
        'array-event_id',
        'source_alt_prediction',
        'source_az_prediction',
        'source_alt_prediction_2',
        'source_az_prediction_2',
    ]
    #print(columns, model_config.columns_to_read_train)

    df = read_telescope_data(
        data_path, config,
        columns,
        feature_generation_config=[],
        n_sample=model_config.n_signal
    )
    #print(df.keys())

    df_grouped = df.groupby(['run_id', 'array_event_id'])
    array_df = apply_parallel(df_grouped, pairwise_nearest_disp, n_jobs=n_jobs)
    array_df.index.names = ['run_id', 'array_event_id', None]
    array_df = array_df.drop_level(2)

    for new_feature in array_df.columns:
        append_column_to_hdf5(
            data_path, array_df[new_feature].values, config.array_events_key, new_feature
        )



