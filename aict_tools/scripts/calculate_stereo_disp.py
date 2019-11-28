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

from ..cta_helpers import apply_parallel
from ..io import append_column_to_hdf5, HDFColumnAppender, read_telescope_data, get_column_names_in_file, remove_column_from_file, drop_prediction_column
from ..apply import predict_disp
from ..configuration import AICTConfig


logger = logging.getLogger(__name__)


def pairwise_nearest_disp(group, eps=1.0):
    predictions_alt = []
    predictions_az = []
    alt_disps = []
    az_disps = []

    tel_combinations = list(itertools.combinations(group.index, 2))
    for combination in tel_combinations:
        candidate_1 = group[['source_alt_prediction', 'source_az_prediction']].loc[combination[0]]
        candidate_2 = group[['source_alt_prediction_2', 'source_az_prediction_2']].loc[combination[0]]
        candidate_3 = group[['source_alt_prediction', 'source_az_prediction']].loc[combination[1]]
        candidate_4 = group[['source_alt_prediction_2', 'source_az_prediction_2']].loc[combination[1]]
        candidates = np.array([candidate_1, candidate_2, candidate_3, candidate_4])

        disp_combinations = itertools.combinations(range(4), 2)
        min_distance = eps * u.deg  # 0.22 in magic?
        result_alt = np.nan
        result_az = np.nan
        winner = None
        for pair in disp_combinations:
            distance = angular_separation(
                candidates[pair[0]][1] * u.deg,
                candidates[pair[0]][0] * u.deg,
                candidates[pair[1]][1] * u.deg,
                candidates[pair[1]][0] * u.deg).to(u.deg)
            if distance < min_distance:
                min_distance = distance
                winner = pair

        if winner:
            alt_disps.append(candidates[winner[0]][0])
            alt_disps.append(candidates[winner[1]][0])
            az_disps.append(candidates[winner[0]][1])
            az_disps.append(candidates[winner[1]][1])


    result_df = pd.DataFrame()
    if alt_disps and az_disps:
        result_df['source_alt_pairwise_mean'] = [np.nanmean(alt_disps)]
        result_df['source_az_pairwise_mean'] = [np.nanmean(az_disps)]
        result_df['source_alt_pairwise_median'] = [np.nanmedian(alt_disps)]
        result_df['source_az_pairwise_median'] = [np.nanmedian(az_disps)]
        result_df['source_alt_pairwise_clipped'] = [sigma_clipped_stats(alt_disps)[0]]
        result_df['source_az_pairwise_clipped'] = [sigma_clipped_stats(az_disps)[0]]
        result_df['source_alt_pairwise_clipped_median'] = [sigma_clipped_stats(alt_disps)[1]]
        result_df['source_az_pairwise_clipped_median'] = [sigma_clipped_stats(az_disps)[1]]
        result_df['source_alt_pairwise_set'] = [np.nanmean(list(set(alt_disps)))]
        result_df['source_az_pairwise_set'] = [np.nanmean(list(set(az_disps)))]
    else:
        result_df['source_alt_pairwise_mean'] = [np.nan]
        result_df['source_az_pairwise_mean'] = [np.nan]
        result_df['source_alt_pairwise_median'] = [np.nan]
        result_df['source_az_pairwise_median'] = [np.nan]
        result_df['source_alt_pairwise_clipped'] = [np.nan]
        result_df['source_az_pairwise_clipped'] = [np.nan]
        result_df['source_alt_pairwise_clipped_median'] = [np.nan]
        result_df['source_az_pairwise_clipped_median'] = [np.nan]
        result_df['source_alt_pairwise_set'] = [np.nan]
        result_df['source_az_pairwise_set'] = [np.nan]
    result_df['num_pairs'] = len(alt_disps)

    return result_df


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-e', '--eps', default=1.0, multiple=True)
def main(configuration_path, data_path, n_jobs=1, eps=(1.0,)):
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
        'length',
        'width',
        'intensity',
        'sign_proba',
        'mc_alt'
    ]
    df = read_telescope_data(
        data_path, config,
        columns,
        feature_generation_config=[],
        n_sample=model_config.n_signal
    )

    for eps_ in eps:
        df_grouped = df.groupby(['run_id', 'array_event_id'], sort=False)  ##sort false is important!!
        array_df = apply_parallel(df_grouped, pairwise_nearest_disp, n_jobs=n_jobs, eps=eps_)

        for new_feature in array_df.columns:
            name = new_feature + '_' + str(eps_)
            append_column_to_hdf5(
                data_path, array_df[new_feature].values, config.array_events_key, name
            )
