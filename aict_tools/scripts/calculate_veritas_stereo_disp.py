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
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from ..cta_helpers import apply_parallel
from ..io import append_column_to_hdf5, HDFColumnAppender, read_telescope_data, get_column_names_in_file, remove_column_from_file, drop_prediction_column
from ..apply import predict_disp
from ..configuration import AICTConfig


logger = logging.getLogger(__name__)


def biggest_cluster_mean(group, eps=1.0):
    group = group.dropna()
    X = group[['source_alt_prediction', 'source_az_prediction']].dropna(how='any').values
    X2 = group[['source_alt_prediction_2', 'source_az_prediction_2']].dropna(how='any').values
    X = np.concatenate([X,X2], axis=0)

    weights = group['weights']

    result_df = pd.DataFrame()
    result_df['source_alt_cluster'] = [np.nan]
    result_df['source_az_cluster'] = [np.nan]

    if len(X) <1 :
       return result_df

    #min_samples = int(len(group)*0.5) if len(group) > 4 else 2
    min_samples=2
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    unique_labels = set(labels)
    main_id = (0,0)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    for k in unique_labels:

        class_member_mask = (labels == k)
        if class_member_mask.sum() > main_id[1]:
            main_id = (k, class_member_mask.sum())


    class_member_mask = (labels == main_id[0])

    if X[class_member_mask].any():
        mean = np.nanmean(X[class_member_mask], axis=0)
        result_df['source_alt_cluster'] = [mean[0]]
        result_df['source_az_cluster'] = [mean[1]]
    result_df['cluster_size'] = main_id[1]


    return result_df


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-w', '--weights', type=str, help='weights for averaging')
@click.option('-e', '--eps', default=1.0, multiple=True)
def main(configuration_path, data_path, n_jobs=1, weights=None, eps=(1.0,)):
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
        'num_triggered_telescopes',
        'num_triggered_lst',
        'num_triggered_mst',
        'num_triggered_sst',
    ]
    df = read_telescope_data(
        data_path, config,
        columns,
        feature_generation_config=[],
        n_sample=model_config.n_signal
    )

    if weights == 'lw_logsize':
        return 0
        df['weights'] = (df['length']/df['width']*np.log10(df['intensity'])).values
    elif weights == 'sign_proba':
        return 0
        df['weights'] = (df['sign_proba']).values
    else:
        df['weights'] = 1

    for eps_ in eps:
        df_grouped = df.groupby(['run_id', 'array_event_id'], sort=False)
        array_df = apply_parallel(df_grouped, biggest_cluster_mean, n_jobs=n_jobs, eps=eps_)
        #array_df.index.names = ['run_id', 'array_event_id', None]
        #array_df = array_df.droplevel(2)

        for new_feature in array_df.columns:
            if weights:
                name = new_feature + '_' + str(weights) + '_' + str(eps_)
            else:
                name = new_feature + '_' + str(eps_)
            append_column_to_hdf5(
                data_path, array_df[new_feature].values, config.array_events_key, name
            )
