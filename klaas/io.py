from os import path
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import logging
import numpy as np
from .feature_generation import feature_generation
from fact.io import read_data, h5py_get_n_rows
import pandas as pd
import h5py
import click
__all__ = ['pickle_model']


log = logging.getLogger(__name__)


def check_existing_column(data_path, config, yes):
    prediction_column_name = config.class_name + '_prediction'
    with h5py.File(data_path, 'r+') as f:
        columns = f[config.telescope_events_key].keys()
        if prediction_column_name in columns:
            if not yes:
                click.confirm(
                    f'Column \"{prediction_column_name}\" exists in file, overwrite?', abort=True,
                )

            del f[config.telescope_events_key][prediction_column_name]
            if prediction_column_name + '_std' in columns:
                del f[config.telescope_events_key][prediction_column_name + '_std']
            if prediction_column_name + '_mean' in columns:
                del f[config.telescope_events_key][prediction_column_name + '_mean']
            if prediction_column_name + '_avg' in columns:
                del f[config.telescope_events_key][prediction_column_name + '_avg']


def read_telescope_data_chunked(path, klaas_config, chunksize, columns):
    n_rows = h5py_get_n_rows(path, klaas_config.telescope_events_key)
    if chunksize:
        n_chunks = int(np.ceil(n_rows / chunksize))
    else:
        n_chunks = 1
        chunksize = n_rows
    log.info('Splitting data into {} chunks'.format(n_chunks))

    for chunk in range(n_chunks):

        start = chunk * chunksize
        end = min(n_rows, (chunk + 1) * chunksize)

        df = read_telescope_data(
            path,
            klaas_config=klaas_config,
            columns=columns,
            first=start,
            last=end
        )
        df.index = np.arange(start, end)

        yield df, start, end


def read_telescope_data(path, klaas_config, n_sample=None, columns=None, first=None, last=None):
    '''
    Read given columns from data and perform a random sample if n_sample is supplied.
    Returns a single pandas data frame
    '''
    telescope_event_columns = None
    array_event_columns = None
    if klaas_config.has_multiple_telescopes:
        if columns:
            with h5py.File(path, 'r+') as f:
                array_event_columns = set(f[klaas_config.array_events_key].keys()) & set(columns)
                telescope_event_columns = set(f[klaas_config.telescope_events_key].keys()) & set(columns)

        telescope_events = read_data(
            file_path=path,
            key=klaas_config.telescope_events_key,
            columns=telescope_event_columns,
            first=first,
            last=last,
        )
        array_events = read_data(
            file_path=path,
            key=klaas_config.array_events_key,
            columns=array_event_columns,
        )

        keys = [klaas_config.run_id_key, klaas_config.array_event_id_key]
        df = pd.merge(left=array_events, right=telescope_events, left_on=keys, right_on=keys)

    else:
        df = read_data(
            file_path=path,
            key=klaas_config.telescope_events_key,
            columns=klaas_config.columns_to_read,
            first=first,
            last=last,
        )

    if n_sample is not None:
        if n_sample > len(df):
            log.error(
                'number of sampled events {} must be smaller than number events in file {} ({})'
                .format(n_sample, path, len(df))
            )
            raise ValueError
        log.info('Randomly sample {} events'.format(n_sample))
        df = df.sample(n_sample)

    # generate features if given in config
    if klaas_config.feature_generation_config:
        feature_generation(df, klaas_config.feature_generation_config, inplace=True)

    return df


def pickle_model(classifier, feature_names, model_path, label_text='label'):
    p, extension = path.splitext(model_path)
    classifier.feature_names = feature_names

    if (extension == '.pmml'):
        joblib.dump(classifier, p + '.pkl', compress=4)

        pipeline = PMMLPipeline([
            ('classifier', classifier)
        ])
        pipeline.target_field = label_text
        pipeline.active_fields = np.array(feature_names)
        sklearn2pmml(pipeline, model_path)

    else:
        joblib.dump(classifier, model_path, compress=4)


def append_to_h5py(f, array, group, key):
    '''
    Write numpy array to h5py hdf5 file
    '''
    group = f.require_group(group)  # create if not exists

    max_shape = list(array.shape)
    max_shape[0] = None

    if key not in group.keys():
        group.create_dataset(
            key,
            data=array,
            maxshape=tuple(max_shape),
        )
    else:
        n_existing = group[key].shape[0]
        n_new = array.shape[0]

        group[key].resize(n_existing + n_new, axis=0)
        group[key][n_existing:n_existing + n_new] = array
