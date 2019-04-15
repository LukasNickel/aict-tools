import click
import numpy as np
from sklearn.externals import joblib
import logging
from tqdm import tqdm
import pandas as pd

from ..io import append_column_to_hdf5, HDFColumnAppender, read_telescope_data_chunked, get_column_names_in_file, remove_column_from_file, drop_prediction_column
from ..apply import predict_disp
from ..configuration import AICTConfig


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once',
)
def main(configuration_path, data_path, disp_model_path, sign_model_path, chunksize, n_jobs, yes, verbose):
    '''
    Apply given model to data. Two columns are added to the file, energy_prediction
    and energy_prediction_std

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    DISP_MODEL_PATH: Path to the pickled disp model.
    SIGN_MODEL_PATH: Path to the pickled sign model.
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp

    columns_to_delete = [
        'source_x_prediction',
        'source_y_prediction',
        'theta',
        'theta_deg',
        'theta_rec_pos',
        'disp_prediction',
    ]
    for i in range(1, 6):
        columns_to_delete.extend([
            'theta_off_' + str(i),
            'theta_deg_off_' + str(i),
            'theta_off_rec_pos_' + str(i),
        ])

    n_del_cols = 0

    for column in columns_to_delete:
        if config.has_multiple_telescopes:
            drop_prediction_column(
                data_path, group_name=config.array_events_key,
                column_name=column, yes=yes
            )
            log.warn("Deleted {} from the feature set.".format(column))
            n_del_cols += 1
        else:
            drop_prediction_column(
                data_path, group_name=config.telescope_events_key,
                column_name=column, yes=yes
            )            
            log.warn("Deleted {} from the feature set.".format(column))
            n_del_cols += 1
        # if column in get_column_names_in_file(data_path, config.telescope_events_key):
        #     if not yes:
        #         click.confirm(
        #             'Dataset "{}" exists in file, overwrite?'.format(column),
        #             abort=True,
        #         )
        #         yes = True
        #     remove_column_from_file(data_path, config.telescope_events_key, column)
        #     log.warn("Deleted {} from the feature set.".format(column))
        #     n_del_cols += 1

    if n_del_cols > 0:
        log.warn("Source dependent features need to be calculated from the predicted source possition. "
                 + "Use e.g. `fact_calculate_theta` from https://github.com/fact-project/pyfact.")

    log.info('Loading model')
    disp_model = joblib.load(disp_model_path)
    sign_model = joblib.load(sign_model_path)
    log.info('Done')

    if n_jobs:
        disp_model.n_jobs = n_jobs
        sign_model.n_jobs = n_jobs

    df_generator = read_telescope_data_chunked(
        data_path, config, chunksize, model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation
    )

    log.info('Predicting on data...')

    cog_x_column = model_config.cog_x_column
    cog_y_column = model_config.cog_y_column
    delta_column = model_config.delta_column
    if config.has_multiple_telescopes:
        chunked_frames = []

    with HDFColumnAppender(data_path, config.telescope_events_key) as appender:
        for df_data, start, stop in tqdm(df_generator):
            disp = predict_disp(
                df_data[model_config.features], disp_model, sign_model
            )

            source_x = df_data[cog_x_column] + disp * np.cos(df_data[delta_column])
            source_y = df_data[cog_y_column] + disp * np.sin(df_data[delta_column])

            if config.has_multiple_telescopes:
                d = df_data[['run_id', 'array_event_id']].copy()
                
                d['disp'] = disp
                d['source_x'] = source_x
                d['source_y'] = source_y
                chunked_frames.append(d)
                log.info(d.keys()) ###looks good
            appender.add_data(source_x, 'source_x_prediction', start, stop)
            appender.add_data(source_y, 'source_y_prediction', start, stop)
            appender.add_data(disp, 'disp_prediction', start, stop)
        log.info('loop done')
    log.info('with() done')

    
    if config.has_multiple_telescopes:
        log.info(d.keys())
        d = pd.concat(chunked_frames).groupby(
            ['run_id', 'array_event_id'], sort=False
        ).agg(['mean', 'std'])

        append_column_to_hdf5(
            data_path, d['disp']['mean'].values, config.array_events_key, 'disp' + '_mean'
        )
        append_column_to_hdf5(
            data_path, d['disp']['std'].values, config.array_events_key, 'disp' + '_std'
        )
        append_column_to_hdf5(
            data_path, d['source_x']['mean'].values, config.array_events_key, 'source_x' + '_mean'
        )
        append_column_to_hdf5(
            data_path, d['source_x']['std'].values, config.array_events_key, 'source_x' + '_std'
        )
        append_column_to_hdf5(
            data_path, d['source_y']['mean'].values, config.array_events_key, 'source_y' + '_mean'
        )
        append_column_to_hdf5(
            data_path, d['source_y']['std'].values, config.array_events_key, 'source_y' + '_std'
        )

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
