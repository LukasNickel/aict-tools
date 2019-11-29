import click
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd

from ..io import (
    append_column_to_hdf5,
    read_telescope_data_chunked,
    get_column_names_in_file,
    remove_column_from_file,
    load_model,
)
from ..apply import predict_disp
from ..configuration import AICTConfig
from ..logging import setup_logging


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
    log = setup_logging(verbose=verbose)

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp

    columns_to_delete = [
        'source_x_prediction',
        'source_y_prediction',
        'source_alt_prediction',
        'source_az_prediction',
        'theta',
        'theta_deg',
        'theta_rec_pos',
        'disp_prediction',
        'sign_prediction',
    ]
    for i in range(1, 6):
        columns_to_delete.extend([
            'theta_off_' + str(i),
            'theta_deg_off_' + str(i),
            'theta_off_rec_pos_' + str(i),
        ])

    n_del_cols = 0

    for column in columns_to_delete:
        if column in get_column_names_in_file(data_path, config.telescope_events_key):
            if not yes:
                click.confirm(
                    'Dataset "{}" exists in file, overwrite?'.format(column),
                    abort=True,
                )
                yes = True
            remove_column_from_file(data_path, config.telescope_events_key, column)
            log.warn("Deleted {} from the feature set.".format(column))
            n_del_cols += 1

    if n_del_cols > 0:
        log.warn("Source dependent features need to be calculated from the predicted source possition. "
                 + "Use e.g. `fact_calculate_theta` from https://github.com/fact-project/pyfact.")


    log.info('Loading model')
    disp_model = load_model(disp_model_path)
    sign_model = load_model(sign_model_path)
    log.info('Done')

    if n_jobs:
        disp_model.n_jobs = n_jobs
        sign_model.n_jobs = n_jobs

    df_generator = read_telescope_data_chunked(
        data_path, config, chunksize, model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation
    )

    log.info('Predicting on data...')


    # for cta collect results to calculate mean and std later
    if config.has_multiple_telescopes == True:
        chunked_frames = []

    for df_data, start, stop in tqdm(df_generator):
        if config.has_multiple_telescopes == True:
            df_data[model_config.delta_column] = np.deg2rad(df_data[model_config.delta_column])
        disp = predict_disp(
            df_data[model_config.features], disp_model, sign_model,
            log_target=model_config.log_target,
        )
        #from IPython import embed; embed()
        source_x = df_data[model_config.cog_x_column] + disp * np.cos(df_data[model_config.delta_column])
        source_y = df_data[model_config.cog_y_column] + disp * np.sin(df_data[model_config.delta_column])

        key = config.telescope_events_key
        append_column_to_hdf5(data_path, source_x, key, 'source_x_prediction')
        append_column_to_hdf5(data_path, source_y, key, 'source_y_prediction')
        append_column_to_hdf5(data_path, disp, key, 'disp_prediction')

        # collect alt/az predictions to save mean/std over telescope predictions
        if config.has_multiple_telescopes == True:
            d = df_data[['run_id', 'array_event_id']].copy()
            from ..cta_helpers import camera_to_horizontal_cta_simtel

            df_data['source_x_prediction'] = source_x
            df_data['source_y_prediction'] = source_y
            source_alt, source_az = camera_to_horizontal_cta_simtel(df_data)                
            d['source_alt'] = source_alt
            d['source_az'] = source_az

            source_x_2 = df_data[model_config.cog_x_column] - disp * np.cos(df_data[model_config.delta_column])
            source_y_2 = df_data[model_config.cog_y_column] - disp * np.sin(df_data[model_config.delta_column])

            df_data['source_x_prediction_2'] = source_x_2
            df_data['source_y_prediction_2'] = source_y_2
            source_alt_2, source_az_2 = camera_to_horizontal_cta_simtel(df_data, x_key='source_x_prediction_2', y_key='source_y_prediction_2')
            d['source_alt_2'] = source_alt_2
            d['source_az_2'] = source_az_2

            append_column_to_hdf5(data_path, source_alt, key, 'source_alt_prediction')
            append_column_to_hdf5(data_path, source_az, key, 'source_az_prediction')
            append_column_to_hdf5(data_path, source_alt_2, key, 'source_alt_prediction_2')
            append_column_to_hdf5(data_path, source_az_2, key, 'source_az_prediction_2')
            append_column_to_hdf5(data_path, source_x_2, key, 'source_x_prediction_2')
            append_column_to_hdf5(data_path, source_y_2, key, 'source_y_prediction_2')
            
            chunked_frames.append(d)
    
    if config.has_multiple_telescopes == True:
        d = pd.concat(chunked_frames)
        d = d.groupby(
            ['run_id', 'array_event_id'], sort=False
        )


        d_ = d.agg(['mean', 'std', 'median'])

        append_column_to_hdf5(
             data_path, d_['source_alt']['mean'].values, config.array_events_key, 'source_alt' + '_mean'
        )
        append_column_to_hdf5(
             data_path, d_['source_alt']['median'].values, config.array_events_key, 'source_alt' + '_median'
        )
        append_column_to_hdf5(
            data_path, d_['source_alt']['std'].values, config.array_events_key, 'source_alt' + '_std'
        )

        append_column_to_hdf5(
            data_path, d_['source_az']['mean'].values, config.array_events_key, 'source_az' + '_mean'
        )
        append_column_to_hdf5(
            data_path, d_['source_az']['std'].values, config.array_events_key, 'source_az' + '_std'
        )
        append_column_to_hdf5(
            data_path, d_['source_az']['median'].values, config.array_events_key, 'source_az' + '_median'
        )


if __name__ == '__main__':
    main()
