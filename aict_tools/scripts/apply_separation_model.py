import click
from sklearn.externals import joblib
import logging
from tqdm import tqdm
import pandas as pd

from ..apply import predict_separator
from ..io import append_column_to_hdf5, read_telescope_data_chunked, drop_prediction_column, HDFColumnAppender
from ..configuration import AICTConfig


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once'
)
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
def main(configuration_path, data_path, model_path, chunksize, yes, verbose):
    '''
    Apply loaded model to data.

    CONFIGURATION_PATH: Path to the config yaml file.
    DATA_PATH: path to the FACT/CTA data.
    MODEL_PATH: Path to the pickled model.

    The program adds the following columns to the inputfile:
        <class_name>_prediction: the output of model.predict_proba for the
        class name given in the config file.

    If the class name is not given in the config file, the default value of "gamma"
    will be used.
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.separator

    prediction_column_name = config.class_name + '_prediction'

    if config.experiment_name.lower() == 'cta':
        group_name = config.array_events_key
    else:
        group_name = config.telescope_events_key

    drop_prediction_column(
        data_path, group_name=group_name, 
        column_name=prediction_column_name, yes=yes
    )

    log.debug('Loading model')
    model = joblib.load(model_path)
    log.debug('Loaded model')

    df_generator = read_telescope_data_chunked(
        data_path, config, chunksize, model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation
    )

    # collect predictions to calculate mean/std
    if config.experiment_name.lower() == 'cta':
        chunked_frames = []

    with HDFColumnAppender(data_path, config.telescope_events_key) as appender:
        for df_data, start, stop in tqdm(df_generator):

            prediction = predict_separator(df_data[model_config.features], model)

            if config.experiment_name.lower() == 'cta':
                d = df_data[['run_id', 'array_event_id']].copy()
                d[prediction_column_name] = prediction
                chunked_frames.append(d)

            appender.add_data(prediction, prediction_column_name, start, stop)

    # combine predictions
    if config.experiment_name.lower() == 'cta':
        d = pd.concat(chunked_frames).groupby(
            ['run_id', 'array_event_id'], sort=False
        ).agg(['mean', 'std'])
        mean = d[prediction_column_name]['mean'].values
        std = d[prediction_column_name]['std'].values

        append_column_to_hdf5(
            data_path, mean, config.array_events_key, prediction_column_name + '_mean'
        )
        append_column_to_hdf5(
            data_path, std, config.array_events_key, prediction_column_name + '_std'
        )


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
