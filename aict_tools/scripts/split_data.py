import os

import click
import numpy as np
import logging

from ..io import read_data, write_hdf, read_data_chunked
from fact.io import write_data as write_data_fact

import warnings
from math import ceil
from tqdm import tqdm

log = logging.getLogger()


def split_indices(idx, n_total, fractions):
    '''
    splits idx containing n_total distinct events into fractions given in fractions list.
    returns the number of events in each split
    '''
    num_ids = [ceil(n_total * f) for f in fractions]
    if sum(num_ids) > n_total:
        num_ids[-1] -= sum(num_ids) - n_total
    return num_ids


@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=True))
@click.argument('output_basename')
@click.option(
    '--fraction', '-f', multiple=True, type=float,
    help='Fraction of events to use for this part'
)
@click.option(
    '--name',
    '-n',
    multiple=True,
    help='name for one dataset'
)
@click.option(
    '-i',
    '--inkey',
    help='HDF5 key for h5py hdf5 of the input file',
    default='events', show_default=True,
)
@click.option(
    '--key',
    '-k',
    help='Name for the hdf5 group in the output',
    default='events',
    show_default=True,
)
@click.option(
    '--telescope', '-t', type=click.Choice(['fact', 'cta']), default='fact',
    show_default=True, help='Which telescope created the data',
)
@click.option('-s', '--seed', help='Random Seed', type=int, default=0, show_default=True)
@click.option('-v', '--verbose', is_flag=True, help='Verbose log output',)
@click.option('--format',  default='h5py', type=click.Choice(['h5py', 'tables']),)
def main(input_path, output_basename, fraction, name, inkey, key, telescope, seed, verbose, format):
    '''
    Split dataset in INPUT_PATH into multiple parts for given fractions and names
    Outputs hdf5 or csv files to OUTPUT_BASENAME_NAME.FORMAT

    Example call: aict_split_data input.hdf5 output_base -n test -f 0.5 -n train -f 0.5
    '''

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    log.debug("input_path: {}".format(input_path))

    np.random.seed(seed)
    
    use_h5py = format == 'h5py' 

    if telescope == 'fact':
        split_single_telescope_data(input_path, output_basename, inkey, key, fraction, name, use_h5py=use_h5py)
    else:
        split_multi_telescope_data(input_path, output_basename, fraction, name, use_h5py=use_h5py)


def split_multi_telescope_data(input_path, output_basename, fraction, name, use_h5py=True):

    _, file_extension = os.path.splitext(input_path)

    array_events = read_data(input_path, key='array_events')
    runs = read_data(input_path, key='runs')

    # split by runs

    ids = set(runs.run_id)
    log.debug(f'All runs:{ids}')
    n_total = len(runs)

    log.info(f'Found a total of {n_total} runs in the file')
    num_runs = split_indices(ids, n_total, fractions=fraction)

    for n, part_name in zip(num_runs, name):
        selected_run_ids = np.random.choice(list(ids), size=n, replace=False)
        selected_runs = runs[runs.run_id.isin(selected_run_ids)]
        selected_array_events = array_events[array_events.run_id.isin(selected_run_ids)]

        path = output_basename + '_' + part_name + file_extension
        log.info('Writing {} runs events to: {}'.format(n, path))
        write_hdf(selected_runs, path, table_name='runs', use_h5py=use_h5py, mode='w')
        write_hdf(selected_array_events, path, table_name='array_events', use_h5py=use_h5py, mode='a',)

        for telescope_events, _, _ in tqdm(read_data_chunked(input_path, table_name='telescope_events', chunksize=300000)):
            selected_telescope_events = telescope_events[telescope_events.run_id.isin(selected_run_ids)]
            write_hdf(selected_telescope_events, path, table_name='telescope_events', use_h5py=use_h5py, mode='a',)

        log.debug(f'selected runs {set(selected_run_ids)}')
        log.debug(f'Runs minus selected runs {ids - set(selected_run_ids)}')
        ids = ids - set(selected_run_ids)


def split_single_telescope_data(input_path, output_basename, inkey, key, fraction, name, use_h5py=True):

    _, file_extension = os.path.splitext(input_path)

    data = read_data(input_path, key=inkey)
    assert len(fraction) == len(name), 'You must give a name for each fraction'

    if sum(fraction) != 1:
        warnings.warn('Fractions do not sum up to 1')

    ids = data.index.values
    n_total = len(data)

    log.info('Found a total of {} single-telescope events in the file'.format(len(data)))

    num_ids = split_indices(ids, n_total, fractions=fraction)

    for n, part_name in zip(num_ids, name):
        selected_ids = np.random.choice(ids, size=n, replace=False)
        selected_data = data.loc[selected_ids]


        path = output_basename + '_' + part_name + file_extension
        log.info('Writing {} telescope-array events to: {}'.format(n, path))
        write_data_fact(selected_data, path, key=key, use_h5py=use_h5py, mode='w')

        data = data.loc[list(set(data.index.values) - set(selected_data.index.values))]
        ids = data.index.values
