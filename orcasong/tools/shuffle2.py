import os
import time
import datetime

import numpy as np
import psutil
import h5py
from km3pipe.sys import peak_memory_usage

from orcasong.tools.postproc import get_filepath_output, copy_used_files
from orcasong.tools.concatenate import copy_attrs


__author__ = "Stefan Reck"


def h5shuffle2(
    input_file,
    output_file=None,
    iterations=None,
    datasets=("x", "y"),
    max_ram_fraction=0.25,
    max_ram=None,
    seed=42,
):
    """
    Shuffle datasets in a h5file that have the same length.

    Parameters
    ----------
    input_file : str
        Path of the file that will be shuffle.
    output_file : str, optional
        If given, this will be the name of the output file.
        Otherwise, a name is auto generated.
    iterations : int, optional
        Shuffle the file this many times. For each additional iteration,
        a temporary file will be created and then deleted afterwards.
        Default: Auto choose best number based on available RAM.
    datasets : tuple
        Which datasets to include in output.
    max_ram : int, optional
        Available ram in bytes. Default: Use fraction of
        maximum available (see max_ram_fraction).
    max_ram_fraction : float
        in [0, 1]. Fraction of RAM to use for reading one batch of data
        when max_ram is None. Note: when using chunks, this should
        be <=~0.25, since lots of ram is needed for in-memory shuffling.
    seed : int or None
        Seed for randomness.

    Returns
    -------
    output_file : str
        Path to the output file.

    """
    if output_file is None:
        output_file = get_filepath_output(input_file, shuffle=True)
    if iterations is None:
        iterations = get_n_iterations(
            input_file,
            datasets=datasets,
            max_ram_fraction=max_ram_fraction,
            max_ram=max_ram,
        )
    # filenames of all iterations, in the right order
    filenames = (
        input_file,
        *_get_temp_filenames(output_file, number=iterations - 1),
        output_file,
    )
    if seed:
        np.random.seed(seed)
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        _shuffle_file(
            input_file=filenames[i],
            output_file=filenames[i + 1],
            delete=i > 0,
            datasets=datasets,
            max_ram=max_ram,
            max_ram_fraction=max_ram_fraction,
        )
    return output_file


def _shuffle_file(
    input_file,
    output_file,
    datasets=("x", "y"),
    max_ram=None,
    max_ram_fraction=0.25,
    delete=False,
):
    start_time = time.time()
    if os.path.exists(output_file):
        raise FileExistsError(output_file)
    if max_ram is None:
        max_ram = get_max_ram(max_ram_fraction)
    # create file with temp name first, then rename afterwards
    temp_output_file = (
        output_file + "_temp_" + time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())
    )
    with h5py.File(input_file, "r") as f_in:
        _check_dsets(f_in, datasets)
        dset_info = _get_largest_dset(f_in, datasets, max_ram)
        print(f"Shuffling datasets {datasets}")
        indices_per_batch = _get_indices_per_batch(
            dset_info["n_batches"],
            dset_info["n_chunks"],
            dset_info["chunksize"],
        )

        with h5py.File(temp_output_file, "x") as f_out:
            for dset_name in datasets:
                print("Creating dataset", dset_name)
                _shuffle_dset(f_out, f_in, dset_name, indices_per_batch)
                print("Done!")

    copy_used_files(input_file, temp_output_file)
    copy_attrs(input_file, temp_output_file)
    os.rename(temp_output_file, output_file)
    if delete:
        os.remove(input_file)
    print(
        f"Elapsed time: " f"{datetime.timedelta(seconds=int(time.time() - start_time))}"
    )
    return output_file


def get_max_ram(max_ram_fraction):
    max_ram = max_ram_fraction * psutil.virtual_memory().available
    print(f"Using {max_ram_fraction:.2%} of available ram = {max_ram} bytes")
    return max_ram


def get_n_iterations(
    input_file, datasets=("x", "y"), max_ram=None, max_ram_fraction=0.25
):
    """ Get how often you have to shuffle with given ram to get proper randomness. """
    if max_ram is None:
        max_ram = get_max_ram(max_ram_fraction=max_ram_fraction)
    with h5py.File(input_file, "r") as f_in:
        dset_info = _get_largest_dset(f_in, datasets, max_ram)
    n_iterations = int(
        np.ceil(np.log(dset_info["n_chunks"]) / np.log(dset_info["chunks_per_batch"]))
    )
    print(f"Largest dataset: {dset_info['name']}")
    print(f"Total chunks: {dset_info['n_chunks']}")
    print(f"Max. chunks per batch: {dset_info['chunks_per_batch']}")
    print(f"--> min iterations for full shuffle: {n_iterations}")
    return n_iterations


def _get_indices_per_batch(n_batches, n_chunks, chunksize):
    """
    Return a list with the shuffled indices for each batch.

    Returns
    -------
    indices_per_batch : List
        Length n_batches, each element is a np.array[int].
        Element i of the list are the indices of each sample in batch number i.

    """
    chunk_indices = np.arange(n_chunks)
    np.random.shuffle(chunk_indices)
    chunk_batches = np.array_split(chunk_indices, n_batches)

    indices_per_batch = []
    for bat in chunk_batches:
        idx = (bat[:, None] * chunksize + np.arange(chunksize)[None, :]).flatten()
        np.random.shuffle(idx)
        indices_per_batch.append(idx)

    return indices_per_batch


def _get_largest_dset(f, datasets, max_ram):
    """
    Get infos about the dset that needs the most batches.
    This is the dset that determines how many samples are shuffled at a time.
    """
    dset_infos = _get_dset_infos(f, datasets, max_ram)
    return dset_infos[np.argmax([v["n_batches"] for v in dset_infos])]


def _check_dsets(f, datasets):
    # check if all datasets have the same number of lines
    n_lines_list = [len(f[dset_name]) for dset_name in datasets]
    if not all([n == n_lines_list[0] for n in n_lines_list]):
        raise ValueError(
            f"Datasets have different lengths! " f"{n_lines_list}"
        )


def _get_dset_infos(f, datasets, max_ram):
    """ Retrieve infos for each dataset. """
    dset_infos = []
    for i, name in enumerate(datasets):
        dset = f[name]
        n_lines = len(dset)
        chunksize = dset.chunks[0]
        n_chunks = int(np.ceil(n_lines / chunksize))
        bytes_per_line = np.asarray(dset[0]).nbytes
        bytes_per_chunk = bytes_per_line * chunksize
        chunks_per_batch = int(np.floor(max_ram / bytes_per_chunk))

        dset_infos.append({
            "name": name,
            "dset": dset,
            "n_chunks": n_chunks,
            "chunks_per_batch": chunks_per_batch,
            "n_batches": int(np.ceil(n_chunks / chunks_per_batch)),
            "chunksize": chunksize,
        })

    return dset_infos


def _shuffle_dset(f_out, f_in, dset_name, indices_per_batch):
    """
    Create a batchwise-shuffled dataset in the output file using given indices.

    """
    dset_in = f_in[dset_name]
    start_idx = 0
    for batch_number, indices in enumerate(indices_per_batch):
        print(f"Processing batch {batch_number+1}/{len(indices_per_batch)}")
        # remove indices outside of dset
        indices = indices[indices < len(dset_in)]
        # reading has to be done with linearly increasing index
        #  fancy indexing is super slow
        #  so sort -> turn to slices -> read -> conc -> undo sorting
        sort_ix = np.argsort(indices)
        unsort_ix = np.argsort(sort_ix)
        fancy_indices = indices[sort_ix]
        slices = _slicify(fancy_indices)
        data = np.concatenate([dset_in[slc] for slc in slices])
        data = data[unsort_ix]

        if batch_number == 0:
            out_dset = f_out.create_dataset(
                dset_name,
                data=data,
                maxshape=dset_in.shape,
                chunks=dset_in.chunks,
                compression=dset_in.compression,
                compression_opts=dset_in.compression_opts,
                shuffle=dset_in.shuffle,
            )
            out_dset.resize(len(dset_in), axis=0)
            start_idx = len(data)
        else:
            end_idx = start_idx + len(data)
            f_out[dset_name][start_idx:end_idx] = data
            start_idx = end_idx

        print("Memory peak: {0:.3f} MB".format(peak_memory_usage()))

    if start_idx != len(dset_in):
        print(f"Warning: last index was {start_idx} not {len(dset_in)}")


def _slicify(fancy_indices):
    """ [0,1,2, 6,7,8] --> [0:3, 6:9] """
    steps = np.diff(fancy_indices) != 1
    slice_starts = np.concatenate([fancy_indices[:1], fancy_indices[1:][steps]])
    slice_ends = np.concatenate([fancy_indices[:-1][steps], fancy_indices[-1:]]) + 1
    return [slice(slice_starts[i], slice_ends[i]) for i in range(len(slice_starts))]


def _get_temp_filenames(output_file, number):
    path, file = os.path.split(output_file)
    return [os.path.join(path, f"temp_iteration_{i}_{file}") for i in range(number)]


def run_parser():
    # TODO deprecated
    raise NotImplementedError(
        "h5shuffle2 has been renamed to orcasong h5shuffle2")
