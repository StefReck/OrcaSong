import os
from abc import abstractmethod
import h5py
import km3pipe as kp
import km3modules as km

import orcasong
import orcasong.modules as modules
import orcasong.plotting.plot_binstats as plot_binstats


__author__ = 'Stefan Reck'


class BaseProcessor:
    """
    Preprocess km3net/antares events for neural networks.

    This serves as a baseclass, which handles things like reading
    events, calibrating, generating labels and saving the output.

    Parameters
    ----------
    mc_info_extr : function, optional
        Function that extracts desired mc_info from a blob, which is then
        stored as the "y" datafield in the .h5 file.
        The function takes the km3pipe blob as an input, and returns
        a dict mapping str to floats.
        Some examples can be found in orcasong.mc_info_extr.
    det_file : str, optional
        Path to a .detx detector geometry file, which can be used to
        calibrate the hits.
    center_time : bool
        Subtract time of first triggered hit from all hit times. Will
        also be done for McHits if they are in the blob [default: True].
    correct_timeslew : bool
        If true, the time slewing of hits depending on their tot
        will be corrected [default: False].
    add_t0 : bool
        If true, add t0 to the time of hits and mchits. If using a
        det_file, this will already have been done automatically.
        Note: Mchits appear to NOT need t0 added, but its done auto-
        matically by km3pipe calibration, so results might be
        wrong for mchits. [default: False].
    event_skipper : func, optional
        Function that takes the blob as an input, and returns a bool.
        If the bool is true, the blob will be skipped.
        This is placed after the binning and mc_info extractor.
    chunksize : int
        Chunksize (along axis_0) used for saving the output
        to a .h5 file [default: 32].
    keep_event_info : bool
        If True, will keep the "event_info" table [default: True].
    keep_mc_tracks : bool
        If True, will keep the "McTracks" table [default: False].

    Attributes
    ----------
    n_statusbar : int, optional
        Print a statusbar every n blobs.
    n_memory_observer : int, optional
        Print memory usage every n blobs.
    complib : str
        Compression library used for saving the output to a .h5 file.
        All PyTables compression filters are available, e.g. 'zlib',
        'lzf', 'blosc', ... .
    complevel : int
        Compression level for the compression filter that is used for
        saving the output to a .h5 file.
    flush_frequency : int
        After how many events the accumulated output should be flushed to
        the harddisk.
        A larger value leads to a faster orcasong execution,
        but it increases the RAM usage as well.

    """
    def __init__(self, mc_info_extr=None,
                 det_file=None,
                 center_time=True,
                 add_t0=False,
                 correct_timeslew=False,
                 event_skipper=None,
                 chunksize=32,
                 keep_event_info=True,
                 keep_mc_tracks=False):
        self.mc_info_extr = mc_info_extr
        self.det_file = det_file
        self.center_time = center_time
        self.add_t0 = add_t0
        self.correct_timeslew = correct_timeslew
        self.event_skipper = event_skipper
        self.chunksize = chunksize
        self.keep_event_info = keep_event_info
        self.keep_mc_tracks = keep_mc_tracks

        self.n_statusbar = 1000
        self.n_memory_observer = 1000
        self.complib = 'zlib'
        self.complevel = 1
        self.flush_frequency = 1000

    def run(self, infile, outfile=None):
        """
        Process the events from the infile, and save them to the outfile.

        Parameters
        ----------
        infile : str
            Path to the input file.
        outfile : str, optional
            Path to the output file (will be created). If none is given,
            will auto generate the name and save it in the cwd.

        """
        if outfile is None:
            outfile_name = "{}_hist.h5".format(
                os.path.splitext(os.path.basename(infile))[0]
            )
            outfile = os.path.join(os.getcwd(), outfile_name)

        pipe = self.build_pipe(infile, outfile)
        pipe.drain()
        add_version_info(outfile)

    def run_multi(self, infiles, outfolder):
        """
        Process multiple files into their own output files each.
        The output file names will be generated automatically.

        Parameters
        ----------
        infiles : List
            The path to infiles as str.
        outfolder : str
            The output folder to place them in.

        """
        outfiles = []
        for infile in infiles:
            outfile = os.path.join(
                outfolder,
                f"{os.path.splitext(os.path.basename(infile))[0]}_hist.h5")
            outfiles.append(outfile)
            self.run(infile, outfile)
        return outfiles

    def build_pipe(self, infile, outfile, timeit=True):
        """ Initialize and connect the modules from the different stages. """
        components = [
            *self.get_cmpts_pre(infile=infile),
            *self.get_cmpts_main(infile=infile, outfile=outfile),
            *self.get_cmpts_post(outfile=outfile),
        ]
        pipe = kp.Pipeline(timeit=timeit)
        if self.n_statusbar is not None:
            pipe.attach(km.common.StatusBar, every=self.n_statusbar)
        if self.n_memory_observer is not None:
            pipe.attach(km.common.MemoryObserver, every=self.n_memory_observer)
        for cmpt, kwargs in components:
            pipe.attach(cmpt, **kwargs)
        return pipe

    def get_cmpts_pre(self, infile):
        """ Modules that read and preproc the events. """
        cmpts = []
        cmpts.append((kp.io.hdf5.HDF5Pump, {"filename": infile}))
        cmpts.append((km.common.Keep, {"keys": [
            'EventInfo', 'Header', 'RawHeader', 'McTracks', 'Hits', 'McHits']}))

        if self.det_file:
            cmpts.append((modules.DetApplier, {"det_file": self.det_file}))

        if self.center_time or self.add_t0:
            cmpts.append((modules.TimePreproc, {
                "add_t0": self.add_t0,
                "center_time": self.center_time,
                "correct_timeslew": self.correct_timeslew}))
        return cmpts

    @abstractmethod
    def get_cmpts_main(self, infile, outfile):
        """  Produce and store the samples as 'samples' in the blob. """
        raise NotImplementedError

    def get_cmpts_post(self, outfile):
        """ Modules that postproc and save the events. """
        cmpts = []
        if self.mc_info_extr is not None:
            cmpts.append((modules.McInfoMaker, {
                "mc_info_extr": self.mc_info_extr,
                "store_as": "mc_info"}))

        if self.event_skipper is not None:
            cmpts.append((modules.EventSkipper, {
                "event_skipper": self.event_skipper}))

        keys_keep = ['samples', 'mc_info']
        if self.keep_event_info:
            keys_keep.append('EventInfo')
        if self.keep_mc_tracks:
            keys_keep.append('McTracks')
        cmpts.append((km.common.Keep, {"keys": keys_keep}))

        cmpts.append((kp.io.HDF5Sink, {
            "filename": outfile,
            "complib": self.complib,
            "complevel": self.complevel,
            "chunksize": self.chunksize,
            "flush_frequency": self.flush_frequency}))
        return cmpts


class FileBinner(BaseProcessor):
    """
    For making binned images and mc_infos, which can be used for conv nets.

    Can also add statistics of the binning to the h5 files, which can
    be plotted to show the distribution of hits among the bins and how
    many hits were cut off.

    Parameters
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin
        edges, including the left- and right-most bin edge.
        Example: For 10 bins in the z direction, and 100 bins in time:
            bin_edges_list = [
                ["pos_z", np.linspace(0, 10, 11)],
                ["time", np.linspace(-50, 550, 101)],
            ]
        Some examples can be found in orcasong.bin_edges.
    add_bin_stats : bool
        Add statistics of the binning to the output file. They can be
        plotted with util/bin_stats_plot.py [default: True].
    hit_weights : str, optional
        Use blob["Hits"][hit_weights] as weights for samples in histogram.

    """
    def __init__(self, bin_edges_list,
                 add_bin_stats=True,
                 hit_weights=None,
                 **kwargs):
        self.bin_edges_list = bin_edges_list
        self.add_bin_stats = add_bin_stats
        self.hit_weights = hit_weights
        super().__init__(**kwargs)

    def get_cmpts_main(self, infile, outfile):
        """ Generate nD images. """
        cmpts = []
        if self.add_bin_stats:
            cmpts.append((modules.BinningStatsMaker, {
                "outfile": outfile,
                "bin_edges_list": self.bin_edges_list}))
        cmpts.append((modules.ImageMaker, {
            "bin_edges_list": self.bin_edges_list,
            "hit_weights": self.hit_weights}))
        return cmpts

    def run_multi(self, infiles, outfolder, save_plot=False):
        """
        Bin multiple files into their own output files each.
        The output file names will be generated automatically.

        Parameters
        ----------
        infiles : List
            The path to infiles as str.
        outfolder : str
            The output folder to place them in.
        save_plot : bool
            Save the binning hists as a pdf. Only possible if add_bin_stats
            is True.

        """
        if save_plot and not self.add_bin_stats:
            raise ValueError("Can not make plot when add_bin_stats is False")

        name, shape = self.get_names_and_shape()
        print("Generating {} images with shape {}".format(name, shape))

        outfiles = super().run_multi(infiles=infiles, outfolder=outfolder)

        if save_plot:
            plot_binstats.plot_hist_of_files(
                files=outfiles, save_as=outfolder+"binning_hist.pdf")

    def get_names_and_shape(self):
        """
        Get names and shape of the resulting x data,
        e.g. (pos_z, time), (18, 50).
        """
        names, shape = [], []
        for bin_name, bin_edges in self.bin_edges_list:
            names.append(bin_name)
            shape.append(len(bin_edges) - 1)
        return tuple(names), tuple(shape)

    def __repr__(self):
        return "<FileBinner: {} {}>".format(*self.get_names_and_shape())


class FileGraph(BaseProcessor):
    """
    Turn km3 events to graph data.

    Parameters
    ----------
    max_n_hits : int
        Maximum number of hits that gets saved per event. If an event has
        more, some will get cut!
    time_window : tuple, optional
        Two ints (start, end). Hits outside of this time window will be cut
        away (base on 'Hits/time').
        Default: Keep all hits.
    hit_infos : tuple, optional
        Which entries in the '/Hits' Table will be kept. E.g. pos_x, time, ...
        Default: Keep all entries.

    """
    def __init__(self, max_n_hits,
                 time_window=None,
                 hit_infos=None,
                 **kwargs):
        self.max_n_hits = max_n_hits
        self.time_window = time_window
        self.hit_infos = hit_infos
        super().__init__(**kwargs)

    def get_cmpts_main(self, infile, outfile):
        return [((modules.PointMaker, {
            "max_n_hits": self.max_n_hits,
            "time_window": self.time_window,
            "hit_infos": self.hit_infos,
            "dset_n_hits": "EventInfo"}))]


def add_version_info(file):
    """ Add current orcasong version to h5 file. """
    with h5py.File(file, "a") as f:
        f.attrs.create("orcasong", orcasong.__version__, dtype="S6")
