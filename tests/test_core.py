import os
from unittest import TestCase
import tempfile
import numpy as np
import h5py
import orcasong.core
import orcasong.mc_info_extr
from orcasong.plotting.plot_binstats import read_hists_from_h5file


__author__ = 'Stefan Reck'


test_dir = os.path.dirname(os.path.realpath(__file__))
MUPAGE_FILE = os.path.join(test_dir, "data", "mupage.root.h5")
DET_FILE = os.path.join(test_dir, "data", "KM3NeT_-00000001_20171212.detx")
NEUTRINO_FILE = os.path.join(test_dir, "data", "neutrino_file.h5")
DET_FILE_NEUTRINO = os.path.join(test_dir, "data", "neutrino_detector_file.detx")

class TestFileBinner(TestCase):
    """ Assert that the filebinner still produces the same output. """
    @classmethod
    def setUpClass(cls):
        cls.proc = orcasong.core.FileBinner(
            bin_edges_list=[
                ["pos_z", np.linspace(0, 200, 3)],
                ["time", np.linspace(0, 600, 3)],
                ["channel_id", np.linspace(-0.5, 30.5, 3)],
            ],
            mc_info_extr=orcasong.mc_info_extr.get_real_data_info_extr(MUPAGE_FILE),
            det_file=DET_FILE,
            add_t0=True,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outfile = os.path.join(cls.tmpdir.name, "binned.h5")
        cls.proc.run(infile=MUPAGE_FILE, outfile=cls.outfile)
        cls.f = h5py.File(cls.outfile, "r")

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.tmpdir.cleanup()

    def test_keys(self):
        self.assertSetEqual(set(self.f.keys()), {
            '_i_event_info', '_i_group_info', '_i_y', 'bin_stats',
            'event_info', 'group_info', 'x', 'x_indices', 'y'})

    def test_x(self):
        target = np.array([
            [[[4, 1], [6, 3]],
                [[11, 5], [7, 7]]],
            [[[4, 2], [1, 3]],
                [[5, 7], [8, 5]]],
            [[[3, 3], [2, 4]],
                [[5, 6], [6, 8]]]
        ], dtype=np.uint8)
        np.testing.assert_equal(target, self.f["x"])

    def test_y(self):
        y = self.f["y"][()]
        target = {
            'event_id': np.array([0., 1., 2.]),
            'run_id': np.array([1., 1., 1.]),
            'trigger_mask': np.array([18., 18., 16.]),
            'group_id': np.array([0, 1, 2]),
        }
        for k, v in target.items():
            np.testing.assert_equal(y[k], v)

    def test_bin_stats(self):
        bin_stats = read_hists_from_h5file(self.f)
        target_hists = {
            "channel_id": np.array([20.0, 8.0, 10.0, 8.0, 16.0, 13.0, 11.0, 9.0, 10.0, 11.0]),
            "pos_z": np.array([0.0, 0.0, 0.0, 36.0, 0.0, 19.0, 30.0, 0.0, 31.0, 0.0]),
            "time": np.array([56.0, 60.0]),
        }
        for dim, infos in bin_stats.items():
            np.testing.assert_equal(infos["hist"], target_hists[dim])


class TestFileGraph(TestCase):
    """ Assert that the FileGraph still produces the same output. """
    @classmethod
    def setUpClass(cls):
        cls.proc = orcasong.core.FileGraph(
            max_n_hits=3,
            time_window=[0, 50],
            hit_infos=["pos_z", "time", "channel_id"],
            mc_info_extr=orcasong.mc_info_extr.get_real_data_info_extr(MUPAGE_FILE),
            det_file=DET_FILE,
            add_t0=True,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outfile = os.path.join(cls.tmpdir.name, "binned.h5")
        cls.proc.run(infile=MUPAGE_FILE, outfile=cls.outfile)
        cls.f = h5py.File(cls.outfile, "r")

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.tmpdir.cleanup()

    def test_keys(self):
        self.assertSetEqual(set(self.f.keys()), {
            '_i_event_info', '_i_group_info', '_i_y',
            'event_info', 'group_info', 'x', 'x_indices', 'y'})

    def test_x_attrs(self):
        to_check = {
            "hit_info_0": "pos_z",
            "hit_info_1": "time",
            "hit_info_2": "channel_id",
            "hit_info_3": "is_valid",
        }
        attrs = dict(self.f["x"].attrs)
        for k, v in to_check.items():
            self.assertTrue(attrs[k] == v)

    def test_x(self):
        target = np.array([
            [[676.941,  13.,  30.,   1.],
             [461.111,  32.,   9.,   1.],
             [424.941,   1.,  30.,   1.]],
            [[172.83,  32.,  25.,   1.],
             [316.83,   2.,  14.,   1.],
             [461.059,   1.,   3.,   1.]],
            [[496.83,  34.,  25.,   1.],
             [605.111,   9.,   4.,   1.],
             [424.889,  46.,  29.,   1.]]
        ], dtype=np.float32)
        np.testing.assert_equal(target, self.f["x"])

    def test_y(self):
        y = self.f["y"][()]
        target = {
            'event_id': np.array([0., 1., 2.]),
            'run_id': np.array([1., 1., 1.]),
            'trigger_mask': np.array([18., 18., 16.]),
            'group_id': np.array([0, 1, 2]),
        }
        for k, v in target.items():
            np.testing.assert_equal(y[k], v)

class TestStdRecoExtractor(TestCase):
    """ Assert that the neutrino info is extracted correctly File has 18 events. """
    @classmethod
    def setUpClass(cls):
        cls.proc = orcasong.core.FileGraph(
            max_n_hits=3,
            time_window=[0, 50],
            hit_infos=["pos_z", "time", "channel_id"],
            mc_info_extr=orcasong.mc_info_extr.get_neutrino_mc_info_extr(NEUTRINO_FILE),
            det_file=DET_FILE_NEUTRINO,
            add_t0=True,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outfile = os.path.join(cls.tmpdir.name, "binned.h5")
        cls.proc.run(infile=NEUTRINO_FILE, outfile=cls.outfile)
        cls.f = h5py.File(cls.outfile, "r")

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.tmpdir.cleanup()

    def test_keys(self):
        self.assertSetEqual(set(self.f.keys()), {
            '_i_event_info', '_i_group_info', '_i_y',
            'event_info', 'group_info', 'x', 'x_indices', 'y'})

    def test_y(self):
        y = self.f["y"][()]
        target = {
            'weight_w2': np.array([29650.0,
									297100.0,
									41450.0,
									371400.0,
									1101000000.0,
									2757000.0,
									15280000.0,
									262800000.0,
									22590.0,
									24240.0,
									80030.0,
									3018000.0,
									120600.0,
									872200.0,
									50440000.0,
									21540.0,
									42170.0,
									25230.0]),
									
            'n_gen': np.array([60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0,
								60000.0]),
								
            'dir_z': np.array([-0.896549,
								-0.835252,
								0.300461,
								0.108997,
								0.128445,
								-0.543621,
								-0.23205,
								-0.297228,
								0.694932,
								0.73835,
								-0.007682,
								0.437847,
								-0.126804,
								0.153432,
								-0.263229,
								0.820217,
								0.452473,
								0.294217]),
            
            'is_cc': np.array([2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0,
								2.0]),
								
            'std_dir_z': np.array([-0.923199825369434,
									-0.6422689266782661,
									0.38853917922036363,
									-0.16690804339142448,
									-0.01584853496341109,
									-0.10151549881670698,
									-0.0409694104272829,
									-0.32964369874021787,
									-0.3294926806601529,
									0.6524241250799204,
									-0.3899574246450216,
									0.27872277417339086,
									0.0019490791409933206,
									0.20341370281708737,
									-0.15739475718286297,
									0.8040250543935723,
									0.08772622550043882,
									-0.7766722433951796]),
									
            'std_energy': np.array([4.7187625606210775,
									4.169818842606011,
									1.0056373761749966,
									5.908597073055873,
									12.409377607517195,
									7.566695371401163,
									1.3546775620239864,
									2.659528737837978,
									1.0056373761749966,
									2.1968321463948755,
									1.4821714294894754,
									10.135831333340658,
									2.6003934443336765,
									1.4492149732348223,
									71.69167874147956,
									8.094744120333358,
									3.148088080484504,
									1.0056373761749966]),
            
        }
        for k, v in target.items():
            np.testing.assert_equal(y[k], v)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            