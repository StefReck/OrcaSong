import warnings
import numpy as np


def get_real_data(infile, only_downgoing_tracks=False):
    """ Get info present in real data. """

    def extr(blob):
        # just take everything from event info
        if not len(blob['EventInfo']) == 1:
            warnings.warn(f"Event info has length {len(blob['EventInfo'])}, not 1")
        track = dict(zip(blob['EventInfo'].dtype.names, blob['EventInfo'][0]))
        track.update(**get_best_track(blob, only_downgoing_tracks=only_downgoing_tracks))

        track["n_hits"] = len(blob["Hits"])
        track["n_triggered_hits"] = blob["Hits"]["triggered"].sum()
        is_triggered = blob["Hits"]["triggered"].astype(bool)
        track["n_triggered_doms"] = len(np.unique(blob["Hits"]["dom_id"][is_triggered]))
        track["t_last_triggered"] = blob["Hits"]["time"][is_triggered].max()

        unique_hits = get_only_first_hit_per_pmt(blob["Hits"])
        track["n_pmts"] = len(unique_hits)
        track["n_triggered_pmts"] = unique_hits["triggered"].sum()

        if "n_hits_intime" in blob["EventInfo"]:
            n_hits_intime = blob["EventInfo"]["n_hits_intime"]
        else:
            n_hits_intime = np.nan
        track["n_hits_intime"] = n_hits_intime
        return track

    return extr


def get_only_first_hit_per_pmt(hits):
    """ Keep only the first hit of each pmt. """
    idents = np.stack((hits["dom_id"], hits["channel_id"]), axis=-1)
    sorted_time_indices = np.argsort(hits["time"])
    # indices of first hit per pmt in time sorted array:
    indices = np.unique(idents[sorted_time_indices], axis=0, return_index=True)[1]
    # indices of first hit per pmt in original array:
    first_hit_indices = np.sort(sorted_time_indices[indices])
    return hits[first_hit_indices]


def get_best_track(blob, missing_value=np.nan, only_downgoing_tracks=False):
    """
    I mean first track, i.e. the one with longest chain and highest lkl/nhits.
    Can also take the best track only of those that are downgoing.
    """
    # hardcode names here since the first blob might not have Tracks
    names = ('E',
             'JCOPY_Z_M',
             'JENERGY_CHI2',
             'JENERGY_ENERGY',
             'JENERGY_MUON_RANGE_METRES',
             'JENERGY_NDF',
             'JENERGY_NOISE_LIKELIHOOD',
             'JENERGY_NUMBER_OF_HITS',
             'JGANDALF_BETA0_RAD',
             'JGANDALF_BETA1_RAD',
             'JGANDALF_CHI2',
             'JGANDALF_LAMBDA',
             'JGANDALF_NUMBER_OF_HITS',
             'JGANDALF_NUMBER_OF_ITERATIONS',
             'JSHOWERFIT_ENERGY',
             'JSTART_LENGTH_METRES',
             'JSTART_NPE_MIP',
             'JSTART_NPE_MIP_TOTAL',
             'JVETO_NPE',
             'JVETO_NUMBER_OF_HITS',
             'dir_x',
             'dir_y',
             'dir_z',
             'id',
             'idx',
             'length',
             'likelihood',
             'pos_x',
             'pos_y',
             'pos_z',
             'rec_type',
             't',
             'group_id')
    index = None
    if "Tracks" in blob:
        if only_downgoing_tracks:
            downs = np.where(blob["Tracks"].dir_z < 0)[0]
            if len(downs) != 0:
                index = downs[0]
        else:
            index = 0

    if index is not None:
        track = blob["Tracks"][index]
        return {f"jg_{name}_reco": track[name] for name in names}
    else:
        return {f"jg_{name}_reco": missing_value for name in names}


class MupageMcInfoExtractor:
    """
    For mupage muon simulations.

    Store some mc infos about the event as a whole (direction, n_muons, ...),
    as well as info about each muon in the sample.

    Attributes
    ----------
    detx_file : str
        Detector file for reading out the positions of DOMs (for calculating
        distance to muon track).
        Required for top_n_muons and min_n_hits.
    inactive_du : int or None
        Dont count mchits in this du.
    min_n_mchits : int
        Dont store info for muons with less than this mchits.
    top_n_muons : int
        Store things like energy, position, ... for the top_n_muons with the
        highest mchits in the active line.
    missing_value : float
        If a value is missing, use this value instead.
    with_primary : bool
        Add info about the primary particle. Corsika only! Also removes
        the primary from mctracks for stuff like summed energy...
    mc_index : int, optional
        Add a column called mc_index containing this number.
    only_downgoing_tracks : bool
        For tracks (JG reco), consider only the ones that are downgoing.

    """
    def __init__(self,
                 infile,
                 inactive_du=1,
                 min_n_mchits=10,
                 min_n_hits=5,
                 min_n_hits_time_window=35,
                 top_n_muons=20,
                 missing_value=np.nan,
                 with_primary=True,
                 only_downgoing_tracks=False,
                 additional_stuff=False):
        mc_index = get_mc_index(infile)
        print(f"Using mc_index {mc_index}")

        self.inactive_du = inactive_du
        self.min_n_mchits = min_n_mchits
        self.min_n_hits = min_n_hits
        self.min_n_hits_time_window = min_n_hits_time_window
        self.top_n_muons = top_n_muons
        self.missing_value = missing_value
        self.with_primary = with_primary
        self.mc_index = mc_index
        self.only_downgoing_tracks = only_downgoing_tracks
        self.additional_stuff = additional_stuff

        self.bundle_settings = {
            "plane_mode": "normal",
            "plane_point_mode": "mid_4line",
        }
        self.real_data_extr = get_real_data(infile, only_downgoing_tracks=only_downgoing_tracks)

    def __call__(self, blob):
        mc_info = self.real_data_extr(blob)

        # all muons in a bundle are parallel, so just take dir of first muon
        mc_info["dir_x"] = blob["McTracks"].dir_x[0]
        mc_info["dir_y"] = blob["McTracks"].dir_y[0]
        mc_info["dir_z"] = blob["McTracks"].dir_z[0]

        primary_track = None
        if self.with_primary:
            # store info about the primary, which is track 0 with id 0
            to_store = ("dir_x", "dir_y", "dir_z", "pos_x", "pos_y", "pos_z",
                        "pdgid", "energy", "time")
            if blob["McTracks"][0]["id"] == 0:
                primary_track = blob["McTracks"][0]
                for fld in to_store:
                    mc_info[f"primary_{fld}"] = primary_track[fld]
                primary_xy = get_primary_onplane(primary_track, self.bundle_settings)
                mc_info["primary_x"] = primary_xy[0]
                mc_info["primary_y"] = primary_xy[1]
                # remove primary from all the other stuff
                blob["McTracks"] = blob["McTracks"][1:]
            else:
                warnings.warn("Error finding primary: mc_tracks[0]['id'] != 0")
                for fld in to_store:
                    mc_info[f"primary_{fld}"] = self.missing_value
                mc_info["primary_x"] = self.missing_value
                mc_info["primary_y"] = self.missing_value

        center_of_mass = get_com(blob, self.bundle_settings)
        mc_info["center_of_mass_x"] = center_of_mass[0]
        mc_info["center_of_mass_y"] = center_of_mass[1]

        mc_info["sim_energy"] = blob["McTracks"]["energy"].sum()
        mc_info["sim_energy_lost_in_can"] = blob["McTracks"]["energy_lost_in_can"].sum()

        # mc_hits in active line
        mchits_per_muon = get_mchits_per_muon(blob, inactive_du=self.inactive_du)
        # actual hits
        hits_per_muon = self.get_hits_per_muon(blob)
        # mctracks, but only muons with >= min_n_mchits
        mc_tracks_sel = blob["McTracks"][mchits_per_muon >= self.min_n_mchits]
        # mctracks, but only muons with >= min_n_hits
        mc_tracks_sel_hit = blob["McTracks"][hits_per_muon >= self.min_n_hits]

        # n_mchits of the visible muons
        mc_info["n_mc_hits"] = np.sum(
            mchits_per_muon[mchits_per_muon >= self.min_n_mchits])  # TODO rename?
        # n_hits of ALL muons
        mc_info["n_signal_hits"] = np.sum(hits_per_muon)
        # n_muons with at least the given hits or mchits
        mc_info[f"n_muons_1_hit"] = (hits_per_muon >= 1).sum()
        n_muons_sel = len(mc_tracks_sel)
        mc_info[f"n_muons_{self.min_n_mchits}_mchits"] = n_muons_sel
        n_muons_sel_hits = len(mc_tracks_sel_hit)
        mc_info[f"n_muons_{self.min_n_hits}_hits"] = n_muons_sel_hits

        # properties of all simulated muons
        add_info_from_mctracks(
            mc_info, blob["McTracks"], self.bundle_settings, missing_value=self.missing_value, suffix="_sim", primary_track=primary_track, center_of_mass=center_of_mass)
        # properties of all VISIBLE muons (mchit criterium)
        add_info_from_mctracks(
            mc_info, mc_tracks_sel, self.bundle_settings, missing_value=self.missing_value, primary_track=primary_track, center_of_mass=center_of_mass)
        # properties of all VISIBLE muons (hit criterium)
        add_info_from_mctracks(
            mc_info, mc_tracks_sel_hit, self.bundle_settings, missing_value=self.missing_value, suffix="_hit", primary_track=primary_track, center_of_mass=center_of_mass)

        if self.additional_stuff:
            # additional stuff, might delete later
            # TODO add center_of_mass, change suffix to _3hit for 2nd one
            add_info_from_mctracks(
                mc_info, blob["McTracks"][hits_per_muon >= 1],
                self.bundle_settings, missing_value=self.missing_value, suffix="_1hit", primary_track=primary_track)
            add_info_from_mctracks(
                mc_info, blob["McTracks"][hits_per_muon >= 3],
                self.bundle_settings, missing_value=self.missing_value, suffix="_3hit", primary_track=primary_track)

        if self.top_n_muons:
            # For highest mc_hits muons (even if they have less than the
            # threshold), store these parameters, muon-by-muon
            for i, muon_info in enumerate(self.muon_info_gen(blob, mchits_per_muon, primary_mctracks=primary_track)):
                if muon_info is not None:
                    mc_info[f"energy_{i}"] = muon_info["mc_tracks"]["energy"]
                    mc_info[f"n_mc_hits_{i}"] = muon_info["n_mc_hits"]
                    mc_info[f"pos_x_{i}"] = muon_info["points"][0]
                    mc_info[f"pos_y_{i}"] = muon_info["points"][1]
                    mc_info[f"min_dist_{i}"] = muon_info["dist"]
                else:
                    mc_info[f"energy_{i}"] = self.missing_value
                    mc_info[f"n_mc_hits_{i}"] = self.missing_value
                    mc_info[f"pos_x_{i}"] = self.missing_value
                    mc_info[f"pos_y_{i}"] = self.missing_value
                    mc_info[f"min_dist_{i}"] = self.missing_value

        if self.mc_index:
            mc_info["mc_index"] = self.mc_index

        return mc_info

    def muon_info_gen(self, blob, mchits_per_muon, primary_mctracks=None):
        """
        Yield muon info of blob, muon-by-muon, give None if there are no
        more muons.
        """
        desc_order = np.argsort(-mchits_per_muon)

        points_on_plane = BundleCollider.from_mctracks(
            blob["McTracks"], primary_mctracks=primary_mctracks, **self.bundle_settings,
        ).get_points_on_plane()

        for i in range(self.top_n_muons):
            if i < len(desc_order):
                muon_index = desc_order[i]
                yield {
                    "mc_tracks": blob["McTracks"][muon_index],
                    "n_mc_hits": mchits_per_muon[muon_index],
                    "points": points_on_plane[muon_index],
                    "dist": get_min_distance_from_doms(
                        blob["McTracks"][muon_index],
                        dom_positions=self.dom_positions,
                    )
                }
            else:
                yield None

    def get_hits_per_muon(self, blob):
        """
        Get the actual number of hits each particle in McTracks produces.
        Done by matching mchits to hits.

        time_window: Timewindow in ns around each hit. A mchit must be
            in this time window on the same pmt for the hit to be counted
            as signal.

        """
        merged = match_mchits(blob["Hits"], blob["McHits"], append=False)
        has_mchit_in_timewindow = (
            (~np.isnan(merged["mchit_dt"])) & (np.abs(merged["mchit_dt"]) <= self.min_n_hits_time_window)
        )
        origin = merged["mchit_origin"]
        hits_per_muon = np.zeros(len(blob["McTracks"]), dtype=int)
        for i, muon_id in enumerate(blob["McTracks"]["id"]):
            hits_per_muon[i] = (
                (origin == muon_id) & has_mchit_in_timewindow
            ).sum()
        return hits_per_muon


class ExtendedBundle(MupageMcInfoExtractor):
    def __init__(self, *args, detx_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.detx_file = detx_file

        if detx_file is not None:
            self.dom_positions = get_dom_positions(
                detx_file, inactive_du=self.inactive_du)
        else:
            self.dom_positions = None

    def __call__(self, blob):
        # type is now called pdgid. Rename pdgid to type for backward compatibility:
        info_dict = super()(blob)
        if "primary_type" in info_dict:
            info_dict["primary_pdgid"] = info_dict.pop("primary_type")
        return info_dict


def get_mchits_per_muon(blob, inactive_du=None):
    """
    For each muon in McTracks, get the number of McHits.

    Parameters
    ----------
    blob
        The blob.
    inactive_du : int, optional
        McHits in this DU will not be counted.

    Returns
    -------
    np.array
        n_mchits, len = number of muons

    """
    ids = blob["McTracks"]["id"]
    # Origin of each mchit (as int) in the active line
    origin = blob["McHits"]["origin"]
    if inactive_du:
        # only hits in active line
        origin = origin[blob["McHits"]["du"] != inactive_du]
    # get how many mchits were produced per muon in the bundle
    origin_dict = dict(zip(*np.unique(origin, return_counts=True)))
    return np.array([origin_dict.get(i, 0) for i in ids])


def get_mc_index(aanet_filename):
    # e.g. mcv5.40.mupage_10G.sirene.jterbr00005782.jorcarec.aanet.365.h5
    return int(aanet_filename.split(".")[-2])
