import unittest
import os
import numpy as np
import sys
import ray

if sys.version_info[1] < 8:
    raise Exception("At least Python 3.8 is recommend!")

from scenario_criticality.binary_metrics.gap_time.gap_time import GapTime
from scenario_criticality.binary_metrics.time_to_collision.ttc import TTC
from scenario_criticality.binary_metrics.post_encroachment_time.pet import PET
from scenario_criticality.binary_metrics.post_encroachment_time.et import ET
from scenario_criticality.binary_metrics.potential_time_to_collision.pttc import PTTC
from scenario_criticality.binary_metrics.trajectory_distance.trajectory_distance import TrajectoryDistance
from scenario_criticality.safety_potential.safety_potential import SafetyPotential
from csv_object_list_dataset_loader.loader import Loader

try:
    from utils.lanelet_tools import read_to_lanelet_map
    lanelet2_import = True
except ImportError:
    print("Lanelet2 could not be imported - Testing without lanelet")
    lanelet2_import = False


class TestImports(unittest.TestCase):

    def setUp(self):
        pass

    def test_csv_loader(self):
        loader_class = Loader()
        self.assertIsInstance(loader_class, Loader)

    def tearDown(self):
        pass


class TestSafetyPotential(unittest.TestCase):

    def setUp(self):
        self.dataset_loader = Loader()
        self.dataset_loader.load_dataset("tests/resources/test_tracks_001.csv")
        self.scenario = self.dataset_loader.return_scenario("tests/resources/test_tracks_001.csv")
        maps_path = "tests/resources/test_map_001.osm"
        if lanelet2_import:
            self.lanelet_map = read_to_lanelet_map(maps_path, origin=(49.005306, 8.4374089))

    def test_safety_potential(self):
        validation_matrix = np.array([[0, 0, 0.94682845],
                                      [0, 0, 0],
                                      [0.70990054, 0, 0]])
        safety_potential_class = SafetyPotential(self.scenario,
                                                 5900,
                                                 procedure_type=None,
                                                 lanelet_map=None,
                                                 normalized=True)
        safety_potential_class.calculate_metric()
        self.assertListEqual(np.round(safety_potential_class.results_matrix,
                             4).tolist(), np.round(validation_matrix, 4).tolist())

    if lanelet2_import:
        def test_safety_potential_lanelet(self):
            validation_matrix = np.array([[0., 0.72889542, 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0.37088598, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            safety_potential_class = SafetyPotential(self.scenario,
                                                     36400,
                                                     procedure_type="along_lane",
                                                     lanelet_map=self.lanelet_map,
                                                     normalized=True)
            safety_potential_class.calculate_metric()
            self.assertListEqual(np.round(safety_potential_class.results_matrix,
                                          4).tolist(), np.round(validation_matrix, 4).tolist())

        def tearDown(self):
            os.remove("tests/resources/test_tracks_001.pdata")


class TestCarFollowingScenario(unittest.TestCase):

    def setUp(self):
        self.dataset_loader = Loader()
        self.dataset_loader.load_dataset("./tests/resources/CarFollowing_001.csv")
        self.scenario = self.dataset_loader.return_scenario("./tests/resources/CarFollowing_001.csv")  # noqa

    def test_multiple_timestamps(self):
        validation_dict = {109079: {'ttc': -2.0, 'pttc': -1.0, 'pet': -2.0}, 109891:
            {'ttc': 39.21, 'pttc': 37.12, 'pet': -2.0}, 110370: {'ttc': 17.88, 'pttc': 17.66, 'pet': -2.0}, 110661:
            {'ttc': 12.69, 'pttc': 12.62, 'pet': -2.0}, 111984: {'ttc': 6.83, 'pttc': 6.84, 'pet': -2.0}}

        result_dict = {}
        timestamps = [109079, 109891, 110370, 110661, 111984]
        for ts in timestamps:
            result_dict[ts] = {}
            ttc = TTC(self.scenario, ts)
            ttc.calculate_metric()
            result_dict[ts]["ttc"] = np.max(ttc.results_matrix)
            pttc = PTTC(self.scenario, ts)
            pttc.calculate_metric()
            result_dict[ts]["pttc"] = np.max(pttc.results_matrix)
            # for verification pet should not be computed
            pet = PET(self.scenario, ts)
            pet.calculate_metric()
            result_dict[ts]["pet"] = np.max(pet.results_matrix)
        self.assertDictEqual(result_dict, validation_dict)

    def tearDown(self):
        os.remove("tests/resources/CarFollowing_001.pdata")


class TestIntersection(unittest.TestCase):

    def setUp(self):
        self.dataset_loader = Loader()
        self.dataset_loader.load_dataset("./tests/resources/Intersection_001.csv")
        self.scenario = self.dataset_loader.return_scenario("./tests/resources/Intersection_001.csv")  # noqa

    def test_multiple_timestamps(self):
        validation_dict = {773152: {'pet': 1.704, 'et': 0.751, 'gt': -5.0, 'td': 82.74313140982625, 'ttc': -3.0},
                           777290: {'pet': 1.704, 'et': 0.751, 'gt': 0.5109629908474131, 'td': 67.33412747606418,
                                    'ttc': -3.0},
                           780448: {'pet': 1.704, 'et': 0.751, 'gt': 0.5416890038721549, 'td': 31.609417666408913,
                                    'ttc': -3.0},
                           785720: {'pet': 1.704, 'et': 0.751, 'gt': -6.0, 'td': -6.0, 'ttc': -3.0},
                           788803: {'pet': 1.704, 'et': 0.751, 'gt': -6.0, 'td': -6.0, 'ttc': -3.0}}
        result_dict = {}
        timestamps = [773152, 777290, 780448, 785720, 788803]
        for ts in timestamps:
            result_dict[ts] = {}
            pet = PET(self.scenario, ts)
            pet.calculate_metric()
            result_dict[ts]["pet"] = np.max(pet.results_matrix)
            et = ET(self.scenario, ts)
            et.calculate_metric()
            result_dict[ts]["et"] = np.max(et.results_matrix)
            gt = GapTime(self.scenario, ts)
            gt.calculate_metric()
            result_dict[ts]["gt"] = np.max(gt.results_matrix)
            td = TrajectoryDistance(self.scenario, ts)
            td.calculate_metric()
            result_dict[ts]["td"] = np.max(td.results_matrix)
            # for verification ttc should not be computed
            ttc = TTC(self.scenario, ts)
            ttc.calculate_metric()
            result_dict[ts]["ttc"] = np.max(ttc.results_matrix)
        self.assertDictEqual(result_dict, validation_dict)

    def tearDown(self):
        os.remove("tests/resources/Intersection_001.pdata")


@ray.remote
def iterate_parallel(scenario, timestamp):
    result_dict = {}
    safety_potential_class = SafetyPotential(scenario,
                                             timestamp,
                                             procedure_type=None,
                                             lanelet_map=None,
                                             normalized=True)
    safety_potential_class.calculate_metric()
    result_dict["sp"] = np.max(safety_potential_class.results_matrix)

    ttc = TTC(scenario, timestamp)
    ttc.calculate_metric()
    result_dict["ttc"] = np.max(ttc.results_matrix)

    pet = PET(scenario, timestamp)
    pet.calculate_metric()
    result_dict["pet"] = np.max(pet.results_matrix)

    tj = TrajectoryDistance(scenario, timestamp, intersection_times=pet.intersection_times)
    tj.calculate_metric()
    result_dict["tj"] = np.max(tj.results_matrix)

    return result_dict


class TestParallelComputing(unittest.TestCase):

    def setUp(self):
        ray.init()
        self.dataset_loader = Loader()
        self.dataset_loader.load_dataset("./tests/resources/test_tracks_001.csv")
        self.scenario = self.dataset_loader.return_scenario("./tests/resources/test_tracks_001.csv")  # noqa

    def test_multiple_timestamps(self):
        timestamps = [0, 100, 1100, 1200, 3500]
        results = []
        for timestamp in timestamps:
            results.append(iterate_parallel.remote(self.scenario, timestamp))
        output = ray.get(results)
        self.assertIsInstance(output[0], dict)
        self.assertEqual(len(output), 5)

    def tearDown(self):
        os.remove("tests/resources/test_tracks_001.pdata")
        ray.shutdown()


if __name__ == '__main__':
    unittest.main()
