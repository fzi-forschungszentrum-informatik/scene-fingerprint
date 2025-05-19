import unittest
import numpy as np
from oncrit.oncrit import ObjectState, Scene, MetricBase


class TestMetricBase(unittest.TestCase):

    def assert_sorted_lists_almost_equal(self, actual_list, expected_list, places=2):
        sorted_actual = sorted(actual_list, key=lambda x: (x[0], x[1]))
        sorted_expected = sorted(expected_list, key=lambda x: (x[0], x[1]))
        self.assertEqual(len(sorted_actual), len(sorted_expected))
        for (exp_a, exp_b, exp_d), (act_a, act_b, act_d) in zip(sorted_expected, sorted_actual):
            self.assertEqual((exp_a, exp_b), (act_a, act_b))
            self.assertAlmostEqual(
                exp_d, act_d, places=places, msg=f"({exp_a}, {exp_b}): {exp_d} != {act_d} (actual)")

    def setUp(self):
        self.future_paths = []
        self.obj1 = ObjectState(
            id="obj1",
            x=0.0,
            y=0.0,
            vx=5.0,
            vy=0.0,
            psi_rad=0.0,
            width=1,
            length=2,
        )
        obj1_future_paths = [[[0, 5, 10, 15],
                              [0, 0, 0, 0]]]
        self.future_paths.append(("obj1", obj1_future_paths))

        self.obj2 = ObjectState(
            id="obj2",
            x=0.0,
            y=10.0,
            vx=6.0,
            vy=0.0,
            psi_rad=0.0,
            width=1,
            length=2,
        )
        obj2_future_paths = [[[0, 5, 15],
                              [10, 10, 10]]]
        self.future_paths.append(("obj2", obj2_future_paths))

        self.obj3 = ObjectState(
            id="obj3",
            x=10.0,
            y=10.0,
            vx=0.0,
            vy=-1.0,
            psi_rad=-1.57079632679,
            width=1,
            length=2,
        )
        obj3_future_paths = [[[10, 10, 10, 10],
                              [10, 5, 0, -5]]]
        self.future_paths.append(("obj3", obj3_future_paths))

        self.obj4 = ObjectState(
            id="obj4",
            x=15.0,
            y=0.0,
            vx=3.0,
            vy=0.0,
            psi_rad=0.0,
            width=1,
            length=2,
        )
        obj4_future_paths = [[[15, 17, 20],
                              [0, 0, 0]]]
        self.future_paths.append(("obj4", obj4_future_paths))

        self.obj5 = ObjectState(
            id="obj5",
            x=4.0,
            y=4.0,
            vx=-0.70710678118,
            vy=-0.70710678118,
            psi_rad=2.35619449019,
            width=1,
            length=2,
        )
        obj5_future_paths = [[[4, 0, -4],
                              [4, 0, -4]]]
        self.future_paths.append(("obj5", obj5_future_paths))

        self.obj6 = ObjectState(
            id="obj6",
            x=7.0,
            y=0.0,
            vx=0,
            vy=0.5,
            psi_rad=1.57079632679,
            width=1,
            length=2,
        )
        obj6_future_paths = [[[7, 7, 7],
                              [0, 0.5, 1]]]
        self.future_paths.append(("obj6", obj6_future_paths))

        self.scene = Scene(
            [self.obj1, self.obj2, self.obj3, self.obj4, self.obj5, self.obj6])

    def test_metric_base_creation(self):
        MetricBase(self.scene)

    def test_filter_relevant_objects_with_ids(self):
        metric_base = MetricBase(self.scene)
        metric_base.filter_relevant_objects(ids=["obj1", "obj2"])
        relevant_objects = metric_base.relevant_objects
        self.assertEqual(len(relevant_objects), 2)
        self.assertTrue(any(obj.id == "obj1" for obj in relevant_objects))
        self.assertTrue(any(obj.id == "obj2" for obj in relevant_objects))
        self.assertFalse(any(obj.id == "obj3" for obj in relevant_objects))

    # Metrics

    def test_euclidean_distance_simple(self):
        metric_base = MetricBase(self.scene)
        actual_list = metric_base.euclidean_distance_simple()

        expected_list = [
            ("obj2", "obj1", 10.0),
            ("obj1", "obj2", 10.0),
            ("obj3", "obj1", 14.142),
            ("obj1", "obj3", 14.142),
            ("obj2", "obj3", 10.0),
            ("obj3", "obj2", 10.0),
            ("obj1", "obj4", 15.0),
            ("obj4", "obj1", 15.0),
            ("obj2", "obj4", 18.028),
            ("obj4", "obj2", 18.028),
            ("obj3", "obj4", 11.18),
            ("obj4", "obj3", 11.18),
            ("obj1", "obj5", 5.66),
            ("obj5", "obj1", 5.66),
            ("obj2", "obj5", 7.211),
            ("obj5", "obj2", 7.211),
            ("obj3", "obj5", 8.485),
            ("obj5", "obj3", 8.485),
            ("obj4", "obj5", 11.7),
            ("obj5", "obj4", 11.7),
            ("obj1", "obj6", 7.0),
            ("obj6", "obj1", 7.0),
            ("obj2", "obj6", 12.207),
            ("obj6", "obj2", 12.207),
            ("obj3", "obj6", 10.44),
            ("obj6", "obj3", 10.44),
            ("obj4", "obj6", 8.0),
            ("obj6", "obj4", 8.0),
            ("obj5", "obj6", 5.0),
            ("obj6", "obj5", 5.0),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)

    def test_euclidean_distance_approximated(self):
        metric_base = MetricBase(self.scene)
        actual_list = metric_base.euclidean_distance_approximated()

        expected_list = [
            ("obj2", "obj1", 9.0),
            ("obj1", "obj2", 9.0),
            ("obj3", "obj1", 12.435028842546133),
            ("obj1", "obj3", 12.435028842546133),
            ("obj2", "obj3", 8.513),
            ("obj3", "obj2", 8.513),
            ("obj1", "obj4", 13.0),
            ("obj4", "obj1", 13.0),
            ("obj2", "obj4", 16.205),
            ("obj4", "obj2", 16.205),
            ("obj3", "obj4", 9.512),
            ("obj4", "obj3", 9.512),
            ("obj1", "obj5", 4.305),
            ("obj5", "obj1", 4.305),
            ("obj2", "obj5", 5.463),
            ("obj5", "obj2", 5.463),
            ("obj3", "obj5", 7.133),
            ("obj5", "obj3", 7.133),
            ("obj4", "obj5", 9.782),
            ("obj5", "obj4", 9.782),
            ("obj1", "obj6", 5.519),
            ("obj6", "obj1", 5.519),
            ("obj2", "obj6", 10.51),
            ("obj6", "obj2", 10.51),
            ("obj3", "obj6", 8.487),
            ("obj6", "obj3", 8.487),
            ("obj4", "obj6", 6.517),
            ("obj6", "obj4", 6.517),
            ("obj5", "obj6", 3.111),
            ("obj6", "obj5", 3.111),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)

    def test_spaceing(self):
        metric_base = MetricBase(self.scene)
        metric_base.set_future_paths(self.future_paths)
        actual_list = metric_base.spacing(threshold=2.0)

        expected_list = [
            ("obj2", "obj3", 10.0),
            ("obj1", "obj4", 15.0),
            ("obj5", "obj1", np.sqrt(32.0)),
            ("obj1", "obj6", 7.0),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)

    def test_spacing_with_approximation(self):
        appr_future_paths = []
        objX = ObjectState(
            id="objX",
            x=1014.267,
            y=994.852,
            vx=4.014,
            vy=1.194,
            psi_rad=0.289,
            width=2.6,
            length=9.86,
        )
        objX_future_paths = [[[0, 1],
                             [0, 1]]]
        appr_future_paths.append(("objX", objX_future_paths))

        objY = ObjectState(
            id="objY",
            x=1004.615,
            y=992.207,
            vx=4.389,
            vy=0.442,
            psi_rad=0.1,
            width=2.6,
            length=7.76,
        )
        objY_future_paths = [[[1004.615, 1006.6100737837503, 1008.6051475675005, 1010.6002213512508, 1012.5952951350009, 1014.5903689187512, 1016.5854427025015, 1018.5805164862517, 1020.575590270002, 1022.5706640537522, 1024.5657378375024, 1026.5608116212527, 1028.555885405003, 1030.5509591887533, 1032.5460329725033, 1034.5411067562536, 1036.536180540004, 1038.5312543237542, 1040.5263281075045, 1042.5214018912545, 1044.5164756750048, 1046.511549458755, 1048.5066232425054],
                              [992.207, 992.4071750738789, 992.6073501477578, 992.8075252216365, 993.0077002955154, 993.2078753693943, 993.4080504432732, 993.608225517152, 993.8084005910308, 994.0085756649097, 994.2087507387886, 994.4089258126675, 994.6091008865463, 994.8092759604252, 995.009451034304, 995.2096261081829, 995.4098011820618, 995.6099762559406, 995.8101513298195, 996.0103264036983, 996.2105014775772, 996.410676551456, 996.6108516253349]]]
        appr_future_paths.append(("objY", objY_future_paths))
        appr_scene = Scene(
            [objX, objY])
        metric_base = MetricBase(appr_scene)
        metric_base.set_future_paths(appr_future_paths)
        expected_list_spacing = [
            ("objY", "objX", 10.007),
        ]
        expected_list_ttc = [
            ("objY", "objX", 4.02),
        ]
        expected_list_gaptime = [
            ("objY", "objX", 0.2715),
        ]
        self.assert_sorted_lists_almost_equal(
            metric_base.spacing(threshold=2.5), expected_list_spacing)
        self.assert_sorted_lists_almost_equal(
            metric_base.ttc(threshold=2.5), expected_list_ttc)
        self.assert_sorted_lists_almost_equal(
            metric_base.gaptime(threshold=2.5), expected_list_gaptime)

    def test_clearance(self):
        metric_base = MetricBase(self.scene)
        metric_base.set_future_paths(self.future_paths)
        actual_list = metric_base.clearance(threshold=2.0)

        expected_list = [
            ("obj2", "obj3", 8.5),
            ("obj1", "obj4", 13.0),
            ("obj5", "obj1", np.sqrt(32.0)-1.0-np.sqrt(.5**2+1.0)),
            ("obj1", "obj6", 5.5),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)

    def test_headway(self):
        metric_base = MetricBase(self.scene)
        metric_base.set_future_paths(self.future_paths)
        actual_list = metric_base.headway(threshold=2.0)

        expected_list = [
            ("obj2", "obj3", 10.0/6.0),
            ("obj1", "obj4", 15.0/5.0),
            ("obj5", "obj1", np.sqrt(32.0)/1.0),
            ("obj1", "obj6", 7.0/5.0),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)

    def test_gaptime(self):

        metric_base = MetricBase(self.scene)
        metric_base.set_future_paths(self.future_paths)
        actual_list = metric_base.gaptime(threshold=2.0)

        expected_list = [
            ("obj2", "obj3", 8.5/6.0),
            ("obj1", "obj4", 13.0/5.0),
            ("obj5", "obj1", np.sqrt(32.0)-1.0-np.sqrt(.5**2+1.0)/1.0),
            ("obj1", "obj6", 5.5/5.0),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)

    def test_ttc(self):
        metric_base = MetricBase(self.scene)
        metric_base.set_future_paths(self.future_paths)
        actual_list = metric_base.ttc(threshold=2.0)

        expected_list = [
            ("obj2", "obj3", 1.417),
            ("obj1", "obj4", 13.0/(5.0-3.0)),
            ("obj5", "obj1", (np.sqrt(32.0)-1.0-np.sqrt(.5**2+1.0)) /
             (1.0-(np.cos(3*np.pi/4)*5.0))),
            ("obj1", "obj6", 5.5/5.0),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)

    def test_predictive_encroachment_time(self):
        metric_base = MetricBase(self.scene)
        metric_base.set_future_paths(self.future_paths)
        actual_list = metric_base.predictive_encroachment_time(threshold=2.0)
        expected_list = [
            ("obj3", "obj1", 10.0 / 5.0 - 10.0 / 1.0),
            ("obj1", "obj3", 10.0 / 1.0 - 10.0 / 5.0),
        ]
        self.assert_sorted_lists_almost_equal(actual_list, expected_list)


if __name__ == "__main__":
    unittest.main()
