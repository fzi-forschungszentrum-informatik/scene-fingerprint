import unittest
import numpy as np
import pandas as pd
import math
from oncrit.utils.lanelet_operation import get_possible_paths
from oncrit.utils.lanelet_operation import read_to_lanelet_map, get_possible_paths, get_best_fitting_lanelet
from oncrit.utils.helpers import generate_metric_base
from oncrit.oncrit import Point, LineSegment, LineString, LineStringFan
from oncrit.oncrit import ObjectState, Scene, MetricBase


EPSILON = 1e-9

csv_path = "tests/data/tracks_cam1_taf_cut.csv"
osm_path = "tests/data/k729_2022-03-16_fix.osm"
osm_origin = (49.01160993928274, 8.43856470258739)


class TestScenarioLoading(unittest.TestCase):
    def assertFloatListsAlmostEqual(self, list1, list2, places=3, msg=None):
        self.assertEqual(len(list1), len(list2), msg or "Length mismatch")
        for a, b in zip(list1, list2):
            self.assertTrue(
                math.isclose(a, b, rel_tol=10**-places, abs_tol=10**-places),
                msg or f"Floats not almost equal: {a} != {b}"
            )

    def setUp(self):
        self.llet_map, self.routing_graph = read_to_lanelet_map(
            osm_path, origin=osm_origin)
        pd_dataframe = pd.read_csv(csv_path)
        self.scene = pd_dataframe[pd_dataframe["frame_id"] == 750]

    def test_possible_path_real_data(self):
        objects_future_paths = []
        for i, row in self.scene.iterrows():
            match = get_best_fitting_lanelet(
                self.llet_map, row.x, row.y, row.psi_rad)
            if match:
                future_paths = get_possible_paths(
                    match,
                    self.routing_graph,
                    max_routing_cost=70,
                    distance_threshold=5,
                    smooth_value=2)
                objects_future_paths.append((str(row.track_id), future_paths))

        expected_list = {"124975": [([16.982, 15.812, 13.622, 11.346, 8.143, 5.021, 1.918, -1.497, -2.983, -4.767, -8.988, -12.665, -15.635, -18.445, -19.524],
                                     [-20.084, -18.3, -14.8, -10.814, -5.057, 0.372, 6.364, 13.751, 17.089, 21.185, 31.014, 39.836, 47.596, 55.538, 58.809])],
                         "145725": [([22.97, 25.373, 31.12, 36.972, 43.221, 49.938, 57.003, 63.96, 66.829],
                                     [-3.675, -2.727, -0.227, 2.456, 5.285, 8.49, 12.386, 16.503, 18.23])],
                         "179842": [([20.477, 19.326, 17.005, 14.202, 10.622, 7.586, 4.57, 1.558, 0.475, -1.309, -6.249, -10.645, -12.964, -15.112, -16.181],
                                     [-19.156, -17.244, -13.548, -8.763, -2.276, 3.284, 9.406, 15.742, 18.03, 22.126, 33.763, 44.393, 50.754, 57.041, 60.056])],
                         "189212": [([63.269, 61.455, 57.889, 54.315, 48.427, 41.876, 39.39, 36.499, 28.309, 20.33, 17.651, 15.967, 12.108, 9.488, 8.031, 6.387, 5.035, 4.035, 3.517, 3.107, 3.138, 3.383, 4.208, 5.704, 6.415, 7.216, 8.789, 10.363, 11.936, 13.51, 15.083, 16.657, 18.23, 19.804, 21.377, 22.15],
                                     [36.671, 35.047, 31.87, 29.269, 25.572, 21.599, 20.272, 18.912, 15.465, 12.419, 11.461, 10.899, 9.607, 8.38, 7.046, 5.114, 2.233, -0.357, -2.376, -4.83, -8.161, -11.955, -15.179, -17.938, -19.174, -20.383, -22.783, -25.184, -27.584, -29.984, -32.384, -34.784, -37.184, -39.585, -41.985, -43.176])],
                         "197270": [([-16.225, -14.856, -11.266, -7.717, -3.772, 1.019, 3.194, 3.877, 5.36, 6.843, 8.326, 9.81, 11.293, 12.776, 14.259, 15.742, 17.225, 18.708, 19.391, 20.543, 22.874, 25.206, 27.537, 29.868, 32.2, 34.531, 36.668, 37.654],
                                     [17.542, 14.583, 6.839, -0.699, -8.053, -16.537, -20.422, -21.46, -23.707, -25.953, -28.2, -30.447, -32.693, -34.94, -37.187, -39.434, -41.68, -43.927, -44.965, -46.659, -50.103, -53.548, -56.992, -60.437, -63.882, -67.326, -70.439, -71.858])]
                         }
        actual_dict = {name: pairs for name, pairs in objects_future_paths}
        self.assertEqual(set(actual_dict.keys()), set(expected_list.keys()))

        for key in expected_list:
            expected_pairs = expected_list[key]
            actual_pairs = actual_dict[key]
            self.assertEqual(len(expected_pairs), len(actual_pairs))

            for (exp_x, exp_y), (act_x, act_y) in zip(expected_pairs, actual_pairs):
                self.assertFloatListsAlmostEqual(
                    exp_x, act_x, places=3, msg=f"X-values mismatch for object {key}")
                self.assertFloatListsAlmostEqual(
                    exp_y, act_y, places=3, msg=f"Y-values mismatch for object {key}")

    def test_generate_metric_base(self):
        actual_base = generate_metric_base(
            object_list_file=csv_path, frame_id=750, case_id=None, map_file=osm_path, map_origin=osm_origin)
        self.assertEqual(len(actual_base.relevant_objects), 5)


class TestLineSegment(unittest.TestCase):
    def setUp(self):
        # Setup some reusable line segments.
        # Diagonal line
        self.l1 = LineSegment(Point(0, 0), Point(10, 10))
        # Another diagonal line crossing l1
        self.l2 = LineSegment(Point(0, 10), Point(10, 0))
        # Overlapping line with l1
        self.l3 = LineSegment(Point(5, 5), Point(15, 15))
        # Horizontal line crossing l1
        self.l4 = LineSegment(Point(0, 5), Point(10, 5))
        # Line outside l1's range
        self.l5 = LineSegment(Point(15, 15), Point(20, 20))
        # Horizontal line
        self.l6 = LineSegment(Point(0, 0), Point(10, 0))
        # Parallel horizontal line
        self.l7 = LineSegment(Point(0, 1), Point(10, 1))
        # Vertical line
        self.l9 = LineSegment(Point(5, 0), Point(5, 10))
        # Diagonal line 2
        self.l10 = LineSegment(Point(20, 20), Point(30, 30))
        # Horizontal line  2
        self.l12 = LineSegment(Point(0, 0), Point(0, 10))

    def test_project(self):
        # Test projection of a point onto the segment within its range.
        point = Point(5, 5)
        projected_point, distance = self.l1.project(point)
        self.assertAlmostEqual(projected_point.x, 5.0, delta=EPSILON)
        self.assertAlmostEqual(projected_point.y, 5.0, delta=EPSILON)
        self.assertAlmostEqual(distance, 0.0, delta=EPSILON)

        # Test projection of a point outside the segment.
        point = Point(15, 15)
        projected_point, distance = self.l1.project(point)
        self.assertAlmostEqual(projected_point.x, 10.0, delta=EPSILON)
        self.assertAlmostEqual(projected_point.y, 10.0, delta=EPSILON)
        self.assertAlmostEqual(
            distance, (5.0**2 + 5.0**2) ** 0.5, delta=EPSILON)

        # Test projection clamping to the start of the segment.
        point = Point(-5, -5)
        projected_point, distance = self.l1.project(point)
        self.assertAlmostEqual(projected_point.x, 0.0, delta=EPSILON)
        self.assertAlmostEqual(projected_point.y, 0.0, delta=EPSILON)
        self.assertAlmostEqual(
            distance, (5.0**2 + 5.0**2) ** 0.5, delta=EPSILON)

        # Test projection clamping to the end of the segment.
        point = Point(16, 8)
        projected_point, distance = self.l6.project(point)
        self.assertAlmostEqual(projected_point.x, 10.0, delta=EPSILON)
        self.assertAlmostEqual(projected_point.y, 0.0, delta=EPSILON)
        self.assertAlmostEqual(distance, 10.0, delta=EPSILON)

        # Test projection onto a vertical line segment.
        point = Point(5, 5)
        projected_point, distance = self.l12.project(point)
        self.assertAlmostEqual(projected_point.x, 0.0, delta=EPSILON)
        self.assertAlmostEqual(projected_point.y, 5.0, delta=EPSILON)
        self.assertAlmostEqual(distance, 5.0, delta=EPSILON)

        # Test projection onto a horizontal line segment.
        point = Point(5, 5)
        projected_point, distance = self.l6.project(point)
        self.assertAlmostEqual(projected_point.x, 5.0, delta=EPSILON)
        self.assertAlmostEqual(projected_point.y, 0.0, delta=EPSILON)
        self.assertAlmostEqual(distance, 5.0, delta=EPSILON)

        # Test projection of a point that lies exactly on the segment.
        point = Point(5, 0)
        projected_point, distance = self.l6.project(point)
        self.assertAlmostEqual(projected_point.x, 5.0, delta=EPSILON)
        self.assertAlmostEqual(projected_point.y, 0.0, delta=EPSILON)
        self.assertAlmostEqual(distance, 0.0, delta=EPSILON)

    def test_intersection(self):
        # Test basic intersection case.
        result = self.l1.intersection(self.l2)
        self.assertIsNotNone(result, "Expected intersection but got None.")
        intersection_point, dist1, dist2 = result
        self.assertAlmostEqual(intersection_point.x, 5.0, delta=EPSILON)
        self.assertAlmostEqual(intersection_point.y, 5.0, delta=EPSILON)
        self.assertAlmostEqual(dist1, np.sqrt(0.5)*10, delta=EPSILON)
        self.assertAlmostEqual(dist2, np.sqrt(0.5)*10, delta=EPSILON)

        # Test no intersection for parallel lines.
        self.assertIsNone(self.l6.intersection(
            self.l7), "Expected no intersection for parallel lines.")

        # Test no intersection when segments do not overlap.
        self.assertIsNone(self.l1.intersection(
            self.l5), "Expected no intersection for non-overlapping segments.")

        # Test intersection exactly at endpoints.
        l8 = LineSegment(Point(10, 10), Point(21, 20))
        result = self.l1.intersection(l8)
        self.assertIsNotNone(result, "Expected intersection at endpoint.")
        intersection_point, dist1, dist2 = result
        self.assertAlmostEqual(intersection_point.x, 10.0, delta=EPSILON)
        self.assertAlmostEqual(intersection_point.y, 10.0, delta=EPSILON)
        self.assertAlmostEqual(dist1, np.sqrt(2)*10, delta=EPSILON)
        self.assertAlmostEqual(dist2, 0.0, delta=EPSILON)

        # Test intersection of vertical and horizontal lines.
        result = self.l4.intersection(self.l9)
        self.assertIsNotNone(
            result, "Expected intersection for vertical and horizontal lines.")
        intersection_point, dist1, dist2 = result
        self.assertAlmostEqual(intersection_point.x, 5.0, delta=EPSILON)
        self.assertAlmostEqual(intersection_point.y, 5.0, delta=EPSILON)
        self.assertAlmostEqual(dist1, 5.0, delta=EPSILON)
        self.assertAlmostEqual(dist2, 5.0, delta=EPSILON)

        # Test no intersection for disjoint segments on the same line.
        self.assertIsNone(self.l1.intersection(
            self.l10), "Expected no intersection for disjoint segments on the same line.")


class TestLineString(unittest.TestCase):
    def test_new_valid(self):
        # Test valid initialization
        points_list = [[0.0, 1.0, 2.0],
                       [0.0, 1.0, 2.0]]
        line = LineString(points_list)
        self.assertIsInstance(line, LineString)

        # Test error when x and y lists have different lengths
        points_list = [[0.0, 1.0],
                       [0.0, 1.0, 2.0]]
        with self.assertRaises(ValueError):
            LineString(points_list)

        # Test error when points_list does not contain exactly two lists
        points_list = [[0.0, 1.0, 2.0]]
        with self.assertRaises(ValueError):
            LineString(points_list)

    def test_intersection(self):
        line1 = LineString([[0.0, 2.0, 4.0],
                            [0.0, 0.0, 0.0]])
        line2 = LineString([[1.0, 1.0, 1.0],
                            [-1.0, 1.0, 2.0]])
        result = line1.intersection(line2)
        self.assertIsNotNone(result)
        intersection, dist1, dist2 = result[0]
        self.assertIsInstance(intersection, Point)
        self.assertAlmostEqual(intersection.x, 1)
        self.assertAlmostEqual(intersection.y, 0)
        self.assertAlmostEqual(dist1, 1.0)
        self.assertAlmostEqual(dist2, 1.0)

        # Test intersection between two LineStrings
        line1 = LineString([[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]])
        line2 = LineString([[1.0, 3.0, 5.0], [4.0, 2.0, 0.0]])
        result = line1.intersection(line2)
        self.assertIsNotNone(result)
        intersection, dist1, dist2 = result[0]
        self.assertIsInstance(intersection, Point)
        self.assertAlmostEqual(intersection.x, 2.5)
        self.assertAlmostEqual(intersection.y, 2.5)
        self.assertAlmostEqual(dist1, np.sqrt(12.5))
        self.assertAlmostEqual(dist2, np.sqrt(4.5))

        # Test when two LineStrings are completely overlapping
        line1 = LineString([[0.0, 2.0, 4.0],
                            [0.0, 2.0, 4.0]])
        line2 = LineString([[2.0, 4.0, 6.0],
                            [2.0, 4.0, 6.0]])
        result = line1.intersection(line2)
        # Overlapping segments do not return intersections
        self.assertIsNone(result)
        # Test when not all LineStrings are overlapping
        line1 = LineString([[1.0, 2.0, 4.0],
                            [0.0, 2.0, 4.0]])
        line2 = LineString([[0.0, 2.0, 4.0],
                            [0.0, 2.0, 4.0]])
        result = line1.intersection(line2)
        intersection, dist1, dist2 = result[0]
        self.assertIsNotNone(result)
        self.assertIsInstance(intersection, Point)
        self.assertAlmostEqual(intersection.x, 2.0)
        self.assertAlmostEqual(intersection.y, 2.0)
        self.assertAlmostEqual(dist1, np.sqrt(5))
        self.assertAlmostEqual(dist2, np.sqrt(8))

        # Test when multiple segments are intersecting
        line1 = LineString([[0.0, 2.0, 4.0],
                            [0.0, 2.0, 0.0]])
        line2 = LineString([[4.0, 2.0, 0.0],
                            [2.0, 0.0, 2.0]])
        result = line1.intersection(line2)
        self.assertEqual(len(result), 2)
        intersection, dist1, dist2 = result[0]
        self.assertIsNotNone(result)
        self.assertIsInstance(intersection, Point)
        self.assertAlmostEqual(intersection.x, 1.0)
        self.assertAlmostEqual(intersection.y, 1.0)
        self.assertAlmostEqual(dist1, np.sqrt(2))
        self.assertAlmostEqual(dist2, np.sqrt(8)+np.sqrt(2))
        intersection, dist1, dist2 = result[1]
        self.assertIsNotNone(result)
        self.assertIsInstance(intersection, Point)
        self.assertAlmostEqual(intersection.x, 3.0)
        self.assertAlmostEqual(intersection.y, 1.0)
        self.assertAlmostEqual(dist2, np.sqrt(2))
        self.assertAlmostEqual(dist1, np.sqrt(8)+np.sqrt(2))

    def test_first_proximity(self):
        line1 = LineString([[0.0, 2.0, 4.0, 6.0],
                            [0.0, 0.0, 0.0, 0.0]])
        line2 = LineString([[6.0, 5.0, 4.0, 3.0, 0.0],
                            [3.0, 1.0, 3.0, 1.0, 3.0]])
        result = line1.first_proximity(line2, 1.2)
        self.assertIsNotNone(result)
        point1, point2, dist1, dist2, angle_dif = result
        self.assertIsInstance(point1, Point)
        self.assertIsInstance(point2, Point)
        self.assertAlmostEqual(point1.x, 3.0)
        self.assertAlmostEqual(point1.y, 0.0)
        self.assertAlmostEqual(point2.x, 3.0)
        self.assertAlmostEqual(point2.y, 1.0)
        self.assertAlmostEqual(dist1, 3.0)
        self.assertAlmostEqual(dist2, 3*np.sqrt(5))
        self.assertAlmostEqual(angle_dif, -2.03444393)
        result = line2.first_proximity(line1, 1.2)
        self.assertIsNotNone(result)
        point1, point2, dist1, dist2, angle_dif = result
        self.assertIsInstance(point1, Point)
        self.assertIsInstance(point2, Point)
        self.assertAlmostEqual(point1.x, 5.0)
        self.assertAlmostEqual(point1.y, 1.0)
        self.assertAlmostEqual(point2.x, 5.0)
        self.assertAlmostEqual(point2.y, 0.0)
        self.assertAlmostEqual(dist1, np.sqrt(5))
        self.assertAlmostEqual(dist2, 5.0)
        self.assertAlmostEqual(angle_dif, -2.03444393)

    def test_first_proximity_point(self):
        ls = LineString([[0.0, 0.0, 1.0, 2.0, 1.0],
                         [0.0, 1.0, 1.5, 1.0, 1.0]])
        p1 = Point(0.5, 0.5)
        result = ls.first_proximity_point(p1, threshold=0.50001)
        self.assertIsNotNone(result)
        res_point, res_distance_along, angle = result
        self.assertAlmostEqual(res_point, Point(0, 0.5))
        self.assertAlmostEqual(res_distance_along, 0.5)
        self.assertAlmostEqual(angle, np.pi/2)
        p1 = Point(0.5, 0.5)
        result = ls.first_proximity_point(p1, threshold=0.5)
        self.assertIsNone(result)
        p1 = Point(3.0, 1.0)
        result = ls.first_proximity_point(p1, threshold=1.01)
        self.assertIsNotNone(result)
        res_point, res_distance_along, angle = result
        self.assertAlmostEqual(res_point, Point(2.0, 1.0))
        self.assertAlmostEqual(res_distance_along, 1+2*np.sqrt(1+0.5**2))
        self.assertAlmostEqual(angle, -0.4636476)
        p1 = Point(1.0, 0.5)
        result = ls.first_proximity_point(p1, threshold=1.01)
        self.assertIsNotNone(result)
        res_point, res_distance_along, angle = result
        self.assertAlmostEqual(res_point, Point(0, 0.5))
        self.assertAlmostEqual(res_distance_along, 0.5)
        self.assertAlmostEqual(angle, np.pi/2)


class TestLineStringFan(unittest.TestCase):

    def test_initialization(self):
        # Test that LineStringFan initializes properly with an empty list.
        fan = LineStringFan([])
        self.assertEqual(len(fan.linestrings), 0)

    def test_from_vectors_valid(self):
        # Test that from_vectors correctly creates a LineStringFan instance.
        points_lists = [
            # A spread-out linestring
            [[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, -1.0, -2.0]],
            [[2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 3.0, 4.0]]   # Another line
        ]
        fan = LineStringFan.from_vectors(points_lists)
        self.assertEqual(len(fan.linestrings), 2)

    def test_from_vectors_invalid_shape(self):
        # Test that from_vectors raises an error for invalid input shape.
        with self.assertRaises(ValueError):
            LineStringFan.from_vectors(
                [[[0.0, 1.0, 2.0]]])  # Missing y-coordinates

    def test_from_vectors_mismatched_xy(self):
        # Test that from_vectors raises an error if x and y lengths mismatch.
        with self.assertRaises(ValueError):
            LineStringFan.from_vectors([[[0.0, 1.0], [0.0, 1.0, 2.0]]])

    def test_compare_intersection(self):
        # Test all Linestrings start at the very same point
        fan1 = LineStringFan.from_vectors([
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, 1.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, 0.5, 1.0, 1.5, 2.0]]  # A parallel line
        ])
        fan2 = LineStringFan.from_vectors([
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, -1.0, -2.0, -3.0, -4.0]],
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, -0.5, -1.0, -1.5, -2.0]]
        ])
        intersections = fan1.compare_intersection(fan2)
        self.assertGreater(len(intersections), 0)
        for intersection_info in intersections:
            self.assertEqual(intersection_info[0], Point(0.0, 0.0))
            self.assertEqual(intersection_info[1], 0.0)
            self.assertEqual(intersection_info[2], 0.0)

        # Test simple intersections
        fan3 = LineStringFan.from_vectors([
            [[2.5, 2.5, 2.5], [0.0, 2.0, 2.75]]])
        intersections2 = fan1.compare_intersection(fan3)
        self.assertEqual(intersections2[0][0], Point(2.5, 0.5))
        self.assertEqual(intersections2[0][1], np.sqrt(8)+1/np.sqrt(2))
        self.assertEqual(intersections2[0][2], 0.5)

        self.assertEqual(intersections2[1][0], Point(2.5, 1.25))
        self.assertEqual(intersections2[1][1], np.sqrt(2.5**2+1.25**2))
        self.assertEqual(intersections2[1][2], 1.25)

    def test_compare_intersection_optimized_linestrings(self):
        # Test all Linestrings start at the very same point
        fan1 = LineStringFan.from_vectors([
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, 1.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, 0.5, 1.0, 1.5, 2.0]]  # A parallel line
        ])
        fan1.optimize_linestrings()
        fan2 = LineStringFan.from_vectors([
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, -1.0, -2.0, -3.0, -4.0]],
            [[0.0, 1.0, 2.0, 3.0, 4.0],
             [0.0, -0.5, -1.0, -1.5, -2.0]]
        ])
        fan2.optimize_linestrings()
        intersections = fan1.compare_intersection(fan2)
        self.assertGreater(len(intersections), 0)
        for intersection_info in intersections:
            self.assertEqual(intersection_info[0], Point(0.0, 0.0))
            self.assertEqual(intersection_info[1], 0.0)
            self.assertEqual(intersection_info[2], 0.0)

        # Test simple intersections
        fan3 = LineStringFan.from_vectors([
            [[2.5, 2.5, 2.5], [0.0, 2.0, 2.75]]])
        fan3.optimize_linestrings()
        intersections2 = fan1.compare_intersection(fan3)
        self.assertEqual(intersections2[0][0], Point(2.5, 0.5))
        self.assertEqual(intersections2[0][1], np.sqrt(8)+1/np.sqrt(2))
        self.assertEqual(intersections2[0][2], 0.5)
        self.assertEqual(intersections2[1][0], Point(2.5, 1.25))
        self.assertEqual(intersections2[1][1], np.sqrt(2.5**2+1.25**2))
        self.assertEqual(intersections2[1][2], 1.25)

    def test_compare_first_proximity_point(self):
        fan1 = LineStringFan.from_vectors([
            [[0.0, 0.0, 1.0, 2.0, 1.0],
             [0.0, 1.0, 1.5, 1.0, 1.0]]])
        fan1.optimize_linestrings()
        p1 = Point(0.5, 0.5)
        result = fan1.compare_first_proximity_point(p1, threshold=0.50001)
        self.assertIsNotNone(result)
        res_point, res_distance_along, angle_close_linestring = result
        self.assertAlmostEqual(res_point, Point(0, 0.5))
        self.assertAlmostEqual(res_distance_along, 0.5)
        self.assertAlmostEqual(angle_close_linestring, np.pi/2)
        p1 = Point(0.5, 0.5)
        result = fan1.compare_first_proximity_point(p1, threshold=0.5)
        self.assertIsNone(result)
        p1 = Point(3.0, 1.0)
        result = fan1.compare_first_proximity_point(p1, threshold=1.01)
        self.assertIsNotNone(result)
        res_point, res_distance_along, angle_close_linestring = result
        self.assertAlmostEqual(res_point, Point(2.0, 1.0))
        self.assertAlmostEqual(res_distance_along, 1+2*np.sqrt(1+0.5**2))
        self.assertAlmostEqual(angle_close_linestring, -0.4636476)
        p1 = Point(1.0, 0.5)
        result = fan1.compare_first_proximity_point(p1, threshold=1.01)
        self.assertIsNotNone(result)
        res_point, res_distance_along, angle_close_linestring = result
        self.assertAlmostEqual(res_point, Point(0, 0.5))
        self.assertAlmostEqual(res_distance_along, 0.5)
        self.assertAlmostEqual(angle_close_linestring, np.pi/2)


class TestGeometryMethods(unittest.TestCase):

    def test_point(self):
        p1 = Point(0.0, 0.0)
        p2 = Point(3.0, 4.0)

        # Test distance between two points
        self.assertAlmostEqual(p1.distance(p2), 5.0)

        # Test almost_equal method
        p3 = Point(0.000001, 0.000001)
        self.assertTrue(p1.almost_equal(p3, 1e-5))
        self.assertFalse(p1.almost_equal(p3, 1e-7))

        # Test equal
        p4 = Point(3.0, 4.0)
        self.assertTrue(p2 == p4)


if __name__ == '__main__':
    unittest.main()
