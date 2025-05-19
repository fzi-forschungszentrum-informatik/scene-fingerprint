# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------
#
# ---------------------------------------------------------------------
# !\file
#
# \author  Maximilian Zipfl <zipfl@fzi.de>
#
# ---------------------------------------------------------------------
import itertools
import os
from typing import Dict, List, Tuple

import lanelet2
import matplotlib.pyplot as plt
import numpy as np
from oncrit.utils.map_generator import create_figure, read_lanelets_standalone


def read_to_lanelet_map(
    osm_path: str, origin: Tuple[float, float]
) -> Tuple[lanelet2.core.LaneletMap, lanelet2.routing.RoutingGraph]:
    """
    Creates a LaneletMap object from a .osm file.

    This function reads an OpenStreetMap (OSM) file and generates a LaneletMap object,
    which represents the road network and related metadata. It also creates a RoutingGraph
    to assist in pathfinding and routing within the map.

    Args:
        osm_path (str): The file path to the .osm file containing the road network data.
        origin (Tuple[float, float]): The latitude and longitude coordinates of the map's origin.
            For INTERACTION dataset use (0,0)

    Returns:
        Tuple[lanelet2.core.LaneletMap, lanelet2.routing.RoutingGraph]:
            A tuple containing the LaneletMap object and the corresponding RoutingGraph.
    """
    projector = lanelet2.projection.UtmProjector(
        lanelet2.io.Origin(origin[0], origin[1]))
    lanelet_map, error_list = lanelet2.io.loadRobust(osm_path, projector)

    traffic_rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
    return lanelet_map, routing_graph


def remove_close_polylines(
    path_dict: Dict[int, List[Tuple[float, float]]],
    end_nodes_dict: Dict[int, bool],
    distance_threshold: float,
):
    """
    Identifies and marks as invalid the end points of paths that are closer to each other than
    the specified distance threshold.


    Args:
        path_dict (Dict[int, List[Tuple[float, float]]]):
            A dictionary where the key is an integer representing the ID of the path and
            the value is a list of tuples, where each tuple represents a point (x, y) in the path's centerline.
        end_nodes_dict (Dict[int, bool]):
            A dictionary that keeps track of whether the end node of each path is valid (True) or invalid (False).

        distance_threshold (float):
            The minimum allowable distance between the end points of different paths. If the distance
            between any two end points is less than this threshold, one end point of one path is
            marked as invalid.
    """
    last_point_of_path = [
        (lanelet_end_id, centerlines[-1])
        for lanelet_end_id, centerlines in path_dict.items()
        if end_nodes_dict[lanelet_end_id]
    ]
    invalidated_lanelets = set()

    for (end_llet_id, point1), (end_llet_id_2, point2) in itertools.combinations(
        last_point_of_path, 2
    ):
        if end_llet_id in invalidated_lanelets or end_llet_id_2 in invalidated_lanelets:
            continue

        dist = np.hypot(point1[0] - point2[0], point1[1] - point2[1])
        if dist < distance_threshold:
            invalidated_lanelets.add(end_llet_id)
            end_nodes_dict[end_llet_id] = False

            if len(invalidated_lanelets) >= len(last_point_of_path) - 1:
                break


def uniform_point_distribution(
    path: List[Tuple[float, float]], distance=1
) -> List[Tuple[float, float]]:
    NotImplemented


def get_all_possible_paths_fast(
    start_lanelet: lanelet2.core.Lanelet,
    routing_graph: lanelet2.routing.RoutingGraph,
    max_routing_cost: float = 60,
) -> Tuple[Dict[int, List[Tuple[float, float]]], Dict[int, bool]]:
    """
    Generates possible visitable nodes (lanelets) from a given root node.

    Args:
        start_lanelet (lanelet2.core.Lanelet): Lanelet representing the root node.
        routing_graph (lanelet2.routing.RoutingGraph): Graph structure to determine which lanelets can be visited next
        max_routing_cost (float, optional): Maximum distance in m which can be traveled to finde new pathes.
            Defaults to 60 (m).

    Returns:
        path_dict (Dict[int, List[Tuple[float,float]]]): Dict stores all visited lanelet IDs as keys. The value is a list
            of tuples with the x,y coordinates of all centerline points that connect from the root lanelet to the lanelet
            (including itself) defined by the key.
        end_nodes (Dict[int, bool]): Dict holds all visited lanelet ids as keys. If the value of a key is true the lanelet
            is the end node of a path; having no successor nodes
    """

    path_dict = {}
    end_nodes = {}

    def next_node_checker(vi: lanelet2.routing.LaneletVisitInformation) -> bool:
        """
        Determines if a given lanelet should be a terminating node in a path.
        This function evaluates a lanelet based on its VisitInformation and determines
        whether it should act as a terminating node within a path, or if the path should
        continue propagating using Breadth-First Search (BFS).
        The function also updates the path_dict which holds possible pathes

        Args:
            vi (lanelet2.routing.LaneletVisitInformation):
                An object containing visit information about the current lanelet (node),
                including the lanelet itself, its predecessor, and associated costs.

        Returns:
            bool:
                True if the cost of visiting the current lanelet is less than
                the maximum routing cost, indicating that the search should continue;
                otherwise, False, indicating a terminating node.
        """
        current_id = vi.lanelet.id
        predecessor_id = vi.predecessor.id
        end_nodes[current_id] = True

        if predecessor_id == current_id:
            path_dict[current_id] = [(point.x, point.y)
                                     for point in vi.lanelet.centerline]
        else:
            path_dict[current_id] = path_dict[predecessor_id] + [
                (point.x, point.y) for point in vi.lanelet.centerline
            ]
            end_nodes[predecessor_id] = False

        return vi.cost < max_routing_cost

    routing_graph.forEachSuccessor(
        start_lanelet, next_node_checker, False
    )  # does not check lane switches! Switch to True to do so

    return path_dict, end_nodes


def filter_close_polylines(
    path_dict: Dict[int, List[Tuple[float, float]]],
    end_nodes: Dict[int, bool],
    distance_threshold: float = 5,
) -> List[Tuple[List[float], List[float]]]:
    """
    Only takes pathes of the path dict that have a end node as an end -> removing identical paths segments
    Also removes pathes that end at the same approximate location (threshold)

    Args:
        path_dict (Dict[int, List[Tuple[float, float]]]): Dict stores all visited lanelet IDs as keys. The value is a list
            of tuples with the x,y coordinates of all centerline points that connect from the root lanelet to the lanelet
            (including itself) defined by the key.
        end_nodes (Dict[int, bool]): Dict holds all visited lanelet ids as keys. If the value of a key is true the lanelet
            is the end node of a path; having no successor nodes
        distance_threshold (float): Distance which is used to compare end points of paths

    Returns:
        List[Tuple[List[float], List[float]]]: List of paths.  A path is a tuple of 2 lists containing floats as x and y.
            The n-th item of both lists is the n-th coordinate (x,y)
    """
    if sum(end_nodes.values()) > 1 and distance_threshold > 0:
        remove_close_polylines(path_dict, end_nodes, distance_threshold)

    future_paths = [
        np.array(path_dict[lanelet_end_id]).T.tolist()
        for lanelet_end_id in path_dict
        if end_nodes[lanelet_end_id]
    ]
    return future_paths


def smooth_paths(
    future_paths: List[Tuple[List[float], List[float]]], smooth_value: int = 15
) -> List[Tuple[List[float], List[float]]]:
    """Smoothes path by using a moving average (low pass filter)

    Args:
        future_paths (List[Tuple[List[float], List[float]]]): List of paths. A path is a tuple of 2 lists containing floats as x and y.
            The n-th item of both lists is the n-th coordinate (x,y)
        smooth_value (int, optional): Number of neighbouring points which are taken into account. Defaults to 15.

    Returns:
        List[Tuple[List[float], List[float]]]: List of smooth paths. A path is a list of multiple tupes (x,y) coordinates
    """

    # TODO
    # uniform distance of points (set distance in m)
    # uniform_point_distribution()

    # TODO check behavior with padding
    def moving_average(path, n=15):
        kernel = np.ones(n) / n

        # Padding to handle boundary effects
        padded_path_x = np.pad(path[0], (n//2, n//2), mode='edge')
        padded_path_y = np.pad(path[1], (n//2, n//2), mode='edge')
        smoothed_x = np.convolve(padded_path_x, kernel, mode='valid').tolist()
        smoothed_y = np.convolve(padded_path_y, kernel, mode='valid').tolist()

        return smoothed_x, smoothed_y

    if smooth_value > 1:
        smooth_future_paths = []
        for path in future_paths:
            if len(path[0]) > smooth_value:
                smooth_future_paths.append(moving_average(path, smooth_value))
            else:
                smooth_future_paths.append(moving_average(path, len(path[0])))
        return smooth_future_paths

    return future_paths


def get_possible_paths(
    start_lanelet: lanelet2.core.Lanelet,
    routing_graph: lanelet2.routing.RoutingGraph,
    max_routing_cost: float = 60,
    distance_threshold: float = 5,
    smooth_value: int = 15,
) -> List[Tuple[List[float], List[float]]]:
    path_dict, end_nodes = get_all_possible_paths_fast(
        start_lanelet=start_lanelet, routing_graph=routing_graph, max_routing_cost=max_routing_cost
    )
    future_paths = filter_close_polylines(
        path_dict=path_dict, end_nodes=end_nodes, distance_threshold=distance_threshold
    )

    return smooth_paths(future_paths=future_paths, smooth_value=smooth_value)


def show_lanelets(osm_path: str, origin: Tuple[float, float] = (0, 0)):
    """
    Display lanelets from an OSM file.
    This function reads lanelets from a given OpenStreetMap (OSM) file and
    visualizes them on a plot using a dark background style. The lanelets are
    displayed in a coordinate system based on the specified origin.

    Args:
        osm_path (str): The path to the OSM file containing lanelet data.
        origin (Tuple[float, float], optional): A tuple representing the
            origin of the coordinate system in latitude and longitude.
            Defaults to (0, 0).

    Returns:
        None
    """
    plt.style.use("dark_background")
    plt.cla()
    pseudo_lanelets = read_lanelets_standalone(
        osm_path, (-10e9, 10e9, -10e9, 10e9), gnss_ref=origin
    )
    fig, _, _, _, _ = create_figure(pseudo_lanelets)
    return fig


def get_closest_lanelets_to_object(llet_map, x, y, phi=0, max_distance=5.0):
    obj = lanelet2.matching.ObjectWithCovariance2d(1, lanelet2.matching.Pose2d(
        x, y, phi), [], lanelet2.matching.PositionCovariance2d(1., 1., 0.), 2)
    matches = lanelet2.matching.getProbabilisticMatches(
        llet_map, obj, max_distance)
    return matches


def get_best_fitting_lanelet(llet_map, x, y, phi=0, max_distance=5.0):
    matches = get_closest_lanelets_to_object(llet_map, x, y, phi, max_distance)
    if len(matches) > 0:
        return matches[0].lanelet
    return None


def get_centerline_of_lanelet(llet):
    return [(point.x, point.y) for point in llet.centerline]
