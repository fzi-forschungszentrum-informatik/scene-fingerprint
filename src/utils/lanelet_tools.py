import numpy as np
import math

from typing import Tuple as tuple
from typing import Dict as dict
from typing import List as list
from csv_object_list_dataset_loader import EntityState

try:
    import lanelet2
    lanelet2_flag = True

    def read_to_lanelet_map(
        osm_path: str, origin: tuple[float, float], verbose=False
    ) -> lanelet2.core.LaneletMap:
        """Conveniently creates a LaneletMap object from .osm file.

        Args:
            osm_path (str): file path to the .osm file
            origin (tuple[float, float]): coordinates to the road segment
                (must have a high level of accuracy)
            verbose (bool, optional): Set to True for additional output. Defaults to False.
        """
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(origin[0], origin[1]))
        lanelet_map, error_list = lanelet2.io.loadRobust(osm_path, projector)

        if verbose:
            print(
                f"{len(error_list)} errors, {len([lane for lane in lanelet_map.laneletLayer])} lanes")  # noqa
        return lanelet_map

    def distance_from_centerline(self, entity_id, lanelet) -> float:
        """Returns the distance of the Entity from the centerline of the Lanelet.

        Args:
            entity_id (int): The id of the Entity.
            lanelet (lanelet2.core.Lanelet): Lanelet object.

        Returns:
            float: The calculated distance between Entity and the centerline of the Lanelet.
        """
        return lanelet2.geometry.toArcCoordinates(
            lanelet2.geometry.to2D(lanelet.centerline),
            lanelet2.geometry.to2D(
                lanelet2.core.BasicPoint3d(
                    self.__scene.get_entity_state(entity_id).x,
                    self.__scene.get_entity_state(entity_id).y,
                    0,
                )
            ),
        ).distance

    def calc_direction_vector(point_a, point_b):
        norm = np.linalg.norm(point_a - point_b)
        return np.array(point_a - point_b) if norm == 0 else np.array(point_a - point_b) / norm

    def poses_along_path(lanelet_map: lanelet2.core.LaneletMap, entity: EntityState, distances):
        """TODO

        Args:
            lanelet_map (lanelet2.core.LaneletMap): _description_
            entity (EntityState): _description_
            distances (_type_): _description_

        Returns:
            _type_: _description_
        """
        point3d = lanelet2.geometry.to2D(lanelet2.core.BasicPoint3d(entity.x, entity.y, 0))
        point2d = lanelet2.geometry.to2D(point3d)
        matching_tuple = lanelet2.geometry.findNearest(
            lanelet_map.laneletLayer, point2d, 10)  # number of found lanelet = 10
        lanelet = matching_tuple[0][1]
        distance = matching_tuple[0][0]
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)

        min_length = 50
        future_lanelet_paths = graph.possiblePaths(
            lanelet, min_length, False)
        while len(future_lanelet_paths) < 1:
            min_length /= 2
            future_lanelet_paths = graph.possiblePaths(
                lanelet, min_length, False)

        proj_point_on_lanelet = lanelet2.geometry.project(
            lanelet.centerline, lanelet2.core.BasicPoint3d(entity.x, entity.y, 0))

        # find nearest point & create new linestring
        linestring_points = []
        smallest_distance = 1e10
        for index, lanelet_point in enumerate(lanelet.centerline):
            tmp_distance = lanelet2.geometry.distance(lanelet_point, proj_point_on_lanelet)
            if smallest_distance > tmp_distance:
                smallest_distance = tmp_distance
            else:
                linestring_points.append(lanelet2.core.Point3d(lanelet2.core.getId(),
                                                               lanelet_point.x,
                                                               lanelet_point.y,
                                                               lanelet_point.z))

        first_lanelet_string = lanelet2.core.ConstLineString3d(lanelet2.core.getId(),
                                                            linestring_points,
                                                            lanelet2.core.AttributeMap({"key": "value"}))  # noqa TODO

        # create linestring of all points after the index point
        paths = []

        for future_lanelet_path in future_lanelet_paths:
            future_centerlines = [first_lanelet_string]

            for i, lanelet_el in enumerate(future_lanelet_path):
                if i == 0:  # ignore the first lanelet (its already in first_lanelet_string)
                    continue
                future_centerlines.append(lanelet_el.centerline)

            compound_centerlines = lanelet2.core.CompoundLineString3d(future_centerlines)
            # accumulate distances
            distance_driven = 0

            new_poses = np.expand_dims(
                np.array((proj_point_on_lanelet.x,
                          proj_point_on_lanelet.y,
                          entity.yaw)), 1).transpose()

            for distance in distances:
                if distance > 0.01:
                    distance_driven += distance
                    point = lanelet2.geometry.interpolatedPointAtDistance(
                        compound_centerlines, distance_driven)
                    dir_vector = calc_direction_vector(
                        np.array([point.x, point.y]), [new_poses[-1][:2]])
                    yaw = math.atan2(dir_vector[0][1], dir_vector[0][0])
                    pose = np.expand_dims(np.array((point.x, point.y, yaw)),
                                          1).transpose()  # zero pose as dummy
                    new_poses = np.append(new_poses,
                                          pose,
                                          axis=0)
                else:
                    new_poses = np.append(new_poses,
                                          np.expand_dims(new_poses[-1], 1).transpose(),
                                          axis=0)

            paths.append(new_poses)

        return paths


except ImportError:
    # print("Lanelet2 could not be imported. Lanelet functionalities are disabled.")
    lanelet2_flag = False
