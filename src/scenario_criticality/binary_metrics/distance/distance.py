from csv_object_list_dataset_loader.loader import EntityState, Scenario
import numpy as np
from shapely.geometry import Polygon

from scenario_criticality.base_metric import BaseMetric


class Distance(BaseMetric):

    def __init__(self, scenario: Scenario, timestamp: int):
        super().__init__(scenario, timestamp)

    def calculate_metric(self):
        obj_list = self._scene.entity_states
        length = len(obj_list)
        metric = np.ones((length, length)) * -10.0
        row = 0
        for ego_object in obj_list:
            col = 0
            for object_adversary in obj_list:
                if ego_object.entity_id != object_adversary.entity_id:
                    m = self.calculate_metric_single(ego_object, object_adversary)
                    metric[row, col] = m
                col += 1
            row += 1
        self.results_matrix = metric

    def calculate_metric_single(self, ego: EntityState, adversary: EntityState):
        assert ego.timestamp == adversary.timestamp
        ego_box = self.calculate_bounding_box(ego.x, ego.y, ego.yaw, ego.width, ego.length)
        obj_box = self.calculate_bounding_box(adversary.x, adversary.y,
                                              adversary.yaw,
                                              adversary.width,
                                              adversary.length)
        distance = ego_box.distance(obj_box)
        return np.round(distance, 4)

    def calculate_bounding_box(self, x, y, yaw, width, length):
        """
        Calculate the bounding box of an object
        @param x: x_pos1: x position of actor [m]
        @param y: y position of actor [m]
        @param yaw: yaw angle of actor [rad]
        @param width: length of actor [m]
        @param length: width of actor [m]
        @return: bounding box as shapely polygon object
        """
        # TODO: if position is not in the middle of the object, then 0.5 has to be replaced
        # right now, carla, interaction data set use center of mass or middle of actor
        # as position
        box_points = []
        x1 = x - np.cos(yaw) * 0.5 * length + np.sin(yaw) * 0.5 * width
        y1 = y - np.sin(yaw) * 0.5 * length - np.cos(yaw) * 0.5 * width
        box_points.append((x1, y1))

        x2 = x - np.cos(yaw) * 0.5 * length - np.sin(yaw) * 0.5 * width
        y2 = y - np.sin(yaw) * 0.5 * length + np.cos(yaw) * 0.5 * width
        box_points.append((x2, y2))

        x3 = x + np.cos(yaw) * 0.5 * length - np.sin(yaw) * 0.5 * width
        y3 = y + np.sin(yaw) * 0.5 * length + np.cos(yaw) * 0.5 * width
        box_points.append((x3, y3))

        x4 = x + np.cos(yaw) * 0.5 * length + np.sin(yaw) * 0.5 * width
        y4 = y + np.sin(yaw) * 0.5 * length - np.cos(yaw) * 0.5 * width
        box_points.append((x4, y4))
        box = Polygon(box_points)
        return box
