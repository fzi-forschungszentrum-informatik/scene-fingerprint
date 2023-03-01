from csv_object_list_dataset_loader.loader import EntityState, Scenario
import numpy as np

from scenario_criticality.base_metric import BaseMetric
from utils.geometry import transform_to_ego_frame
from utils.metric_helper import get_intersection_angles


class TTC(BaseMetric):

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

    def calculate_metric_single(self, ego: EntityState, adversary: EntityState) -> float:
        assert ego.timestamp == adversary.timestamp

        # check if car following scenario
        angle_diff = get_intersection_angles(ego.yaw, adversary.yaw)
        if angle_diff > 0.523599:
            ttc = -3.0
        else:
            distance = np.round(np.sqrt((ego.x - adversary.x) ** 2
                                        + (ego.y - adversary.y) ** 2), 4)

            # identify leading and following vehicle
            new_obj_x, _, _ = transform_to_ego_frame(ego, adversary)
            if new_obj_x < 0.0:
                # ego is leading
                if ego.vel >= adversary.vel:
                    ttc = -2.0
                else:
                    ttc = distance / np.abs(adversary.vel - ego.vel)
            else:
                # adversary is leading
                if adversary.vel >= ego.vel:
                    ttc = -2.0
                else:
                    ttc = distance / np.abs(ego.vel - adversary.vel)
        return np.round(ttc, 2)
