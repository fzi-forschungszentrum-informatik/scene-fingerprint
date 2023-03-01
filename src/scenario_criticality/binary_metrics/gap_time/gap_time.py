from csv_object_list_dataset_loader.loader import Entity, Scenario
import numpy as np

from utils.geometry import get_intersection
from shapely.geometry import LineString

from scenario_criticality.base_metric import BaseMetric


class GapTime(BaseMetric):

    def __init__(self, scenario: Scenario, timestamp: int, intersection_times=None):
        super().__init__(scenario, timestamp, intersection_times=intersection_times)
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self.t1_idx = None
        self.t2_idx = None
        self.t3_idx = None
        self.t4_idx = None

    def calculate_metric(self):
        obj_list = self._scene.entity_states
        length = len(obj_list)
        metric = np.ones((length, length)) * -10.0
        row = 0
        for ego_object in obj_list:
            col = 0
            for object_adversary in obj_list:
                if ego_object.entity_id != object_adversary.entity_id:
                    m = self.calculate_metric_single(self._scenario.get_entity(ego_object.entity_id),  # noqa
                                                     self._scenario.get_entity(object_adversary.entity_id))  # noqa
                    metric[row, col] = m
                col += 1
            row += 1
        self.results_matrix = metric

    def calculate_metric_single(self, ego: Entity, adversary: Entity) -> np.ndarray:
        '''
        numpy array[N,11] where N is the number of time slices and 11 the EntityState data
            in the following order: timestamp, id, x position, y position, yaw, x velocity,
            y velocity, velocity, x acceleration, y acceleration, acceleration.
        '''

        metric_gt, i, j = self.entry_and_exit(adversary, ego)

        if not metric_gt < 0.0:

            ego_data = ego.get_all_entity_states_as_time_series()
            adv_data = adversary.get_all_entity_states_as_time_series()
            line_ego = LineString(list(zip(ego_data[:, 2], ego_data[:, 3])))
            line_obj = LineString(list(zip(adv_data[:, 2], adv_data[:, 3])))

            ego_offset = int(np.where(ego_data[:, 0] == self._timestamp)[0][0])
            adv_offset = int(np.where(adv_data[:, 0] == self._timestamp)[0][0])

            adversary_line_idx = ego_line_idx = None
            if line_ego.intersects(line_obj):
                # todo solve issue with multipoint
                adversary_line_idx, ego_line_idx = get_intersection(
                    adv_data, ego_data, line_ego, line_obj)
            elif self.intersection_times[i, j, 5] is not None:
                adversary_line_idx = self.intersection_times[i, j, 6]
                ego_line_idx = self.intersection_times[i, j, 5]

            if ego_data[ego_offset, 7] > 0.0001 and adv_data[adv_offset, 7] > 0.0001:
                if ego_offset < ego_line_idx and adv_offset < adversary_line_idx:
                    ego_line_length = 0.0
                    obj_line_length = 0.0

                    ego_line_idx = int(ego_line_idx)
                    adversary_line_idx = int(adversary_line_idx)
                    ego_tuples = list(
                        zip(ego_data[ego_offset:ego_line_idx, 2],
                            ego_data[ego_offset:ego_line_idx, 3]))
                    if len(ego_tuples) > 1:
                        new_ego_line = LineString(ego_tuples)
                        ego_line_length = new_ego_line.length
                    obj_tuples = list(zip(adv_data[adv_offset:adversary_line_idx, 2],
                                          adv_data[adv_offset:adversary_line_idx, 3]))
                    if len(obj_tuples) > 1:
                        new_obj_line = LineString(obj_tuples)
                        obj_line_length = new_obj_line.length

                    time_ego = ego_line_length / ego_data[ego_offset, 7]
                    time_adversary = obj_line_length / adv_data[adv_offset, 7]
                    if time_ego > time_adversary:
                        metric_gt = time_ego - time_adversary
                    else:
                        metric_gt = time_adversary - time_ego
                else:
                    metric_gt = -6.0  # one of the participants already passed intersection area
            else:
                metric_gt = -5.0  # standing still
        return metric_gt
