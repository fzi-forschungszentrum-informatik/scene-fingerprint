from csv_object_list_dataset_loader.loader import Entity, Scenario
import numpy as np
from scenario_criticality.base_metric import BaseMetric


class PET(BaseMetric):

    def __init__(self, scenario: Scenario, timestamp: int, intersection_times=None):
        super().__init__(scenario, timestamp, intersection_times=intersection_times)

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
                                                     self._scenario.get_entity(
                                                         object_adversary.entity_id),
                                                     self._timestamp)
                    metric[row, col] = m
                col += 1
            row += 1
        self.results_matrix = metric

    def calculate_metric_single(self, ego: Entity, adversary: Entity,
                                timestamp: float) -> np.ndarray:
        """
        numpy array[N,11] where N is the number of time slices and 11 the EntityState data
            in the following order: timestamp, id, x position, y position, yaw, x velocity,
            y velocity, velocity, x acceleration, y acceleration, acceleration.
        """
        # this assertion is not necessary for PET, but since we don't know if the actors are at
        # the map at the same time we use this to check they are not too far apart
        # ret -1: no intersection, -2: both participants are at intersection area at the same time
        # -3: intersection angle too small, -4: intersecting area too long
        metric_pet, i, j = self.entry_and_exit(adversary, ego)

        if not metric_pet < 0 and self.intersection_times[i, j, 1] == self.intersection_times[i, j, 2] \
                and self.intersection_times[i, j, 3] == self.intersection_times[i, j, 4]:
            metric_pet = 0.0
        elif not metric_pet < 0 and self.intersection_times[i, j, 1] > self.intersection_times[i, j, 2]:
            metric_pet = self.intersection_times[i, j, 1] - self.intersection_times[i, j, 4]
            # Ego enters first.
        elif not metric_pet < 0:
            metric_pet = self.intersection_times[i, j, 2] - self.intersection_times[i, j, 3]

        if metric_pet > 0:
            metric_pet = metric_pet / 1000.0

        return metric_pet
