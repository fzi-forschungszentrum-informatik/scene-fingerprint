import matplotlib.pyplot as plt

from csv_object_list_dataset_loader.loader import Loader, EntityState, Scenario
import os
import numpy as np
from scenario_criticality.base_metric import BaseMetric


def calculate_metric_single(ego: EntityState, adversary: EntityState) -> float:
    assert ego.timestamp == adversary.timestamp
    return np.round(np.sqrt((ego.x - adversary.x) ** 2 + (ego.y - adversary.y) ** 2), 4)


class DistanceSimple(BaseMetric):

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
                    m = calculate_metric_single(ego_object, object_adversary)
                    metric[row, col] = m
                col += 1
            row += 1
        self.results_matrix = metric


if __name__ == "__main__":
    # -----------------------------------------------------------
    #  Load Object List
    # -----------------------------------------------------------
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_loader = Loader()
    dataset_path = dir_path + "/../../../example/vehicle_tracks_000.csv"
    dataset_loader.load_dataset(dataset_path)
    complete_scenario = dataset_loader.return_scenario(dataset_path)
    timestamp = 1000

    # -----------------------------------------------------------
    #  Calculate distance for full scene
    # -----------------------------------------------------------
    dist = DistanceSimple(complete_scenario, timestamp)
    dist.calculate_metric()
    distance_matrix = dist.results_matrix
    dist.visualize_matrix()

    print(dist.accumulate_to_list(np.argmin))
    print(dist.accumulate_to_scalar(np.argmin))

    plt.show()
