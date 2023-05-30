from csv_object_list_dataset_loader.loader import Entity, Scenario
import numpy as np
from scenario_criticality.binary_metrics.post_encroachment_time.pet import PET


class ET(PET):

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

    def calculate_metric_single(self, ego: Entity, adversary:
                                Entity, timestamp: float) -> np.ndarray:
        '''
        numpy array[N,11] where N is the number of time slices and 11 the EntityState data
            in the following order: timestamp, id, x position, y position, yaw, x velocity,
            y velocity, velocity, x acceleration, y acceleration, acceleration.
        '''
        # this assertion is not necessary for PET, but since we don't know if the actors
        # are at the map at the same time we use this to check they are not too far apart
        # ret -1: no intersection, -2: both participants are at intersection area at the same time
        # -3: intersection angle too small
        metric_et, i, j = self.entry_and_exit(adversary, ego)

        if not metric_et < 0 and self.intersection_times[i, j, 4] is not None \
                and self.intersection_times[i, j, 2] is not None:
            metric_et = self.intersection_times[i, j, 4] - self.intersection_times[i, j, 2]

        if metric_et >= 0:
            metric_et = metric_et / 1000.0

        return metric_et
