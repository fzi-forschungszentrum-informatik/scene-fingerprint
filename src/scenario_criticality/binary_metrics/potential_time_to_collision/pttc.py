from csv_object_list_dataset_loader.loader import EntityState, IndEntityState, Scenario
import numpy as np
from scenario_criticality.base_metric import BaseMetric


def form_points(x_coordinates: np.ndarray, y_coordinates: np.ndarray):
    """
    Form array of points with x and y coordinates of vehicle.

    Parameters:
        x_coordinates (np.ndarray): Position of vehicle along the x-axis.
        y_coordinates (np.ndarray): Position of vehicle along the y-axis.

    Returns:
        points (np.ndarray): Single 2D-array containing x and y coordinates.

    """

    points = np.concatenate([x_coordinates[:, np.newaxis], y_coordinates[:, np.newaxis]], axis=1)

    return points


class PTTC(BaseMetric):

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

        # Compute speed difference.
        delta_v = np.array([np.abs(ego.vel - adversary.vel)])

        # Compute euclidean distance
        distance = np.round(np.sqrt((ego.x - adversary.x) ** 2 + (ego.y - adversary.y) ** 2), 4)

        # Compute longitudinal acceleration of adversary.
        if isinstance(adversary, IndEntityState):
            adv_acc = [adversary.ax]
        else:
            timestamps = self._scenario.timestamps
            try:
                prev_timestamp_idx = timestamps.index(ego.timestamp) - 1
                scen = self._scenario.get_scene(timestamps[prev_timestamp_idx])
                adv_old = scen.get_entity_state(adversary.entity_id)
                adv_acc = np.array([(adversary.vel - adv_old.vel)
                                   / (ego.timestamp - timestamps[prev_timestamp_idx])])
            except KeyError:
                adv_acc = np.array([0.0])

        # Compute discriminant in quadratic equation
        discriminant = (delta_v ** 2) + (2.0 * distance * adv_acc)

        # Mask for non-negative discriminant values (where a collision might take place),
        # and for non-zero acceleration values (avoid division by 0).
        non_zero = adv_acc != 0
        collision_mask = (discriminant >= 0) & non_zero

        # Compute lhs of p-q-formula (note that lhs should be non-negative).
        lhs = np.ones(shape=discriminant.size) * (-1)
        lhs[collision_mask] = (1 / adv_acc[collision_mask]) * (- delta_v[collision_mask])

        first_root = np.ones(shape=discriminant.size) * (-1)
        second_root = np.ones(shape=discriminant.size) * (-1)

        # Compute arrays of roots
        first_root[collision_mask] = lhs[collision_mask] + (1 / adv_acc[collision_mask]) * \
            np.sqrt(discriminant[collision_mask])
        second_root[collision_mask] = lhs[collision_mask] - (1 / adv_acc[collision_mask]) * \
            np.sqrt(discriminant[collision_mask])

        # Set value of negative roots to infinity to avoid considering them as PTTC values
        first_root[collision_mask & (first_root < 0) & (first_root != -1)] = np.inf
        second_root[collision_mask & (second_root < 0) & (second_root != -1)] = np.inf

        # Roots
        roots = np.concatenate([
            first_root[collision_mask][:, np.newaxis],
            second_root[collision_mask][:, np.newaxis]],
            axis=1)

        pttc = np.ones(shape=discriminant.size) * (-1)

        # Choose the smallest root for each pair and convert to ms
        pttc[collision_mask] = np.min(roots, axis=1)

        angle_diff = np.abs(ego.yaw - adversary.yaw)
        if angle_diff > 0.523599:
            pttc[collision_mask] = -3.0
        return np.round(pttc, 2)
