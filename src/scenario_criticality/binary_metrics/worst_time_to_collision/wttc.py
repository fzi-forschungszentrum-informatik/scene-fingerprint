from csv_object_list_dataset_loader.loader import Scenario
import numpy as np

from scenario_criticality.base_metric import BaseMetric


class WTTC(BaseMetric):

    def __init__(self, scenario: Scenario, timestamp: int):
        self.radius = 0.0
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

    def get_max_acc(self, obj):
        # in ms**(-2)
        if obj.classification == 'bicycle':
            return 2.5
        else:
            return 6.9

    # this function is only tested for the interaction dataset
    def get_length(self, obj):
        if obj.length != 0.0:
            return np.sqrt((obj.length / 2) ** 2 + (obj.width / 2) ** 2)
        else:
            return 1.0  # length for a half bike

    def calculate_metric_single(self, ego, adversary) -> float:
        assert ego.timestamp == adversary.timestamp
        # -----------------------------------------------------------
        #  Calculate new center of mass position and radius after x seconds (timehorizon_s)
        # -----------------------------------------------------------
        # max_acc = 2.5  # max acceleration for car -> how to proceed?
        obj_max_acc = self.get_max_acc(ego)
        adversary_max_acc = self.get_max_acc(adversary)
        ego_frame = self.get_length(ego)
        adversary_frame = self.get_length(adversary)
        obj_vel_x = np.sin(ego.yaw) * ego.vel
        obj_vel_y = np.cos(ego.yaw) * ego.vel
        obj_adv_vel_x = np.sin(adversary.yaw) * adversary.vel
        obj_adv_vel_y = np.cos(adversary.yaw) * adversary.vel

        if ego.vel == 0.0 and adversary.vel == 0.0:
            worst_time_to_collision = -1
        else:
            A = -0.25 * (obj_max_acc + adversary_max_acc) ** 2
            C = -(obj_max_acc + adversary_max_acc) * (ego_frame + adversary_frame) + (
                obj_adv_vel_x - obj_vel_x) ** 2 + (
                obj_adv_vel_y - obj_vel_y) ** 2
            D = 2 * (obj_adv_vel_x - obj_vel_x) * (adversary.x - ego.x) + 2 * (
                obj_adv_vel_y - obj_vel_y) * (
                adversary.y - ego.y)
            E = (adversary.x - ego.x) ** 2 + (adversary.y - ego.y) ** 2 - (
                adversary_frame + ego_frame) ** 2
            # solve quartic equation
            solution = np.roots([A, 0, C, D, E])
            real_solution = []
            # sort out complex numbers, there should be one real number that is positive
            # and that is our solution
            for r in solution:
                real_valued = r.real[abs(r.imag) < 1e-5]
                if len(real_valued) > 0:
                    real_solution.append(real_valued[0])
            if len(real_solution) > 0 and isinstance(real_solution[0], float) and real_solution[0] >= 0.0:  # noqa
                worst_time_to_collision = np.round(float(real_solution[0]), 4)
            elif len(real_solution) > 0 and isinstance(real_solution[1], float) and real_solution[1] >= 0.0:  # noqa
                worst_time_to_collision = np.round(float(real_solution[1]), 4)
            else:
                worst_time_to_collision = -1
        self.radius = worst_time_to_collision * (ego.vel + adversary.vel)
        return worst_time_to_collision
