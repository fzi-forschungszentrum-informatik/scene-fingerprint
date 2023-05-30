# Hallerbach, Sven, et al. "Simulation-based identification of critical scenarios for
# cooperative and automated vehicles." SAE International
# Journal of Connected and Automated Vehicles 1.2018-01-1066 (2018): 93-106.
# Following code is a scene metric for urban scenarios, similarly divided into 4 different parts.
# it is the same as in the other file/class, however, without reference values
import matplotlib.pyplot as plt

import numpy as np
from csv_object_list_dataset_loader.loader import Scenario  # noqa
from shapely.geometry import Polygon
from scenario_criticality.base_metric import BaseMetric
from scenario_criticality.binary_metrics.distance.distance import Distance


def is_object_in_radius(ego, adversary, radius):
    """
    Returns if adversary is within radius around ego
    @param ego: ego object
    @param adversary: adversary object
    @param radius: radius for circle around ego
    @return:True, if within radius, otherwise False
    """
    distance = np.sqrt(
        (adversary.x - ego.x) ** 2 + (adversary.y - ego.y) ** 2)
    return distance < radius


def get_radius(ego):
    """
    Returns radius around ego vehicle, depending on ego velocity
    (the faster, the longer the radius)
    @param ego: ego object
    @return: radius of circle around ego vehicle
    """
    return (ego.vel * 3.6 / 10.) ** 2 + ego.length + ego.vel


class TrafficQuality(BaseMetric):

    def __init__(self, scenario: Scenario, timestamp: int):
        super().__init__(scenario, timestamp)

    def calculate_metric(self):
        scene_ssq = self._scenario.get_scene(self._timestamp)
        road_objects_ssq = scene_ssq.entity_states
        length = len(road_objects_ssq)
        tqs = np.zeros((1, length))
        row = 0
        for ego_object in road_objects_ssq:
            tq, _, _, _, _ = self.get_traffic_quality(int(ego_object.entity_id), self._timestamp)
            tqs[0, row] = tq
            row += 1
        self.results_matrix = tqs

    def get_traffic_quality(self, ego_id: int, ts: int):
        """
        Returns the scene quality, where zero is defined as the best grade and one the worst
        @param ego_id: id of ego vehicle
        @param ts: timestamp of scene
        @return: scene quality from [0.0,..,1.0]
        """
        scene_sq = self._scenario.get_scene(ts)
        timesteps = self._scenario.timestamps
        sq_result = -1.0
        macro = -1.0
        micro = -1.0
        nano = -1.0
        individual_complete = -1.0
        if scene_sq.get_entity_state(ego_id) != -1 and ts in timesteps:
            macro = self.get_macroscopic_traffic_quality(ts, ego_id)

            micro = self.get_microscopic_traffic_quality(ts, ego_id)

            nano = self.get_nanoscopic_traffic_quality(ts, ego_id)

            prev_ts = int(min(timesteps, key=lambda x:abs(x-2000)))
            individual_complete = self.get_individual_traffic_quality(
                prev_ts, ts, 100.0, ego_id)

            distance = Distance(self._scenario, ts)
            distance.calculate_metric()
            penalty = np.exp(-(distance.accumulate_to_scalar(np.argmin)[0]) / 5.0)
            sq_result = penalty * np.sqrt(macro ** 2 + micro ** 2 + nano ** 2 + individual_complete ** 2)
            #sq_result = 0.25 * macro + 0.25 * micro + 0.25 * nano + 0.25 * individual_complete
        else:
            print("Error: timestamp", ts, "or ego id ", ego_id, " are not valid!")
        return [sq_result, macro, micro, nano, individual_complete]

    def get_detailed_traffic_quality(self):
        """
        Returns traffic quality and all sub-metrics for each object within one scene
        @param ts: timestamp for scene quality
        @return: scene quality and sub-metrics as matrix (5 x obj) with values from [0.0,..,1.0]
        """
        scene_dtq = self._scenario.get_scene(self._timestamp)
        road_objects_dtq = scene_dtq.entity_states
        length = len(road_objects_dtq)
        tqs = np.zeros((4, length))
        row = 0
        for ego_object in road_objects_dtq:
            if len(road_objects_dtq) <=1:
                tq, macro, micro_complete, \
                    nano_complete, individual_complete = self.get_traffic_quality(
                        int(ego_object.entity_id), self._timestamp)
                # tqs[0, row] = tq
                tqs[0, row] = macro
                tqs[1, row] = micro_complete
                tqs[2, row] = nano_complete
                tqs[3, row] = individual_complete
                row += 1
            else:
                tqs[0, row] = 0.0
                tqs[1, row] = 0.0
                tqs[2, row] = 0.0
                tqs[3, row] = 0.0
        self.results_matrix = tqs
        return tqs

    def get_single_traffic_qualities(self, ts):
        """
        Returns scene quality for each object within one scene
        @param ts: timestamp for scene quality
        @return: scene quality as array (1 x objects) with values from [0.0,..,1.0]
        """
        scene_ssq = self._scenario.get_scene(ts)
        road_objects_ssq = scene_ssq.entity_states
        length = len(road_objects_ssq)
        tqs = np.zeros((1, length))
        row = 0
        for ego_object in road_objects_ssq:
            tq, macro, micro_complete, nano_complete, \
                individual_complete = self.get_traffic_quality(
                    int(ego_object.entity_id), ts)
            tqs[0, row] = tq
            row += 1
        return np.where(tqs > 0, tqs, np.inf).max(axis=0)

    def get_worst_values_detailed(self):
        tq = self.get_detailed_traffic_quality()
        last_idx = 0,
        last_area = 0.0
        for nth in range(len(tq[0, :])):
            poly_points = []
            poly_points.append((0.0, -tq[0, nth]))
            poly_points.append((tq[1, nth], 0.0))
            poly_points.append((0.0, tq[2, nth]))
            poly_points.append((-tq[3, nth], 0.0))
            poly = Polygon(poly_points)
            # print(str(nth) + ': ' + str(poly.area))
            if poly.area > last_area:
                last_area = poly.area
                last_idx = nth
        # max_vs = np.amax(tq, axis=1)
        # return max_vs
        return tq[:, last_idx]

    ##########################################
    #   Macroscopic traffic quality
    ##########################################
    def get_macroscopic_traffic_quality(self, ts, ego_id):
        """
               Returns macroscopic scene quality or a given object
               @param ts: timestamp for macroscopic scene quality
               @param ego_id: object id for which the scene quality should be calculated
               @return: microscopic scene quality with values from [0.0,..,1.0]
               """
        macro_scene_std, macro_scene_mean = self.get_mean_and_std_in_radius(
            ts, -1, is_object_in_radius)
        if abs(macro_scene_mean) < 0.00001:
            result = 0.0
        else:
            # this formula is called coefficient of variation
            result = macro_scene_std / macro_scene_mean
        # return term_scene, micros_scene_mean
        return result

    ##########################################
    #   Microscopic traffic quality
    ##########################################
    def get_microscopic_traffic_quality(self, ts, ego_id):
        """
        Returns microscopic scene quality or a given object
        @param ts: timestamp for macroscopic scene quality
        @param ego_id: object id for which the scene quality should be calculated
        @return: macroscopic scene quality with values from [0.0,..,1.0]
        """
        # traffic flow prediction
        vehicle_count_scene = self.get_vehicle_count(ts, -1, is_object_in_radius)
        vehicle_count_near = self.get_vehicle_count(ts, ego_id, is_object_in_radius)
        return vehicle_count_near / vehicle_count_scene

    ##########################################
    #   Nanoscopic traffic quality
    ##########################################
    def get_nanoscopic_traffic_quality(self, ts, ego_id):
        """
        Returns nanoscopic scene quality or a given object
        @param ts: timestamp for macroscopic scene quality
        @param ego_id: object id for which the scene quality should be calculated
        @return: nanoscopic scene quality with values from [0.0,..,1.0]
        """
        nano_radius_std, nano_radius_mean = self.get_mean_and_std_in_radius(
            ts, ego_id, is_object_in_radius)
        if abs(nano_radius_mean) < 0.00001:
            result = 0.0
        else:
            result = nano_radius_std / nano_radius_mean
        return result

    ##########################################
    #   individual traffic quality
    ##########################################
    def get_individual_traffic_quality(self, start_time_ms, end_time_ms, time_step_size, ego_id):
        """
        Returns individual scene quality for a given object
        @param ts: timestamp for macroscopic scene quality
        @param ego_id: object id for which the scene quality should be calculated
        @return: individual scene quality with values from [0.0,..,1.0]
        """
        ego_vel = []
        ego_acc = []
        mean_vel = 0.0
        mean_acc = 0.0
        old_ts = 0
        timesteps = self._scenario.timestamps
        for ts in timesteps: #np.arange(start_time_ms, end_time_ms + time_step_size, time_step_size):
            if ts >= start_time_ms and ts <= end_time_ms:
                scene_ind = self._scenario.get_scene(ts)
                try:
                    curr_ego = scene_ind.get_entity_state(ego_id)
                except KeyError:
                    curr_ego = -1
                if curr_ego != -1:
                    ego_vel.append(curr_ego.vel)
                    if len(ego_vel) > 1:
                        ego_acc.append(
                            (curr_ego.vel - self._scenario.get_scene(old_ts).get_entity_state(ego_id).vel) * (1000.0 / time_step_size))  # noqa
                    else:
                        ego_acc.append(0.0)
            old_ts = ts
        # returns ego's acc standard deviation and ego's mean velocity

        if len(ego_vel) > 0:
            mean_vel = np.mean(ego_vel)
        if len(ego_acc) > 0:
            mean_acc = np.mean(ego_acc)
        result = ((np.abs(mean_acc) / 1.5) + (mean_vel / 13.89)) / 2.0
        return result

    ##########################################
    #   Helper
    ##########################################
    # DOI = domain of interest (area which is measured) or radius around ego vehicle
    def get_vehicle_count(self, ts, ego, func):
        """
        Count all vehicles at scene in given timestamp that fulfill a requirement (func)
        @param ts: timestamp for counting vehicles
        @param ego: ego vehicle id
        @param func: function that vehicle needs to fulfill, e.g., being within a given radius
                     around the ego vehicle
        @return: vehicle count
        """
        scene_vc = self._scenario.get_scene(ts)
        count = 0
        if ego < 0:
            count = len(scene_vc.entity_states)
        else:
            for o in scene_vc.entity_states:
                if o.entity_id != ego and func(scene_vc.get_entity_state(ego), o,
                                               get_radius(scene_vc.get_entity_state(ego))):
                    count += 1
        return count

    def get_mean_and_std_in_radius(self, ts, ego_id, func):
        """
        Returns mean and standard deviation of object velocity within a given radius (func)
        @param ts: timestamp
        @param ego_id: ego id for center of radius
        @param func: function to calculate ego radius (depends on velocity)
        @return: mean and standard deviation of object velocity
        """
        scene_mstd = self._scenario.get_scene(ts)
        road_objects_mstd = scene_mstd.entity_states
        end_velocity = []
        if ego_id > -1:
            ego_es = scene_mstd.get_entity_state(ego_id)
            radius = get_radius(ego_es)
        else:
            radius = -1
        for o in road_objects_mstd:
            if ego_id != o.entity_id:
                if radius >= 0:
                    if func(scene_mstd.get_entity_state(ego_id), o, radius):
                        end_velocity.append(o.vel)
                else:
                    end_velocity.append(o.vel)
        if len(end_velocity) > 0:
            std_vel = np.std(end_velocity)
            mean_vel = np.mean(end_velocity)
        else:
            std_vel = 0.0
            mean_vel = 0.0
        return [std_vel, mean_vel]
