"""
    Safety Potential class. The Safety Potential describes a criticality of a traffic scene.
    It is calculated in the SafetyForceField Framework to determine the intensity and direction
    of a safety procedure.
"""
import math
import itertools
import numpy as np
from typing import List
from shapely.geometry import Point, Polygon, mapping
from shapely import affinity

from csv_object_list_dataset_loader import EntityState
import utils.lanelet_tools as lt
from utils import visualization_helper
from utils import map_vis_lanelet2
from scenario_criticality.base_metric import BaseMetric

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def normalization(x):
    q = []
    for el in x:
        q.append(np.tanh(el / 60))
    return q


class SFFActor(EntityState):  # aka State
    """ Inheritance class of an entity state. Also contain reaction time and breaking values"""

    def __init__(self, csv_obstacle, parent=None, debug=False):
        self.__dict__ = csv_obstacle.__dict__
        self._parent = parent
        self._debug = debug
        self._reaction_time = 1  # s
        self._a_min = -8  # m/s^2
        self._a_prime = -6  # m/s^2
        # List[t:s, pose_t:Point(m), pose_t_prime:Point(m)] safety procedure states
        self._pose_t_list = []
        self._v_t_list = []  # List[t:s, v:m/s, v_prime:v/s]
        self._time_to_stop = None  # t
        self._time_to_stop_t_list = []  # List[t:s, time_to_stop:s, time_to_stop_prime:s]
        self._c_t_vertex_list = []  # Points // for creating a plot
        self._c_t_polygon_list = []  # Polygons

    def get_occupied_polygons(self):
        return self._c_t_polygon_list

    def _calculate_safety_procedure(self, procedure_type=None, time_horizon=4, delta_t=0.1):
        """
        Calculate the positions of the given actor when applying the safety procedure.

        Parameters
        :param type: type of the safety procedure (standard = None is breaking and no steering)
        :param time_horizon: how long the future is calculate depending on the motion model
        :param delta_t: discretization of the time horizon.
                    It will calculate `time_horizon`/`delta_t` futures
        """

        steering = 0.0  # TODO
        reaction_time = self._reaction_time  # s
        a_min = self._a_min  # m/s^2
        a_prime = self._a_prime  # m/s^2

        s_t_list = []
        pose_t_list = []

        if procedure_type is None or procedure_type == "along_lane":
            self._time_to_stop = self.vel / a_min
            self._time_to_stop_prime = self.vel / a_prime + reaction_time
            for t in np.arange(0, time_horizon, delta_t):
                s_t = s_t_prime = 0
                v_t = self.vel + a_min * t

                if v_t > 0:
                    s_t = self.vel * t + a_min * (t**2) / 2
                    pose_t = (Point(self.x + math.cos(self.yaw) * s_t,
                                    self.y + math.sin(self.yaw) * s_t),
                              self.yaw)
                elif v_t == 0 and t == 0:
                    pose_t = (Point(self.x, self.y),
                              self.yaw)
                else:
                    pose_t = pose_t_list[-1][1]
                # min velocity is zero
                v_t = max(v_t, 0)

                v_t_prime = self.vel + a_min * max(t - reaction_time, 0)
                if v_t_prime > 0:
                    s_t_prime = self.vel * t + a_prime * \
                        ((max((t - reaction_time), 0))**2) / 2 + min(t, reaction_time) * self.vel
                    pose_t_prime = (Point(self.x + math.cos(self.yaw) * s_t_prime,
                                          self.y + math.sin(self.yaw) * s_t_prime),
                                    self.yaw)
                elif v_t_prime == 0 and t == 0:
                    pose_t_prime = (Point(self.x, self.y),
                                    self.yaw)
                else:
                    pose_t_prime = pose_t_list[-1][2]
                # min velocity is zero
                v_t_prime = max(v_t_prime, 0)

                s_t_list.append((t, s_t, s_t_prime))
                pose_t_list.append((t, pose_t, pose_t_prime))
                self._v_t_list.append((t, v_t, v_t_prime))
                time_to_stop_prime = v_t_prime / a_prime - max((reaction_time - t), 0)
                self._time_to_stop_t_list.append((t, -v_t / a_min, -time_to_stop_prime))
        else:
            print("No additional safety procedure available! \n Please use \"along_lane\" or None")

        if procedure_type == "along_lane" and lt.lanelet2_flag is True:
            distances = []
            distances_prime = []
            for i, _ in enumerate(s_t_list):
                if i < len(s_t_list) - 1:
                    distances.append(s_t_list[i + 1][1] - s_t_list[i][1])
                    distances_prime.append(s_t_list[i + 1][2] - s_t_list[i][2])

            future_paths = lt.poses_along_path(
                lanelet_map=self._parent._lanelet_map, distances=distances, entity=self)
            future_paths_prime = lt.poses_along_path(
                lanelet_map=self._parent._lanelet_map, distances=distances_prime, entity=self)
            if self._debug is True:
                visualization_helper.debug_show_path(
                    list((future_paths_prime)), axes=self._parent._mpl_path_figure.axes[0])
            poses_along_path = future_paths[0]  # TODO check
            poses_along_path_prime = future_paths_prime[0]  # TODO check
            for i, el in enumerate(pose_t_list):

                point = Point(poses_along_path[i][0], poses_along_path[i][1])
                point_prime = Point(poses_along_path_prime[i][0], poses_along_path_prime[i][1])
                pose_t_list[i] = (el[0],
                                  (point, poses_along_path[i][2]),
                                  (point_prime, poses_along_path_prime[i][2]))

        self._pose_t_list = pose_t_list

        return self._pose_t_list

    def calculate_claimed_set(self, time_horizon=4, expansion=1, safety_margin=0):
        """
        The claimed set :math:`C_A(x_A) ⊆ mathbb{R}^n x T` of actor A from state x_A is the
        union of occupied trajectories that results if the actor applies its safety procedure
        S_A starting from state x_A.
        """
        if len(self._pose_t_list) == 0:
            print("No safety procedure was calculated yet. "
                  + "Calculating standard safety procedure...")
            self._calculate_safety_procedure()

        for t, pose_t, pose_t_prime in self._pose_t_list:
            polygon, vertices = self.calculate_occupied_set(actor_state=pose_t,
                                                            actor_state_prime=pose_t_prime,
                                                            z_axis=(time_horizon - t),
                                                            expansion=expansion,
                                                            safety_margin=safety_margin)
            self._c_t_polygon_list.append(polygon)
            self._c_t_vertex_list.append(vertices)

    def calculate_occupied_set(self,
                               actor_state,
                               actor_state_prime=None,
                               z_axis=0.5,
                               expansion=1,
                               safety_margin=0) -> List[Polygon]:
        """
        The occupied set :math:`o_A(x_A) ⊆ mathbb{R}^n` of actor A is the set of points in space
        that the actor occupies as a function of its state x_A . This includes points physically
        occupied, as well as points needed to maintain a safety margin

        Create a Polygon describing the bounding box aka occupied set on an actor to a giving state
        If `actor_set_prime` is given the bounding box combines both states.

        :param actor_state: tuple[float] contains x and y coordinates of the actors state
        :param actor_state_prime: tuple[float] contains x and y coordinates of the actors state
        :param actor: SFFActor
        :param z_axis: float describes the position in time. It is dependent on the time horizon
        :param expansion: float expands(>1)/reduces(<1) the occupied_set used for uncertainties
        :param safety_margin: float enlarges the occupied_set
        """
        # force actors to have a minimum size
        length = max(self.length, 0.3)
        width = max(self.width, 0.3)

        length = length + safety_margin
        width = width + safety_margin

        x = actor_state[0].x
        y = actor_state[0].y
        x_prime = actor_state_prime[0].x
        y_prime = actor_state_prime[0].y
        yaw = actor_state[1]
        yaw_prime = actor_state_prime[1]

        # bounding box of the vehicle
        fl_bb = Point((length / 2) * expansion, (width / 2) * expansion, z_axis)
        fr_bb = Point((length / 2) * expansion, -(width / 2) * expansion, z_axis)
        rl_bb = Point(-(length / 2) * expansion, (width / 2) * expansion, z_axis)
        rr_bb = Point(-(length / 2) * expansion, -(width / 2) * expansion, z_axis)
        rl_bb = affinity.rotate(rl_bb, yaw * 180 / 3.14, (0, 0))
        rr_bb = affinity.rotate(rr_bb, yaw * 180 / 3.14, (0, 0))

        # front occupied set is defined by actor_state_prime since its in general further away
        if actor_state_prime is not None:
            fl_bb = affinity.rotate(fl_bb, yaw_prime * 180 / 3.14, (0, 0))
            fr_bb = affinity.rotate(fr_bb, yaw_prime * 180 / 3.14, (0, 0))
            front_left = Point(x_prime + fl_bb.x,
                               y_prime + fl_bb.y, z_axis)
            front_right = Point(x_prime + fr_bb.x,
                                y_prime + fr_bb.y, z_axis)
        else:
            fl_bb = affinity.rotate(fl_bb, yaw * 180 / 3.14, (0, 0))
            fr_bb = affinity.rotate(fr_bb, yaw * 180 / 3.14, (0, 0))
            front_left = Point(x + fl_bb.x, y + fl_bb.y, z_axis)
            front_right = Point(x + fr_bb.x, y + fr_bb.y, z_axis)

        rear_left = Point(x + rl_bb.x, y + rl_bb.y, z_axis)
        rear_right = Point(x + rr_bb.x, y + rr_bb.y, z_axis)

        polygon = Polygon([front_left, front_right, rear_right, rear_left])
        return polygon, mapping(polygon)["coordinates"][0]

    def draw_occupied_set(self, axes: Axes3D = None):
        """
        Actors and the corresponding claimed sets are drawn to axes
        """
        if len(self._c_t_vertex_list) == 0:
            print("No occupied set is calculated")
        else:
            poly_collection = Poly3DCollection(self._c_t_vertex_list, alpha=0.1)
            line_collection = Line3DCollection(self._c_t_vertex_list, linewidths=1.1, alpha=0.3)

            visualization_helper.draw_3d_line_poly_collection(axes=axes,
                                                              poly_collection=poly_collection,
                                                              line_collection=line_collection)


class SafetyPotential(BaseMetric):
    """
        Safety Potential class calculates the safety potential for a whole traffic scene

        :param traffic_scene: first value to be calculated
        :param second_param: second value to be calculated
    """

    def __init__(self, scenario,
                 timestamp,
                 time_horizon=4,
                 delta_t=0.1,
                 procedure_type=None,
                 lanelet_map=None,
                 normalized=True):
        super().__init__(scenario, timestamp)
        self._traffic_scene = scenario.get_scene(timestamp)
        self._time_horizon = time_horizon
        self._delta_t = delta_t
        self._lanelet_map = lanelet_map
        self._normalized = normalized
        self._procedure_type = procedure_type

        self.sff_actors = {}
        self._set_up_mpl_axes(z_limits=(-2, time_horizon + 4))

    def calculate_metric(self):
        for actor_el in self._traffic_scene.entity_states:
            sff_actor = self._set_up_safety_potential(
                actor_el, procedure_type=self._procedure_type)
            sff_actor.draw_occupied_set(self._mpl_3d_figure.axes[0])
        self.results_matrix = self._get_weighted_adjacency_matrix()

    # def visualize(self, plot_matrix=True, plot_3d=True, plot_debug=False):
    #     """Visualize matplot figures

    #     Args:
    #         plot_matrix (bool, optional): Plot weighted 3d matrix. Defaults to True.
    #         plot_3d (bool, optional): Plot claimed sets in 3d space with map if available.
    #                                     Defaults to True.
    #         plot_debug (bool, optional): Plots debug paths of actors. Defaults to False.
    #     """
    #     if not plot_matrix:
    #         plt.close(self._mpl_matrix_figure)
    #     if not plot_debug:
    #         plt.close(self._mpl_path_figure)
    #     if not plot_3d:
    #         plt.close(self._mpl_3d_figure)
    #     plt.show()

    def get_3d_scene_axes(self):
        return self._mpl_matrix_figure.axes[0]

    def save_figures(self, path_to_folder="./img/"):
        self._mpl_3d_figure.savefig(f"{path_to_folder}3d/occ_sets_{self._traffic_scene.timestamp}.png")  # noqa
        self._mpl_matrix_figure.savefig(
            f"{path_to_folder}matrix/safety_potential_matrix_{self._traffic_scene.timestamp}.png")  # noqa
        self._mpl_path_figure.savefig(f"{path_to_folder}paths/paths_{self._traffic_scene.timestamp}.png")  # noqa

    def _set_up_mpl_axes(self, z_limits=(-2, 8)):
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.axis('equal')
        ax.set_zlim3d(z_limits[0], z_limits[1])
        # find max/min vals
        min_x, min_y = 9e10, 9e10
        max_x, max_y = -9e10, -9e10
        for actor_el in self._traffic_scene.entity_states:
            min_x = min(min_x, actor_el.x)
            min_y = min(min_y, actor_el.y)
            max_x = max(max_x, actor_el.x)
            max_y = max(max_y, actor_el.y)

        ax.set_xlim3d(min_x - 40, max_x + 40)
        ax.set_ylim3d(min_y - 40, max_y + 40)
        if self._lanelet_map is not None:
            map_vis_lanelet2.draw_lanelet_map(self._lanelet_map, ax)
        self._mpl_3d_figure = fig

    def _set_up_safety_potential(self, actor: EntityState, procedure_type):
        sff_actor = SFFActor(actor, parent=self)
        sff_actor._calculate_safety_procedure(procedure_type=procedure_type)
        sff_actor.calculate_claimed_set(expansion=1)
        self.sff_actors[actor.entity_id] = sff_actor
        return sff_actor

    def _get_weighted_adjacency_matrix(self, all_entities: List[int] = None) -> np.array:
        """
        Args:
            all_entities (List[int], optional): If the scenario has other entities it often
            makes sense to link entities to specific indices in the matrix since not all
            entities are available in all scenes. Defaults to None.
        """
        entity_states = self.sff_actors
        id_index_dict = {}
        index = 0
        if all_entities is None:
            potential_matrix = np.zeros((len(entity_states), len(entity_states)))
            for entity in entity_states.values():
                id_index_dict[entity.entity_id] = index
                index += 1
        else:
            potential_matrix = np.zeros(len(all_entities), len(all_entities))
            for entity_id in all_entities:
                id_index_dict[entity_id] = index
                index += 1

        for subset in list(itertools.combinations(entity_states.values(), 2)):
            actor_a, actor_b = subset[0], subset[1]
            index_a = id_index_dict[actor_a.entity_id]
            index_b = id_index_dict[actor_b.entity_id]
            if self._normalized:
                potential_matrix[index_a, index_b],\
                    potential_matrix[index_b, index_a] = normalization(
                        self.calculate_safety_potential(actor_a, actor_b))
            else:
                potential_matrix[index_a, index_b],\
                    potential_matrix[index_b, index_a] = self.calculate_safety_potential(
                    actor_a, actor_b)
        return potential_matrix

    def _calculate_safety_procedure(self, actor, procedure_type=None, time_horizon=4, delta_t=0.1):
        """
        The safety procedure :math:`S_A` of actor A is a family of control policies that
        depend only on the actor starting state x_A and properties of the world that can be
        considered fixed, each of which brings the actor to a stop within a finite time. The
        safety procedure has a family of associated trajectories derived from any starting
        state x_A. We also require that the safety procedure results in a set of trajectories,
        each of which changes smoothly with its starting state x_A.

        Current Implementation:
        The current direction of the actor is locked and it begins to slow down by  acceleration
        values [a_min, a_prime], where a_min denotes the minimum acceleration (maximum breaking)
        amount. a_min < a_prime < 0
        """
        return actor._calculate_safety_procedure(
            procedure_type=procedure_type, time_horizon=time_horizon, delta_t=delta_t)

    def calculate_occupied_set(self, actor, z_axis=0.5, expansion=1, safety_margin=0):
        """
        The occupied set :math:`o_A(x_A) ⊆ mathbb{R}^n` of actor A is the set of points in space
        that the actor occupies as a function of its state x_A . This includes points physically
        occupied, as well as points needed to maintain a safety margin
        """
        return actor.calculate_occupied_set(z_axis, expansion, safety_margin)

    def calculate_safety_potential(self, actor_a, actor_b):
        """
        A safety potential :math:`\rho_{AB} : Ω_A \times Ω_B → mathbb{R}` of the actor pair A, B
        is a real-valued function on the combined state space that is strictly positive on
        the unsafe set and non-negative elsewhere
        """
        # Check if the actors' claimed set have already be calculated
        [actor_ch.calculate_claimed_set() for actor_ch in [actor_a, actor_b]
         if len(actor_ch._c_t_polygon_list) == 0]

        weighted_area_aggregation = 0
        weighted_area_aggregation_a = 0
        weighted_area_aggregation_b = 0
        last_intersection_area = 0
        for poly_a, poly_b, v_t_a, v_t_b, tts_a, tts_b in zip(actor_a.get_occupied_polygons(),
                                                              actor_b.get_occupied_polygons(),
                                                              actor_a._v_t_list,
                                                              actor_b._v_t_list,
                                                              actor_a._time_to_stop_t_list,
                                                              actor_b._time_to_stop_t_list):

            if not poly_a.is_valid:
                poly_a = poly_a.buffer(0)
            if not poly_b.is_valid:
                poly_b = poly_b.buffer(0)
            intersection_area = poly_a.intersection(poly_b).area
            if intersection_area != last_intersection_area:
                weighted_area_aggregation += intersection_area * (1 + v_t_a[2] + v_t_b[2])
                weighted_area_aggregation_a += intersection_area * tts_a[2]
                weighted_area_aggregation_b += intersection_area * tts_b[2]

        return round(weighted_area_aggregation_a, 3), round(weighted_area_aggregation_b, 3)


if __name__ == "__main__":
    print("Should be used as a library")
