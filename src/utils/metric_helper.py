from typing import Tuple

import numpy as np
from csv_object_list_dataset_loader.loader import Entity, EntityState
from shapely.geometry import LineString
from utils.geometry import form_points, compute_distance_matrix

# t1: Ego enters intersection
# t2: Adversary enters intersection
# t3: Ego leaves intersection
# t4: Adversary leaves intersection


def get_entry_and_exit_times(adversary: Entity, ego: Entity, timestamp: float):
    '''
    t1: Ego enters intersection
    t2: Adversary enters intersection
    t3: Ego leaves intersection
    t4: Adversary leaves intersection
    @param adversary:
    @param ego:
    @param timestamp:
    @return:
    '''
    assert ego.get_entity_state(int(timestamp)) and adversary.get_entity_state(int(timestamp))
    ego_data = ego.get_all_entity_states_as_time_series()
    adv_data = adversary.get_all_entity_states_as_time_series()
    error_code = 0.0
    t1 = t2 = t3 = t4 = 0.0
    t1idx = t2idx = t3idx = t4idx = 0

    # Compute left and right outer lines of ego and adversary respectively (x and y coordinates).
    ego_outer = get_trajectory_borders(ego)
    adv_outer = get_trajectory_borders(adversary)
    # Compute closest outer boundaries and indices for departure points departure points
    closest_idx, ego_out_idx, adv_out_idx = closest_outer_lines(ego_outer, adv_outer)
    ego_closest = ego_outer[closest_idx[1]]
    adv_closest = adv_outer[closest_idx[0]]
    ego_farthest = ego_outer[ego_out_idx[1]]
    adv_farthest = adv_outer[adv_out_idx[0]]
    ego_boundaries = (ego_closest, ego_farthest)
    adv_boundaries = (adv_closest, adv_farthest)

    # Compute intersection (if there is one)
    intersections = compute_intersections(ego_boundaries, adv_boundaries)
    if intersections[0] is None or len(intersections[0]) == 0:
        error_code = -1.0  # no intersection
    else:
        adv_depar, adv_entry, ego_depar, \
            ego_entry, t1idx, t2idx, t3idx, t4idx = get_intersection_times_indexes(intersections)

        diff_angle = get_yaw_angle_difference(adv_entry, adversary, ego, ego_entry)
        if diff_angle < 60.0 * (np.pi / 180.0):
            error_code = -3.0

        [t1, t2, t3, t4] = get_intersection_times(
            adv_depar, adv_entry, adversary, ego, ego_depar, ego_entry)

        if not is_valid_intersection(t1, t2, t3, t4):
            error_code = -2.0

        # compute if intersecting area is long enough
        if not error_code < 0.0:
            ego_line_length = 0.0
            obj_line_length = 0.0
            ego_tuples = list(zip(ego_data[ego_entry[0]:ego_depar[0], 2],
                              ego_data[ego_entry[0]:ego_depar[0], 3]))
            if len(ego_tuples) > 1:
                new_ego_line = LineString(ego_tuples)
                ego_line_length = new_ego_line.length
            obj_tuples = list(zip(adv_data[adv_entry[0]:adv_depar[0], 2],
                                  adv_data[adv_entry[0]:adv_depar[0], 3]))
            if len(obj_tuples) > 1:
                new_obj_line = LineString(obj_tuples)
                obj_line_length = new_obj_line.length
            if ego_line_length > 25.0 or obj_line_length > 25.0:
                error_code = -4.0
    if is_same_vehicle(ego.get_entity_state(int(timestamp)), adversary.get_entity_state(int(timestamp))):  # noqa
        error_code = -10.0  # ego and adversary are same entity

    # sometimes depending on the type of intersection non-scalars are returned; this is just a workaround!
    if not np.isscalar(t1idx):
        t1idx = t1idx[0]

    if not np.isscalar(t2idx):
        t2idx = t2idx[0]

    if not np.isscalar(t3idx):
        t3idx = t3idx[0]

    if not np.isscalar(t4idx):
        t4idx = t4idx[0]
    return error_code, t1, t2, t3, t4, t1idx, t2idx, t3idx, t4idx


def get_intersection_times_indexes(intersections):
    adv_entry = np.array([intersections[0][0]])
    t2_idx = adv_entry
    ego_entry = np.array([intersections[1][0]])
    t1_idx = ego_entry
    adv_depar = np.array([intersections[0][-1]])
    t4_idx = adv_depar
    ego_depar = np.array([intersections[1][-1]])
    t3_idx = ego_depar
    return adv_depar, adv_entry, ego_depar, ego_entry, t1_idx, t2_idx, t3_idx, t4_idx


def get_intersection_times(adv_depar, adv_entry, adversary, ego, ego_depar, ego_entry):
    # Ego enters
    t1 = ego.get_all_entity_states_as_time_series()[ego_entry, 0]
    # Ego leaves
    t3 = ego.get_all_entity_states_as_time_series()[ego_depar, 0]
    # Adversary enters
    t2 = adversary.get_all_entity_states_as_time_series()[adv_entry, 0]
    # Adversary leaves
    t4 = adversary.get_all_entity_states_as_time_series()[adv_depar, 0]
    return [t1, t2, t3, t4]


def is_valid_intersection(t1, t2, t3, t4):
    valid = True
    if (t1 < t2 and t3 > t4) or (t2 < t1 and t4 > t3) or (t1 < t2 < t3) or (t2 < t1 < t4):
        # something went wrong: both participants are at intersection area at the same time
        valid = False
    return valid


def is_same_vehicle(ego: EntityState, adversary: EntityState):
    is_same = False
    if ego.entity_id == adversary.entity_id:
        is_same = True
    return is_same


def get_yaw_angle_difference(adv_entry, adversary, ego, ego_entry):
    if ego_entry > 10:
        ego_yaw = ego.get_all_entity_states_as_time_series()[ego_entry - 10, 4]
    else:
        ego_yaw = ego.get_all_entity_states_as_time_series()[0, 4]
    if adv_entry > 10:
        adv_yaw = adversary.get_all_entity_states_as_time_series()[adv_entry - 10, 4]
    else:
        adv_yaw = adversary.get_all_entity_states_as_time_series()[0, 4]
    diff_angle = get_intersection_angles(ego_yaw, adv_yaw)
    return diff_angle


def get_trajectory_borders(entity: Entity) -> Tuple[np.ndarray, np.ndarray]:
    entity_data = entity.get_all_entity_states_as_time_series()
    left_line = form_points(
        entity_data[:, 2] - np.sin(entity_data[:, 4]) * entity.width * 0.5,
        entity_data[:, 3] + np.cos(entity_data[:, 4]) * entity.width * 0.5
    )
    right_line = form_points(
        entity_data[:, 2] + np.sin(entity_data[:, 4]) * entity.width * 0.5,
        entity_data[:, 3] - np.cos(entity_data[:, 4]) * entity.width * 0.5
    )
    return left_line, right_line


def get_intersection_angles(ego_yaw: float, adversary_yaw: float):
    # todo: check if cases needed
    return np.abs(ego_yaw - adversary_yaw)


def compute_intersections(ego_boundaries: tuple, adv_boundaries: tuple):
    """
    - Compute all intersections in order to calculate the PET.
    - If the objects intersect, then there are exactly four intersections.
    - All parameters are arrays of points (with x and y coordinates).

    Parameters:
        @param adv_boundaries: boundary (1. and 2.) that intersects with ego's trajectory.
        @param ego_boundaries: boundary (1. and 2.) that intersects with adversary's trajectory.
    Return:
        Points (prior and after) intersection of the vehicles'
                trajectories and intersection indices.
    """

    # Initialize empty arrays

    ego_closest = ego_boundaries[0]
    ego_farthest = ego_boundaries[1]
    adv_closest = adv_boundaries[0]
    adv_farthest = adv_boundaries[1]

    common_indices1 = get_indices(ego_closest, adv_closest)
    common_indices2 = get_indices(ego_farthest, adv_closest)
    common_indices3 = get_indices(ego_closest, adv_farthest)
    common_indices4 = get_indices(ego_farthest, adv_farthest)
    idx_adv = np.concatenate(
        (common_indices1[0], common_indices2[0], common_indices3[0], common_indices4[0]))
    _, i = np.unique(idx_adv, return_index=True)
    idx_adv = np.sort(idx_adv[np.sort(i)])
    idx_ego = np.concatenate(
        (common_indices1[1], common_indices2[1], common_indices3[1], common_indices4[1]))
    _, i = np.unique(idx_ego, return_index=True)
    idx_ego = np.sort(idx_ego[np.sort(i)])
    return [idx_adv, idx_ego]


def get_indices(ego_boundary: np.ndarray, adv_boundary: np.ndarray):

    # First, we need to find the closest points of the closest outer lines.
    # For this, again, we use a distance matrix (nxn, where n is the number
    # of points).
    distance_matrix = compute_distance_matrix(ego_boundary, adv_boundary)
    boundaries_idx = np.asarray(distance_matrix < 1.0).nonzero()
    return boundaries_idx


def closest_outer_lines(ego_outer: np.ndarray, adv_outer: np.ndarray):
    """
    Find out which are the closest outer boundaries of ego and adversary with respect to
    each other, in order to compute the intersection of those boundaries.


    Parameters:
        ego_outer  (np.ndarray): Array containing points (x and y coordinates).
        adv_outer  (np.ndarray):

    """
    ego_left = ego_outer[0]
    ego_right = ego_outer[1]
    adv_left = adv_outer[0]
    adv_right = adv_outer[1]

    # Access first point of each line (do this for x and y coordinates)
    ego_left_point = np.array([ego_left[0][0], ego_left[0][1]])
    ego_right_point = np.array([ego_right[0][0], ego_right[0][1]])

    adv_left_point = np.array([adv_left[0][0], adv_right[0][1]])
    adv_right_point = np.array([adv_right[0][0], adv_right[0][1]])

    ego_points = np.concatenate([ego_left_point[np.newaxis], ego_right_point[np.newaxis]])
    adv_points = np.concatenate([adv_left_point[np.newaxis], adv_right_point[np.newaxis]])

    # Compute distance matrix

    distance_matrix = compute_distance_matrix(ego_points, adv_points)

    # In this case our matrix is only a 2x2 matrix
    #             | ego_left | ego_right|
    # -----------------------------------
    # | adv_left  |          |          |
    # -----------------------------------
    # | adv_right |          |          |

    # Finally we compute the index with the smallest distance, and we
    # return the number that encodes the closest outer lines
    flattened_index = np.argmin(distance_matrix)
    closest_idx = np.unravel_index(flattened_index, distance_matrix.shape)
    ego_out_idx = closest_idx[0], 1 - closest_idx[1]
    adv_out_idx = 1 - closest_idx[0], closest_idx[1]

    return closest_idx, ego_out_idx, adv_out_idx
