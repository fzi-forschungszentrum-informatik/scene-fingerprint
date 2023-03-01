import numpy as np
from shapely.ops import unary_union
from csv_object_list_dataset_loader.loader import EntityState


def find_closest_points(x, y, line_x, line_y):
    """
    Find closest point on trajectory to given coordinate
    @param x: x coordinates of intersections [m]
    @param y: y coordinates of intersections [m]
    @param line_x: x-coordinates of line where closest point to intersections has to be found
    @param line_y: y-coordinates of line where closest point to intersections has to be found
    @return: index of closest point (there can be more than 1 since there are several
             intersection points)
    """
    idx = np.argmin(np.sqrt((line_x - x) ** 2 + (line_y - y) ** 2))
    return idx


def get_intersection(adversary_object, ego_object, line_ego, line_obj):

    is_point = unary_union(line_ego).intersection(unary_union(line_obj))
    if is_point.geom_type == 'MultiPoint':
        ego_line_idx = 1 + find_closest_points(is_point[0].xy[0][0],
                                               is_point[0].xy[1][0],
                                               ego_object[:, 2],
                                               ego_object[:, 3])
        adversary_line_idx = 1 + find_closest_points(is_point[0].xy[0][0],
                                                     is_point[0].xy[1][0],
                                                     adversary_object[:, 2],
                                                     adversary_object[:, 3])
    else:
        ego_line_idx = 1 + find_closest_points(is_point.xy[0][0],
                                               is_point.xy[1][0],
                                               ego_object[:, 2],
                                               ego_object[:, 3])
        adversary_line_idx = 1 + find_closest_points(is_point.xy[0][0],
                                                     is_point.xy[1][0],
                                                     adversary_object[:, 2],
                                                     adversary_object[:, 3])

    # sometimes depending on the type of intersection non-scalars are returned; this is just a workaround!
    if not np.isscalar(ego_line_idx):
        ego_line_idx = ego_line_idx[0]

    if not np.isscalar(adversary_line_idx):
        adversary_line_idx = adversary_line_idx[0]

    return adversary_line_idx, ego_line_idx


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


def transform_to_ego_frame(ego: EntityState, adverasry: EntityState):
    """
    Transforms obj frame to ego frame
    @return: new frame coordiantes
    """
    tmp_x = adverasry.x - ego.x
    tmp_y = adverasry.y - ego.y
    new_obj_x = np.cos(ego.yaw) * tmp_x + np.sin(ego.yaw) * tmp_y
    new_obj_y = -np.sin(ego.yaw) * tmp_x + np.cos(ego.yaw) * tmp_y
    new_obj_yaw = adverasry.yaw - ego.yaw
    return [new_obj_x, new_obj_y, new_obj_yaw]


def compute_distance_matrix(first: np.ndarray, second: np.ndarray):
    """
    Given two arrays of points compute their corresponding distance matrix.

    Parameters:
        first  (np.ndarray): 2D-array of points in 2D-Euclidean space.
        second (np.ndarray): 2D-array of points in 2D-Euclidean space.

    Returns:
        euclidean (np.ndarray): Matrix containing Euclidean distance of between
        points in the arrays.
    """

    # Maybe there is a faster way of finding closest points (divide and conquer algorithm)?
    difference = (first[np.newaxis, :, :] - second[:, np.newaxis, :]) ** 2

    euclidean = np.sqrt(np.sum(difference, axis=-1))

    return euclidean


def line_segment_intersection(ego_points: np.ndarray, adv_points: np.ndarray):
    """
    Parameters:

    """
    # Two adjacent points forming an ego line-segment
    ego1 = ego_points[0]
    ego2 = ego_points[1]

    # Two adjacent points forming an adversary line-segment
    adv1 = adv_points[0]
    adv2 = adv_points[1]

    a1 = np.round(ego2[1] - ego1[1], 4)
    b1 = np.round(ego1[0] - ego2[0], 4)
    c1 = a1 * ego1[0] + b1 * ego1[1]

    a2 = np.round(adv2[1] - adv1[1], 4)
    b2 = np.round(adv1[0] - adv2[0], 4)
    c2 = a2 * adv1[0] + b2 * adv1[1]

    denominator = a1 * b2 - a2 * b1

    # Parallel lines
    if denominator == 0:
        return None
    intersect_x = (b2 * c1 - b1 * c2) / denominator
    intersect_y = (a1 * c2 - a2 * c1) / denominator

    # TODO: There are rounding errors in this part of the code. Either replace the
    # method of calculating intersections (e.g. with cross-product) or find
    # avoid the subtractions.

    drx0 = ego2[0] - ego1[0]
    dry0 = ego2[1] - ego1[1]
    drx1 = adv2[0] - adv1[0]
    dry1 = adv2[1] - adv1[1]

    if drx0 == 0 or dry0 == 0 or drx1 == 0 or dry1 == 0:
        return np.array([intersect_x, intersect_y])

    rx0 = (intersect_x - ego1[0]) / (ego2[0] - ego1[0])
    ry0 = (intersect_y - ego1[1]) / (ego2[1] - ego1[1])

    rx1 = (intersect_x - adv1[0]) / (adv2[0] - adv1[0])
    ry1 = (intersect_y - adv1[1]) / (adv2[1] - adv1[1])

    if (((0 <= rx0 <= 1) or (0 <= ry0 <= 1))
            and ((0 <= rx1 <= 1) or (0 <= ry1 <= 1))):

        return np.array([intersect_x, intersect_y])
    else:
        return None


def get_length_boundaries(outer_line: np.ndarray, indices: np.ndarray,
                          yaw: np.ndarray, theta: float):
    """
    Compute head and tail of vehicle at a certain point, given length, yaw and
    set of points that discretize the trajectory in two-dimensional euclidean
    space.

    Paramenters:
        outer_line (np.ndarray): Discretization of trajectory of vehicle.
        index      (np.ndarray): Encodes for the positions that are considered.
        yaw        (np.ndarray): Yaw values of the vehicle's trajectory.
        theta           (float): Length of vehicle.

    Returns:
        (Tuple[np.ndarray]): Array containing head and tail coordinates for each
                      of the given points as an input.

    """
    head_x = outer_line[indices][:, 0] + np.cos(yaw[indices]) * 0.5 * theta
    head_y = outer_line[indices][:, 1] + np.sin(yaw[indices]) * 0.5 * theta

    tail_x = outer_line[indices][:, 0] - np.cos(yaw[indices]) * 0.5 * theta
    tail_y = outer_line[indices][:, 1] - np.sin(yaw[indices]) * 0.5 * theta

    head = form_points(head_x, head_y)
    tail = form_points(tail_x, tail_y)

    return head, tail


def euclidean_distance(point1: np.ndarray, point2: np.ndarray):
    """
    Compute euclidean distance of two points

    Parameters:

        point1 (np.ndarray):
        point2 (np.ndarray):

    Returns:
        Euclidean distance
    """

    x1 = np.zeros(1)
    x2 = np.zeros(1)
    y1 = np.zeros(1)
    y2 = np.zeros(1)

    x1 = point1[:, 0]
    x2 = point2[:, 0]
    y1 = point1[:, 1]
    y2 = point2[:, 1]

    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return distance
