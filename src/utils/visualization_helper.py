import matplotlib
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib import cm as cm
from utils import map_vis_lanelet2

import itertools
import os


def draw_weighted_adj_matrix(weighted_matrix: np.array, axes=None):
    if axes is None:
        _, axes = plt.subplots()
    axes.imshow(weighted_matrix, interpolation='nearest')
    for i in range(weighted_matrix.shape[0]):
        for j in range(weighted_matrix.shape[1]):
            c = round(weighted_matrix[j, i], 2)
            axes.text(i, j, str(c), va='center', ha='center')
    return axes


def draw_3d_line_poly_collection(axes: Axes3D = None,
                                 poly_collection: Poly3DCollection = None,
                                 line_collection: Line3DCollection = None):
    if axes is None or not isinstance(axes, Axes3D):
        fig = plt.figure()
        axes = Axes3D(fig)
    if poly_collection is not None:
        axes.add_collection3d(poly_collection)
    if line_collection is not None:
        axes.add_collection3d(line_collection)
    return axes


def debug_show_path(path_list, axes):
    marker_list = itertools.cycle((',', '+', '.', 'o', '*'))

    for path in path_list:
        color = np.random.rand(3,)
        marker = next(marker_list)
        axes.scatter(path[:, 0],
                     path[:, 1],
                     color=color, marker=marker)

        axes.axis("equal")
        counter = 0
        for path_point in path:
            axes.annotate(counter, (path_point[0], path_point[1]))
            axes.arrow(path_point[0], path_point[1],
                       0.2 * np.cos(path_point[2]),
                       0.2 * np.sin(path_point[2]),
                       fc=color, ec="k", head_width=.2, head_length=.2)
            counter += 1
        break


def draw_2d_actor_scene(road_objects: list, map_file: str, axes=None, ):
    if axes is None:
        _, axes = plt.subplots()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lanelet_map = read_to_lanelet_map(map_file, origin=(0, 0))
    map_vis_lanelet2.draw_lanelet_map(lanelet_map, axes)
    road_objects = assign_colors(road_objects)
    axes.axes.xaxis.set_ticklabels([])
    axes.axes.yaxis.set_ticklabels([])
    for object in road_objects:
        # -----------------------------------------------------------
        #  Plot objects and their radii
        # -----------------------------------------------------------
        if object.classification == 'car':
            rot_object = polygon_xy(object)
            # rect = matplotlib.patches.Polygon(rot_object, closed=True, zorder=20, color='#1f77b4') #blue # noqa
            # rect = matplotlib.patches.Polygon(rot_object, closed=True, zorder=20, color='#ff7f0e') #orange # noqa
            rect = matplotlib.patches.Polygon(
                rot_object, closed=True, zorder=20, color='#2ca02c')  # green
            axes.add_patch(rect)
            # axes.plot(np.append(rot_object[:, 0], rot_object[0, 0]), np.append(rot_object[:, 1], rot_object[0, 1]), color=object.color) # noqa
        else:
            axes.scatter(object.x, object.y, color=object.color, alpha=1)
            axes.scatter(object.x, object.y, marker='.', color='k', alpha=1)
        plt.text(object.x, object.y, str(object.entity_id))
    plt.gca().set_aspect('equal', adjustable='box')
    return axes


def draw_weighted_adj_matrix_with_neutral_value(weighted_matrix: np.array, labels: list,
                                                title: str, vmin=0.0, vmax=None,
                                                reverse=False, axes=None, add_neutral=True):
    if vmax is None:
        vmax = np.max(weighted_matrix)

    if add_neutral:
        if reverse:
            cols = [('white')] + [(cm.RdYlBu_r(i)) for i in range(1, 256)]
        else:
            cols = [('white')] + [(cm.RdYlBu(i)) for i in range(1, 256)]
    else:
        if reverse:
            cols = [(cm.RdYlBu_r(i)) for i in range(1, 256)]
        else:
            cols = [(cm.RdYlBu(i)) for i in range(1, 256)]
    new_map = colors.LinearSegmentedColormap.from_list('new_map', cols, N=256)
    if axes is None:
        _, axes = plt.subplots()
    im = axes.imshow(weighted_matrix, cmap=new_map, vmin=vmin, vmax=vmax)
    # Show all ticks and label them with the respective list entries
    axes.set_xticks(np.arange(len(labels)), labels=labels)
    axes.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(weighted_matrix.shape[0]):
        for j in range(weighted_matrix.shape[1]):
            text = axes.text(i, j, round(weighted_matrix[j, i], 2),
                             ha="center", va="center", color="k")
    axes.set_title(title)
    axes.set_xlabel('Vehicle Id')
    axes.set_ylabel('Vehicle Id')
    plt.colorbar(im, ax=axes)
    return axes


def draw_1d_scene_criticality(weighted_matrix: np.array, labels: list, title: str,
                              vmin=0.0, vmax=None, reverse=False, axes=None):
    if vmax is None:
        vmax = np.max(weighted_matrix)
    if reverse:
        cols = [('white')] + [(cm.RdYlBu_r(i)) for i in range(1, 256)]
    else:
        cols = [('white')] + [(cm.RdYlBu(i)) for i in range(1, 256)]
    new_map = colors.LinearSegmentedColormap.from_list('new_map', cols, N=256)
    if axes is None:
        _, axes = plt.subplots()
    weighted_matrix = np.expand_dims(weighted_matrix, axis=0)  # or axis=1
    im = axes.imshow(weighted_matrix, cmap=new_map, vmin=vmin, vmax=vmax)
    # Show all ticks and label them with the respective list entries
    axes.set_xticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(weighted_matrix.size):
        axes.text(i, 0, round(weighted_matrix[0, i], 2),
                  ha="center", va="center", color="k")
    axes.set_title(title)
    axes.set_xlabel('Vehicle Id')
    plt.colorbar(im, ax=axes)
    return axes


def assign_colors(objects):
    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=len(objects))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    i = 0
    for obj in objects:
        obj.color = scalarMap.to_rgba(i)
        i += 1
    return objects


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)],
                                          [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy(object):
    lowleft = (object.x - object.length / 2., object.y - object.width / 2.)
    lowright = (object.x + object.length / 2., object.y - object.width / 2.)
    upright = (object.x + object.length / 2., object.y + object.width / 2.)
    upleft = (object.x - object.length / 2., object.y + object.width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]),
                                np.array([object.x, object.y]), yaw=object.yaw)
