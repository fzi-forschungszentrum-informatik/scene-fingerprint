import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import matplotlib.colors as colors
from csv_object_list_dataset_loader.loader import Scenario
from utils.metric_helper import get_entry_and_exit_times


class BaseMetric:
    """_summary_
    """

    def __init__(self, scenario: Scenario, timestamp: int, intersection_times=None):
        self._scenario = scenario
        self._timestamp = timestamp
        self._scene = scenario.get_scene(self._timestamp)
        self.results_matrix = None
        # only needed for a subset of binary metric
        # indexes: 0: intersection_error_code; 1: t1; 2: t2; 3: t3; 4: t4; 5: t1_idx; 6: t2_idx; 7: t3_idx; 8: t4_idx;
        if intersection_times is None:
            self.intersection_times = np.empty((len(self._scene.entity_states), len(self._scene.entity_states), 9))
            self.intersection_times[:] = np.nan
        else:
            self.intersection_times = intersection_times

    def calculate_metric(self):
        """
        Abstract method for the calculation of the metric

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('No calculation implemented for calculate_metric.')

    def visualize_matrix(self, vmin=0.0, vmax=1.0, reverse=False, axes=None, add_neutral=True,
                         title='Criticality matrix'):
        """
        Visualize the weighted adjacency matrix containing criticality measures

        Args:
            vmin (float, optional): _description_. Defaults to 0.0.
            vmax (_type_, optional): _description_. Defaults to None.
            reverse (bool, optional): _description_. Defaults to False.
            axes (_type_, optional): _description_. Defaults to None.
            add_neutral (bool, optional): _description_. Defaults to True.
            title: title of matrix

        Returns:
            _type_: _description_
        """
        obj_list = self._scene.entity_states
        labels = [e.entity_id for e in obj_list]
        if vmax is None:
            vmax = np.max(self.results_matrix)

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
        im = axes.imshow(self.results_matrix, cmap=new_map, vmin=vmin, vmax=vmax)
        # Show all ticks and label them with the respective list entries
        axes.set_xticks(np.arange(len(labels)), labels=labels)
        axes.set_yticks(np.arange(len(labels)), labels=labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        if self.results_matrix.shape[0] == self.results_matrix.shape[1]:
            for i in range(self.results_matrix.shape[0]):
                for j in range(self.results_matrix.shape[1]):
                    axes.text(i, j, round(
                        self.results_matrix[j, i], 2), ha="center", va="center", color="k")
        else:
            for i in range(self.results_matrix.size):
                axes.text(i, 0, round(self.results_matrix[0, i], 2),
                          ha="center", va="center", color="k")

        # todo make title adjustable
        axes.set_title(title)
        axes.set_xlabel('Vehicle Id')
        axes.set_ylabel('Vehicle Id')
        plt.colorbar(im, ax=axes)
        return axes

    def accumulate_to_list(self, func):
        """
        Accumulate metric for each traffic participant in the scene of self.timestamp.
        @return: list with the worst metric value for each traffic participant
        """
        # todo: return index as well
        if self.results_matrix.shape[0] == self.results_matrix.shape[1]:
            if self.results_matrix is None:
                self.calculate_metric()
            acc_metric = func(np.where(self.results_matrix >= 0,
                                       self.results_matrix, np.inf), axis=1)
        else:
            acc_metric = self.results_matrix
        return acc_metric

    def accumulate_to_scalar(self, func):
        """
        Accumulate scene metric to one single value
        @return: the worst metric value from all traffic participants and the object
    _            ids of the worst pair
        """
        return_val = [-1, -1, -1]
        obj_list = self._scene.entity_states
        if self.results_matrix is None:
            self.calculate_metric()
        if len(obj_list) > 1:
            # list of indices for axis 1
            if self.results_matrix.shape[0] == self.results_matrix.shape[1]:
                idx1 = func(np.where(self.results_matrix > 0, self.results_matrix, np.inf), axis=1)
                tmp = self.results_matrix[idx1, range(len(idx1))]
                # index for axis 2
                idx2 = func(np.where(tmp > 0, tmp, np.inf))
                return_val = [self.results_matrix[idx1[idx2], idx2],
                              obj_list[idx1[idx2]].entity_id, obj_list[idx2].entity_id]
            else:
                idx1 = func(np.where(self.results_matrix > 0, self.results_matrix, np.inf), axis=1)
                return_val = [self.results_matrix[0, idx1], obj_list[idx1[0]].entity_id, -1]
        return return_val

    def get_entity_list_index(self, entity):
        count = 0
        for ent in self._scene.entity_states:
            if ent.entity_id == entity.entity_id:
                return count
            count += 1

    def entry_and_exit(self, adversary, ego):
        adversary_index = self.get_entity_list_index(adversary)
        ego_index = self.get_entity_list_index(ego)

        if math.isnan(self.intersection_times[adversary_index, ego_index, 0]):
            self.intersection_times[adversary_index, ego_index, 0], \
            self.intersection_times[adversary_index, ego_index, 1], \
            self.intersection_times[adversary_index, ego_index, 2], \
            self.intersection_times[adversary_index, ego_index, 3], \
            self.intersection_times[adversary_index, ego_index, 4], \
            self.intersection_times[adversary_index, ego_index, 5], \
            self.intersection_times[adversary_index, ego_index, 6], \
            self.intersection_times[adversary_index, ego_index, 7], \
            self.intersection_times[adversary_index, ego_index, 8] = get_entry_and_exit_times(
                adversary, ego, self._timestamp)

        return [self.intersection_times[adversary_index, ego_index, 0], adversary_index, ego_index]
