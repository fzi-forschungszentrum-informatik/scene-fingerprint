import lanelet2
import pandas as pd
import numpy as np
from typing import *
import warnings
from oncrit.utils.lanelet_operation import read_to_lanelet_map, get_possible_paths, get_best_fitting_lanelet
from oncrit.utils.generate_future_paths import generate_straight_path

from oncrit.oncrit import ObjectState, Scene, MetricBase

def generate_metric_base(object_list_file: str, frame_id=None, case_id=None, map_file=None, map_origin=(0, 0)) -> MetricBase:
    llet_map, routing_graph = _read_map(map_file, map_origin)
    pd_scenario = _read_objects(object_list_file, case_id)

    if frame_id is None:
        frame_id = pd_scenario["frame_id"].unique()[0]
        print(f"\"frame_id\" was not defined - picking the first occurrence: {frame_id}")

    return _make_base(pd_scenario, frame_id, llet_map, routing_graph)


def generate_metric_base_batch(object_list_file: str, case_id=None, map_file=None, map_origin=(0, 0)) -> List[Tuple[int, MetricBase]]:
    # produces one MetricBase for every frame of a scenario
    llet_map, routing_graph = _read_map(map_file, map_origin)
    pd_scenario = _read_objects(object_list_file, case_id)
    frame_ids = pd_scenario["frame_id"].unique()

    metric_base = []
    for i, frame_id in enumerate(frame_ids):
        metric_base.append((
            frame_id,
            _make_base(pd_scenario, frame_id, llet_map, routing_graph)
        ))
        
    return metric_base


def _read_map(map_file: str, map_origin: Tuple[float, float] = (0, 0)) -> Tuple[Optional[lanelet2.core.LaneletMap], Optional[lanelet2.routing.RoutingGraph]]:
    #######################
    # Check Mapinformation
    #######################
    if isinstance(map_file, str):
        if map_file.lower().endswith(".osm"):
            llet_map, routing_graph = read_to_lanelet_map(map_file, origin=map_origin)
        else:
            raise ValueError("Currently only OSM files (lanelet) are supported.")

    if map_file is None:
        warnings.warn("No map file is found - constant velocity model for all objects is assumed")
        return None, None
        
    return llet_map, routing_graph


def _read_objects(object_list_file: str, case_id: Optional[int] = None) -> pd.DataFrame:
    ##########################
    # Check Scene Information
    ##########################

    # Check what kind of object list file is loaded:
    pd_dataframe = pd.read_csv(object_list_file)
    column_names = pd_dataframe.columns.values.tolist()

    if "recordingId" in column_names and "xCenter" in column_names:
        print("InD, highD, etc. dataset is found.")
        warnings.warn("Warning: Object classes are set to unknown.")

        pd_dataframe["agent_type"] = "unknown"
        pd_dataframe['heading'] = pd_dataframe['heading'] * 3.141592/180
        pd_dataframe = pd_dataframe.rename(columns={
                                           "recordingId": "case_id",
                                           "trackId": "track_id",
                                           "frame": "frame_id",
                                           "xCenter": "x",
                                           "yCenter": "y",
                                           "heading": "psi_rad",
                                           "xVelocity": "vx",
                                           "yVelocity": "vy",
                                           }, errors="raise ")
        # ind uses 25Hz
        pd_dataframe['timestamp_ms'] = pd_dataframe['frame_id'] * 40

    elif "track_id" in column_names and "x" in column_names and "case_id" in column_names:
        print("Interaction dataset (format) is found.")

    elif "track_id" in column_names and "x" in column_names and "case_id" not in column_names:
        print("TAF dataset (format) is found.")
        pd_dataframe['case_id'] = 1.0
        case_id = 1.0

    if case_id is None:
        case_id = pd_dataframe["case_id"].unique()[0]
        print(f"\"case_id\" was not defined - picking the first occurrence: {case_id}")
        
    pd_scenario = pd_dataframe[pd_dataframe["case_id"] == case_id]
    return pd_scenario


def _make_base(df_scenario: pd.DataFrame, frame_id: int, llet_map: Optional[lanelet2.core.LaneletMap], routing_graph: Optional[lanelet2.routing.RoutingGraph]) -> MetricBase:
    objects_future_paths = []
    objects = []
    
    df_scene = df_scenario[df_scenario["frame_id"] == frame_id]

    for _, row in df_scene.iterrows():
        # The metrics are not intended to give good results for pedestrian
        # Thus, most VRU are neglected at this point
        if not any(x in row.agent_type.lower() for x in ["car", "bicycle", "truck", "unknown"]):
            warnings.warn(
                f"Object with id: {row.track_id} has the type {row.agent_type} which is not taken into account")
            continue
        obj = ObjectState(id=str(row.track_id),
                          x=row.x,
                          y=row.y,
                          vx=row.vx,
                          vy=row.vy,
                          psi_rad=row.psi_rad,
                          width=row.width,
                          length=row.length,
                          timestamp=row.timestamp_ms,
                          classification=row.agent_type)
        
        if 'ax' not in row or 'ax' not in row:
            df_scenario = df_scenario.sort_values(by=['track_id', 'timestamp_ms'])
            df_scenario['time_diff_sec'] = df_scenario.groupby('track_id')['timestamp_ms'].diff() / 1000.0

            df_scenario['dvx'] = df_scenario.groupby('track_id')['vx'].diff()
            df_scenario['dvy'] = df_scenario.groupby('track_id')['vy'].diff()

            df_scenario['ax'] = df_scenario['dvx'] / df_scenario['time_diff_sec']
            df_scenario['ay'] = df_scenario['dvy'] / df_scenario['time_diff_sec']

            # Fill NaNs or infinite values (e.g., at first row of each group)
            df_scenario['ax'] = df_scenario['ax'].replace([np.inf, -np.inf], 0).fillna(0)
            df_scenario['ay'] = df_scenario['ay'].replace([np.inf, -np.inf], 0).fillna(0)
                
                
        objects.append(obj)

        if llet_map:
            match = get_best_fitting_lanelet(
                llet_map, row.x, row.y, row.psi_rad)
            if match:
                future_paths = get_possible_paths(
                    match,
                    routing_graph,
                    max_routing_cost=70,
                    distance_threshold=5,
                    smooth_value=2)
                objects_future_paths.append((str(row.track_id), future_paths))
            else:
                warnings.warn("Object with id: ", row.track_id,
                              " did not match any road element - constant velocity model is assumed")
                future_path = generate_straight_path(
                    x_start=row.x, y_start=row.y, psi_rad=row.psi_rad, vx=row.vx, vy=row.vy)
                objects_future_paths.append((str(row.track_id), [future_path]))
        else:
            future_path = generate_straight_path(
                x_start=row.x, y_start=row.y, psi_rad=row.psi_rad, vx=row.vx, vy=row.vy)
            objects_future_paths.append((str(row.track_id), [future_path]))

    scene = Scene(objects)
    metric_base = MetricBase(scene)
    metric_base.set_future_paths(objects_future_paths)
    return metric_base