use crate::geometry::LineStringFan;
use crate::metrics::euclidean_distance;
use crate::metrics::st_metric;
use crate::scene::ObjectState;
use crate::scene::Scene;
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
#[pyclass(eq, eq_int)]
pub enum MetricType {
    EuclideanDistanceSimple,
    EuclideanDistanceApproximated,
    Clearance,
    Spacing,
    Gaptime,
    Headway,
    TTC,
    PredictiveEncroachmentTime,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct MetricBase {
    #[pyo3(get, set)]
    pub scene: Scene,
    #[pyo3(get)]
    pub relevant_objects: Vec<ObjectState>,
    #[pyo3(get)]
    pub ls_fans: HashMap<String, LineStringFan>,
    #[pyo3(get)] //          metric_name      ego_id          alter_id
    pub metric_results: HashMap<MetricType, HashMap<String, HashMap<String, f64>>>, 
}

#[pymethods]
impl MetricBase {
    #[new]
    fn new(scene: Scene) -> PyResult<MetricBase> {
        let objects = scene.objects.values().cloned().collect();
        Ok(MetricBase {
            scene: scene,
            relevant_objects: objects,
            ls_fans: HashMap::new(),
            metric_results: HashMap::new(),
        })
    }
    #[pyo3(signature = (ids=None))]
    fn filter_relevant_objects(&mut self, ids: Option<Vec<String>>) {
        self.metric_results.clear();
        println!("Warning! Cached metric results were cleared!");
        let relevant_objects: Vec<ObjectState> = match ids {
            Some(id_list) => {
                // If IDs are provided, use only the objects with those IDs
                id_list
                    .iter()
                    .filter_map(|id| self.scene.objects.get(id))
                    .cloned()
                    .collect()
            }
            None => {
                // If no IDs are provided, use all objects in the scene
                self.scene.objects.values().cloned().collect()
            }
        };
        self.relevant_objects = relevant_objects;
    }

    // deposes a LSF for each object in the vector
    pub fn set_future_paths(&mut self, id_ls_tuple_vec: Vec<(String, Vec<Vec<Vec<f64>>>)>) {
        for (object_id, lsf_coords) in id_ls_tuple_vec.iter() {
            let mut lsf = LineStringFan::from_vectors(lsf_coords.clone()).unwrap();
            lsf.project_and_cut(self.scene.objects[object_id].position);
            lsf.optimize_linestrings(Some(0.05));
            self.ls_fans.insert(object_id.to_string(), lsf);
        }
    }

    pub fn euclidean_distance_simple(&mut self) -> Option<Vec<(String, String, f64)>> {
        let metric_map = self
            .metric_results
            .entry(MetricType::EuclideanDistanceSimple)
            .or_insert_with(HashMap::new);
        euclidean_distance::euclidean_distance_simple(&self.relevant_objects, metric_map);

        self.metric_results
            .get(&MetricType::EuclideanDistanceSimple)
            .and_then(|map| flatten_hashmap(map, true))
    }

    pub fn euclidean_distance_approximated(&mut self) -> Option<Vec<(String, String, f64)>> {
        let metric_map = self
            .metric_results
            .entry(MetricType::EuclideanDistanceApproximated)
            .or_insert_with(HashMap::new);
        euclidean_distance::euclidean_distance_approximated(&self.relevant_objects, metric_map);

        self.metric_results
            .get(&MetricType::EuclideanDistanceApproximated)
            .and_then(|map| flatten_hashmap(map, true))
    }

    fn longitudinal_st_metrics(&mut self, threshold: f64) {
        st_metric::longitudinal_st_metrics(
            &self.relevant_objects,
            &self.ls_fans,
            threshold,
            &mut self.metric_results,
        );
    }

    pub fn spacing(&mut self, threshold: f64) -> Option<Vec<(String, String, f64)>> {
        if !self.metric_results.contains_key(&MetricType::Spacing) {
            self.longitudinal_st_metrics(threshold);
        }
        self.metric_results
            .get(&MetricType::Spacing)
            .and_then(|map| flatten_hashmap(map, true))
    }

    pub fn clearance(&mut self, threshold: f64) -> Option<Vec<(String, String, f64)>> {
        if !self.metric_results.contains_key(&MetricType::Clearance) {
            self.longitudinal_st_metrics(threshold);
        }
        self.metric_results
            .get(&MetricType::Clearance)
            .and_then(|map| flatten_hashmap(map, true))
    }

    pub fn gaptime(&mut self, threshold: f64) -> Option<Vec<(String, String, f64)>> {
        if !self.metric_results.contains_key(&MetricType::Gaptime) {
            self.longitudinal_st_metrics(threshold);
        }
        self.metric_results
            .get(&MetricType::Gaptime)
            .and_then(|map| flatten_hashmap(map, true))
    }

    pub fn headway(&mut self, threshold: f64) -> Option<Vec<(String, String, f64)>> {
        if !self.metric_results.contains_key(&MetricType::Headway) {
            self.longitudinal_st_metrics(threshold);
        }
        self.metric_results
            .get(&MetricType::Headway)
            .and_then(|map| flatten_hashmap(map, true))
    }

    pub fn ttc(&mut self, threshold: f64) -> Option<Vec<(String, String, f64)>> {
        if !self.metric_results.contains_key(&MetricType::TTC) {
            self.longitudinal_st_metrics(threshold);
        }
        self.metric_results
            .get(&MetricType::TTC)
            .and_then(|map| flatten_hashmap(map, true))
    }

    pub fn predictive_encroachment_time(&mut self, threshold: f64) -> Option<Vec<(String, String, f64)>> {
        if !self
            .metric_results
            .contains_key(&MetricType::PredictiveEncroachmentTime)
        {
            self.longitudinal_st_metrics(threshold);
        }
        self.metric_results
            .get(&MetricType::PredictiveEncroachmentTime)
            .and_then(|map| flatten_hashmap(map, true))
    }
}

fn flatten_hashmap(
    map: &HashMap<String, HashMap<String, f64>>,
    sort: bool,
) -> Option<Vec<(String, String, f64)>> {
    if map.is_empty() || map.values().all(|inner_map| inner_map.is_empty()) {
        return None;
    }
    let mut flatten_map: Vec<(String, String, f64)> = map
        .iter()
        .flat_map(|(key1, inner_map)| {
            inner_map
                .iter()
                .map(move |(key2, &value)| (key1.clone(), key2.clone(), value))
        })
        .collect();
    if sort {
        flatten_map.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    Some(flatten_map)
}
