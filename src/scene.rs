use crate::geometry::Point;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Clone)]
pub struct ObjectState {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub position: Point,
    x: f64,
    y: f64,
    #[pyo3(get, set)]
    pub vx: f64,
    #[pyo3(get, set)]
    pub vy: f64,
    #[pyo3(get)]
    pub velocity_norm: f64,
    #[pyo3(get, set)]
    pub psi_rad: f64,
    #[pyo3(get, set)]
    pub ax: Option<f64>,
    #[pyo3(get, set)]
    pub ay: Option<f64>,
    #[pyo3(get, set)]
    pub width: Option<f64>,
    #[pyo3(get, set)]
    pub length: Option<f64>,
    #[pyo3(get, set)]
    pub timestamp: Option<i64>,
    #[pyo3(get, set)]
    pub classification: Option<String>,
}

#[pymethods]
impl ObjectState {
    #[pyo3(signature = (id, x, y, vx, vy, psi_rad, ax=0.0, ay=0.0, width=0.0, length=0.0, timestamp=0, classification="unknown".to_string()))]
    #[new]
    fn new(
        id: String,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        psi_rad: f64,
        ax: Option<f64>,
        ay: Option<f64>,
        width: Option<f64>,
        length: Option<f64>,
        timestamp: Option<i64>,
        classification: Option<String>,
    ) -> PyResult<ObjectState> {
        Ok(ObjectState {
            id: id,
            position: Point { x, y },
            x: x,
            y: y,
            vx: vx,
            vy: vy,
            velocity_norm: vx.hypot(vy),
            psi_rad: psi_rad,
            ax: Some(ax.unwrap_or(0.0).max(0.0)),
            ay: Some(ay.unwrap_or(0.0).max(0.0)),
            width: Some(width.unwrap_or(0.5).max(0.5)),
            length: Some(length.unwrap_or(0.5).max(0.5)),
            timestamp: Some(timestamp.unwrap_or(0).max(0)),
            classification: classification.or(Some("unknown".to_string())),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ObjectState(id: {}, x: {}, y: {}, vx: {}, vy: {}, psi_rad: {}, ax: {:?}, ay: {:?}, width: {:?}, length: {:?}, timestamp: {:?}, class: {:?})",
            self.id,
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.psi_rad,
            self.ax,
            self.ay,
            self.width,
            self.length,
            self.timestamp,
            self.classification
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Scene {
    #[pyo3(get, set)]
    pub objects: HashMap<String, ObjectState>,
    #[pyo3(get, set)]
    pub timestamp: Option<i64>,
}

#[pymethods]
impl Scene {
    #[pyo3(signature = (object_states, timestamp=None))]
    #[new]
    fn new(object_states: Vec<ObjectState>, timestamp: Option<i64>) -> PyResult<Scene> {
        let mut objects = HashMap::new();
        for obj in object_states {
            objects.insert(obj.id.to_string(), obj);
        }
        let state_timestamp = objects.values().next().and_then(|state| state.timestamp);
        Ok(Scene {
            objects,
            timestamp: timestamp.or(state_timestamp),
        })
    }
    pub fn get_object_ids(&self) -> Vec<&String> {
        let mut keys: Vec<&String> = self.objects.keys().collect();
        keys.sort();
        keys
    }
}
