mod geometry;
mod utils;
mod scene;
pub mod metrics;
// pub mod metrics;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn oncrit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(create_linestring, m)?)?;
    m.add_class::<geometry::LineStringFan>()?;
    m.add_class::<geometry::LineString>()?;
    m.add_class::<geometry::LineSegment>()?;
    m.add_class::<geometry::Point>()?;
    m.add_class::<scene::Scene>()?;
    m.add_class::<metrics::metric_base::MetricBase>()?;
    m.add_class::<scene::ObjectState>()?;
    Ok(())
}