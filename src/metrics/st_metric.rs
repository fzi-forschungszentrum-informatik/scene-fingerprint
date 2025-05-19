use crate::metrics::metric_base::MetricType;
use crate::{geometry::LineStringFan, scene::ObjectState};
use std::collections::HashMap;
use std::f64::consts::PI;

// TODO split longitudinal an intersecting st metrics?
// currently they are not but the name suggest that
pub fn longitudinal_st_metrics(
    relevant_objects: &[ObjectState],
    ls_fan: &HashMap<String, LineStringFan>,
    threshold: f64,
    metric_results: &mut HashMap<MetricType, HashMap<String, HashMap<String, f64>>>,
) {
    for (i, rel_obj_1) in relevant_objects.iter().enumerate() {
        let ls_fan_1 = &ls_fan[&rel_obj_1.id];
        let id_1 = &rel_obj_1.id;
        let vel_1 = rel_obj_1.velocity_norm;

        for rel_obj_2 in relevant_objects.iter().skip(i + 1) {
            let ls_fan_2 = &ls_fan[&rel_obj_2.id];
            let id_2 = &rel_obj_2.id;
            let vel_2 = rel_obj_2.velocity_norm;

            let proximity_result_with_point_1 =
                ls_fan_1.compare_first_proximity_point(&rel_obj_2.position, threshold);
            if proximity_result_with_point_1.is_some() {
                let mut spacing_1 = proximity_result_with_point_1.unwrap().1;
                let angle_of_road_1 = proximity_result_with_point_1.unwrap().2;
                let euclidean_distance = rel_obj_1.position.distance(&rel_obj_2.position);
                // projected point distance should not be way small than Euclidean distance
                // this can happen if the resulution of the futurepath is to fine
                if spacing_1 < 1.0 * euclidean_distance {
                    spacing_1 = euclidean_distance;
                }
                check_st_lon_metrics(
                    metric_results,
                    rel_obj_1,
                    rel_obj_2,
                    spacing_1,
                    angle_of_road_1,
                );
            }

            let proximity_result_with_point_2 =
                ls_fan_2.compare_first_proximity_point(&rel_obj_1.position, threshold);
            if proximity_result_with_point_2.is_some() {
                let mut spacing_2 = proximity_result_with_point_2.unwrap().1;
                let angle_of_road_2 = proximity_result_with_point_2.unwrap().2;
                let euclidean_distance = rel_obj_1.position.distance(&rel_obj_2.position);
                // projected point distance should not be way small than Euclidean distance
                // this can happen if the resulution of the futurepath is to fine
                if spacing_2 < 1.0 * euclidean_distance {
                    spacing_2 = euclidean_distance;
                }
                check_st_lon_metrics(
                    metric_results,
                    rel_obj_2,
                    rel_obj_1,
                    spacing_2,
                    angle_of_road_2,
                );
            }

            if proximity_result_with_point_1.is_none() && proximity_result_with_point_2.is_none() {
                let intersecting_proximity_vector =
                    ls_fan_1.compare_first_proximity(ls_fan_2, threshold, Some(true));
                if !intersecting_proximity_vector.is_empty() {
                    let (_point_1, _point_2, int_spacing_1, int_spacing_2, angle_dif) =
                        intersecting_proximity_vector[0];
                    check_st_int_metrics(
                        metric_results,
                        id_1,
                        id_2,
                        int_spacing_1,
                        int_spacing_2,
                        vel_1,
                        vel_2,
                    );
                }
            }
        }
    }
}

fn check_st_lon_metrics(
    metric_results: &mut HashMap<MetricType, HashMap<String, HashMap<String, f64>>>,
    rel_obj_1: &ObjectState,
    rel_obj_2: &ObjectState,
    spacing: f64,
    angle_of_road: f64,
) {
    let obj_id_1 = &rel_obj_1.id;
    let obj_id_2 = &rel_obj_2.id;
    let vel_1 = rel_obj_1.velocity_norm;
    let vel_2 = rel_obj_2.velocity_norm;

    // calculate the collision model of the obstacle
    let get_collision_length_obstacle =
        |angle_road: f64, angle_obstacle: f64, len_obstacle: f64, wid_obstacle: f64| -> f64 {
            const DEG_30: f64 = PI / 6.0; // 30 degrees
            const DEG_60: f64 = PI / 3.0; // 60 degrees
            const DEG_120: f64 = 2.0 * PI / 3.0; // 120 degrees
            const DEG_150: f64 = 5.0 * PI / 6.0; // 150 degrees

            let angle_diff = (angle_road - angle_obstacle).abs() % PI;
            let diag_half = ((len_obstacle / 2.0).powi(2) + (wid_obstacle / 2.0).powi(2)).sqrt();

            match angle_diff {
                diff if diff < DEG_30 => len_obstacle / 2.0,
                diff if diff < DEG_60 => diag_half,
                diff if diff < DEG_120 => wid_obstacle / 2.0,
                diff if diff < DEG_150 => diag_half,
                _ => len_obstacle / 2.0,
            }
        };

    let collosion_length = get_collision_length_obstacle(
        angle_of_road,
        rel_obj_2.psi_rad,
        rel_obj_2.length.unwrap_or(0.0),
        rel_obj_2.width.unwrap_or(0.0),
    ) + rel_obj_1.length.unwrap_or(0.0) / 2.0;

    let clearance = spacing - collosion_length;
    metric_results
        .entry(MetricType::Spacing)
        .or_default()
        .entry(obj_id_1.clone())
        .or_default()
        .insert(obj_id_2.clone(), spacing);

    metric_results
        .entry(MetricType::Clearance)
        .or_default()
        .entry(obj_id_1.clone())
        .or_default()
        .insert(obj_id_2.clone(), clearance);

    metric_results
        .entry(MetricType::Headway)
        .or_default()
        .entry(obj_id_1.clone())
        .or_default()
        .insert(obj_id_2.clone(), spacing / vel_1);

    metric_results
        .entry(MetricType::Gaptime)
        .or_default()
        .entry(obj_id_1.clone())
        .or_default()
        .insert(obj_id_2.clone(), clearance / vel_1);

    // Takes orientation of vehicles into account
    let velocity_delta_oriented = vel_1 - vel_2 * f64::cos(rel_obj_2.psi_rad - rel_obj_1.psi_rad);
    if velocity_delta_oriented > 0.0 {
        metric_results
            .entry(MetricType::TTC)
            .or_default()
            .entry(obj_id_1.clone())
            .or_default()
            .insert(obj_id_2.clone(), clearance / velocity_delta_oriented);
    }
}

fn check_st_int_metrics(
    metric_results: &mut HashMap<MetricType, HashMap<String, HashMap<String, f64>>>,
    obj_id_1: &String,
    obj_id_2: &String,
    int_spacing_1: f64,
    int_spacing_2: f64,
    vel_1: f64,
    vel_2: f64,
) {
    // TODO: Is the vehicle length needed (with angle of intersection?)
    let time_to_int_1 = int_spacing_1 / vel_1;
    let time_to_int_2 = int_spacing_2 / vel_2;
    metric_results
        .entry(MetricType::PredictiveEncroachmentTime)
        .or_default()
        .entry(obj_id_1.clone())
        .or_default()
        .insert(obj_id_2.clone(), time_to_int_2 - time_to_int_1);

    metric_results
        .entry(MetricType::PredictiveEncroachmentTime)
        .or_default()
        .entry(obj_id_2.clone())
        .or_default()
        .insert(obj_id_1.clone(), time_to_int_1 - time_to_int_2);
}
