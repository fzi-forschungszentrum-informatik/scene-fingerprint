use crate::{geometry::Point, scene::ObjectState};
use std::collections::HashMap;

pub fn euclidean_distance_simple(
    relevant_objects: &[ObjectState],
    metric_results: &mut HashMap<String, HashMap<String, f64>>,
) {
    for i in 0..relevant_objects.len() {
        let obj1 = &relevant_objects[i];

        for j in i + 1..relevant_objects.len() {
            let obj2 = &relevant_objects[j];
            let distance = obj1.position.distance(&obj2.position);

            // Use `.entry()` to insert and get a mutable reference in one step
            metric_results
                .entry(obj1.id.clone())
                .or_insert_with(HashMap::new)
                .insert(obj2.id.clone(), distance);

            metric_results
                .entry(obj2.id.clone())
                .or_insert_with(HashMap::new)
                .insert(obj1.id.clone(), distance);
        }
    }
}

pub fn generate_circle_center_approximation(object_state: &ObjectState) -> Vec<Point> {
    let mut circle_center_points: Vec<Point> = Vec::with_capacity(2);

    let length = object_state.length.unwrap();
    let width = object_state.width.unwrap();

    if length / width < 1.4 {
        circle_center_points.push(object_state.position);
    } else {
        let half_length = length / 2.;
        let half_width = width / 2.;
        let offset_x = (half_length - half_width) * object_state.psi_rad.cos();
        let offset_y = (half_length - half_width) * object_state.psi_rad.sin();
        let circle1_center = Point {
            x: object_state.position.x - offset_x,
            y: object_state.position.y - offset_y,
        };
        let circle2_center = Point {
            x: object_state.position.x + offset_x,
            y: object_state.position.y + offset_y,
        };
        circle_center_points.push(circle1_center);
        circle_center_points.push(circle2_center);
    }
    circle_center_points
}

pub fn euclidean_distance_approximated(
    relevant_objects: &[ObjectState],
    metric_results: &mut HashMap<String, HashMap<String, f64>>,
) {
    let rel_obj_points: Vec<(Vec<Point>, f64, String)> = relevant_objects
        .iter()
        .map(|object| {
            (
                generate_circle_center_approximation(object),
                object.width.unwrap(),
                object.id.clone(),
            )
        })
        .collect();

    for (i, (circle_center_points1, width1, id1)) in rel_obj_points.iter().enumerate() {
        for (circle_center_points2, width2, id2) in rel_obj_points.iter().skip(i + 1) {
            let distance = circle_center_points1
                .iter()
                .flat_map(|point1| {
                    circle_center_points2
                        .iter()
                        .map(move |point2| point1.distance(point2) - width1 / 2.0 - width2 / 2.0)
                })
                .fold(f64::INFINITY, |a, b| a.min(b));

            metric_results
                .entry(id1.clone())
                .or_insert_with(HashMap::new)
                .insert(id2.clone(), distance);

            metric_results
                .entry(id2.clone())
                .or_insert_with(HashMap::new)
                .insert(id1.clone(), distance);
        }
    }
}
