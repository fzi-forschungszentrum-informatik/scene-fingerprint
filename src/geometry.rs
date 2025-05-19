use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Rust Structs
use std::f64::EPSILON;

/// A struct representing a point in 2D space.
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
}

#[pymethods]
impl Point {
    /// Creates a new `Point` with the given x and y coordinates.
    ///
    /// # Arguments
    /// * `x` - The x-coordinate of the point.
    /// * `y` - The y-coordinate of the point.
    ///
    /// # Returns
    /// A new `Point` instance.
    #[new]
    fn new(x: f64, y: f64) -> PyResult<Point> {
        Ok(Point { x, y })
    }

    /// Computes the Euclidean distance between this point and another point.
    ///
    /// # Arguments
    /// * `other` - Another `Point` instance.
    ///
    /// # Returns
    /// The Euclidean distance between the two points.
    pub fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Checks if this point is approximately equal to another point within a given epsilon.
    ///
    /// # Arguments
    /// * `other` - Another `Point` instance.
    /// * `epsilon` - The tolerance for comparing floating-point values.
    ///
    /// # Returns
    /// `true` if the points are approximately equal within the given epsilon, otherwise `false`.
    pub fn almost_equal(&self, other: &Point, epsilon: f64) -> bool {
        (self.x - other.x).abs() < epsilon && (self.y - other.y).abs() < epsilon
    }

    /// Checks if two points are exactly equal using machine epsilon as the tolerance.
    ///
    /// # Arguments
    /// * `other` - Another `Point` instance.
    ///
    /// # Returns
    /// `true` if the points are equal within the smallest possible difference, otherwise `false`.
    pub fn __eq__(&self, other: &Point) -> bool {
        self.almost_equal(other, f64::EPSILON)
    }
}

/// A struct representing a line segment between two ´Point´ elements.
#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    #[pyo3(get)]
    p1: Point,
    #[pyo3(get)]
    p2: Point,
    #[pyo3(get)]
    length: f64,
    #[pyo3(get)]
    length_squared: f64,
    #[pyo3(get)]
    r: Point,
}

#[pymethods]
impl LineSegment {
    #[new]
    /// Creates a new directed `LineSegment` between two points.
    ///
    /// # Arguments
    /// * `p1` - The first startpoint.
    /// * `p2` - The second endpoint.
    ///
    /// # Returns
    /// A new `LineSegment` instance.
    fn new(p1: Point, p2: Point) -> Self {
        let length = p1.distance(&p2);
        let r = Point {
            x: p2.x - p1.x, // Direction vector: p2 - p1
            y: p2.y - p1.y,
        };
        let length_squared = length * length;

        LineSegment {
            p1,
            p2,
            length,
            length_squared,
            r,
        }
    }
    /// Computes the intersection point of two line segments, if any.
    ///
    /// # Arguments
    /// * `other` - Another `LineSegment` instance.
    ///
    /// # Returns
    /// An `Option` containing the intersection point and its relative distances along the segments, or `None` if no intersection.
    fn intersection(&self, other: &LineSegment) -> Option<(Point, f64, f64)> {
        let p = self.p1;
        let q = other.p1;
        let r = self.r;
        let s = other.r;

        let denominator = r.x * s.y - r.y * s.x;

        if denominator.abs() < EPSILON {
            return None; // Parallel or collinear lines
        }

        let t = ((q.x - p.x) * s.y - (q.y - p.y) * s.x) / denominator;
        let u = ((q.x - p.x) * r.y - (q.y - p.y) * r.x) / denominator;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            let intersection_point = Point {
                x: p.x + t * r.x,
                y: p.y + t * r.y,
            };
            Some((intersection_point, t * self.length, u * other.length))
        } else {
            None // No intersection within the segments
        }
    }

    /// Projects a point onto the line segment and returns the closest point on the segment and its distance from the input point.
    ///
    /// # Arguments
    /// * `point` - The point to project.
    ///
    /// # Returns
    /// A tuple containing the projected point on the segment and the distance from the input point.
    pub fn project(&self, point: &Point) -> (Point, f64) {
        let px: f64 = self.p2.x - self.p1.x;
        let py = self.p2.y - self.p1.y;
        let t = ((point.x - self.p1.x) * self.r.x + (point.y - self.p1.y) * self.r.y)
            / self.length_squared;

        let t_clamped = t.clamp(0.0, 1.0);

        let point_on_segment = Point {
            x: self.p1.x + t_clamped * px,
            y: self.p1.y + t_clamped * py,
        };

        (point_on_segment, point_on_segment.distance(point))
    }
}

/// A struct representing a sequence of connected line segments.
#[pyclass]
#[derive(Debug, Clone)]
pub struct LineString {
    #[pyo3(get)]
    pub segments: Vec<LineSegment>,
    #[pyo3(get)]
    pub pre_length: f64,
}

#[pymethods]
impl LineString {
    /// Creates a new `LineString` from a list of coordinate pairs.
    ///
    /// # Arguments
    /// * `points_list` - A vector of x and y coordinates.
    /// * `pre_length` - An optional precomputed length.
    ///
    /// # Returns
    /// A new `LineString` instance.
    #[new]
    #[pyo3(signature = (points_list, pre_length=None))]
    fn new(points_list: Vec<Vec<f64>>, pre_length: Option<f64>) -> PyResult<LineString> {
        if points_list.len() != 2 {
            return Err(PyValueError::new_err(
                "Expected two lists (x and y values).",
            ));
        }
        let xs = &points_list[0];
        let ys = &points_list[1];
        if xs.len() != ys.len() {
            return Err(PyValueError::new_err(
                "x and y lists must have the same length.",
            ));
        }
        let points = points_from_arrays(xs, ys);
        let segments = line_segments_from_points(&points);

        Ok(LineString {
            segments,
            pre_length: pre_length.unwrap_or(0.0),
        })
    }

    /// Creates a `LineString` from a given list of line segments.
    ///
    /// # Arguments
    /// * `segments` - A vector of `LineSegment` instances.
    /// * `pre_length` - An optional precomputed length.
    ///
    /// # Returns
    /// A new `LineString` instance.
    #[staticmethod]
    #[pyo3(signature = (segments, pre_length=None))]
    pub fn from_segments(segments: Vec<LineSegment>, pre_length: Option<f64>) -> Self {
        LineString {
            segments,
            pre_length: pre_length.unwrap_or(0.0),
        }
    }

    /// Returns the x and y coordinates of the points in the `LineString` as separate vectors.
    ///
    /// # Returns
    ///
    /// A tuple containing two vectors:
    /// - The first vector contains the x-coordinates.
    /// - The second vector contains the y-coordinates.
    fn points_as_xy_vector(&self) -> (Vec<f64>, Vec<f64>) {
        let mut x_values = Vec::with_capacity(self.segments.len() + 1);
        let mut y_values = Vec::with_capacity(self.segments.len() + 1);
        if let Some(first) = self.segments.first() {
            x_values.push(first.p1.x);
            y_values.push(first.p1.y);
        }
        for segment in &self.segments {
            x_values.push(segment.p2.x);
            y_values.push(segment.p2.y);
        }

        (x_values, y_values)
    }

    /// Computes the intersection points between this `LineString` and another `LineString`.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another `LineString`.
    ///
    /// # Returns
    ///
    /// An `Option` containing a vector of tuples. Each tuple consists of:
    /// - The intersection `Point`
    /// - The cumulative distance along `self` to the intersection
    /// - The cumulative distance along `other` to the intersection
    ///
    /// Returns `None` if there are no intersections.
    pub fn intersection(&self, other: &LineString) -> Option<Vec<(Point, f64, f64)>> {
        let mut result = Vec::new();
        let mut dist1_total = self.pre_length;
        for seg1 in &self.segments {
            let mut dist2_total = other.pre_length;
            for seg2 in &other.segments {
                if let Some((intersection, dist1, dist2)) = seg1.intersection(seg2) {
                    result.push((intersection, dist1_total + dist1, dist2_total + dist2));
                }
                dist2_total += seg2.length;
            }
            dist1_total += seg1.length;
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Computes the shortest distance between this `LineString` and another `LineString`.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another `LineString`.
    ///
    /// # Returns
    ///
    /// The shortest Euclidean distance between the two `LineString` objects.
    pub fn shortest_distance(&self, other: &LineString) -> f64 {
        let mut min_distance = f64::INFINITY;
        for (i, seg1) in self.segments.iter().enumerate() {
            for (j, seg2) in other.segments.iter().enumerate() {
                let dist = if i == 0 && j == 0 {
                    seg1.p1.distance(&seg2.p1)
                } else if i == self.segments.len() - 1 && j == other.segments.len() - 1 {
                    seg1.p2.distance(&seg2.p2)
                } else {
                    seg1.project(&seg2.p2).1
                };
                min_distance = min_distance.min(dist);
            }
        }
        min_distance
    }

    /// Finds the first point of proximity between this `LineString` and another `LineString`
    /// within a given distance threshold.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another `LineString`.
    /// * `threshold` - A distance threshold.
    ///
    /// # Returns
    ///
    /// An `Option` containing a tuple with:
    /// - The projected point on `self`
    /// - The projected point on `other`
    /// - The cumulative distance along `self` to the projected point
    /// - The cumulative distance along `other` to the projected point
    /// - Angle between the found `Linessegments` where the points were projected on.
    ///
    /// Returns `None` if no such proximity is found.
    pub fn first_proximity(
        &self,
        other: &LineString,
        threshold: f64,
    ) -> Option<(Point, Point, f64, f64, f64)> {
        let mut dist1_total = self.pre_length;
        for seg1 in self.segments.iter() {
            let mut dist2_total = other.pre_length;
            let seg1_length = seg1.length;
            for seg2 in other.segments.iter() {
                let seg2_length = seg2.length;
                let (projected_point1, dist1) = seg1.project(&seg2.p2);
                let (projected_point2, dist2) = seg2.project(&seg1.p2);

                // Check if what projection is closer (seg1 <- seg2.point or seg2 <- seg1.point)
                let (min_dist, is_first) = if dist1 < dist2 {
                    (dist1, true)
                } else {
                    (dist2, false)
                };
                if min_dist < threshold {
                    let angle = seg2.r.y.atan2(seg2.r.x) - seg1.r.y.atan2(seg1.r.x);
                    return Some(if is_first {
                        (
                            projected_point1,
                            seg2.p2,
                            dist1_total + seg1.p1.distance(&projected_point1),
                            dist2_total + seg2_length,
                            angle,
                        )
                    } else {
                        (
                            seg1.p2,
                            projected_point2,
                            dist1_total + seg1_length,
                            dist2_total + seg2.p1.distance(&projected_point2),
                            -angle,
                        )
                    });
                }
                dist2_total += seg2_length;
            }
            dist1_total += seg1_length;
        }
        None
    }

    /// Finds the first point of proximity between this `LineString` and a given `Point`
    /// within a specified distance threshold.
    ///
    /// # Arguments
    ///
    /// * `point` - A reference to a `Point`.
    /// * `threshold` - A distance threshold.
    ///
    /// # Returns
    ///
    /// An `Option` containing a tuple with:
    /// - The projected point on `self`
    /// - The cumulative distance along `self` to the projected point
    /// - The angle (cf. x-axis) of the found segment of the `LineString`.
    ///
    /// Returns `None` if no such proximity is found.
    pub fn first_proximity_point(
        &self,
        point: &Point,
        threshold: f64,
    ) -> Option<(Point, f64, f64)> {
        let mut dist_total = self.pre_length;
        for seg in self.segments.iter() {
            let (projected_point, dist_proj) = seg.project(&point);

            if dist_proj < threshold {
                return Some((
                    projected_point,
                    dist_total + projected_point.distance(&seg.p1),
                    seg.r.y.atan2(seg.r.x),
                ));
            } else {
                dist_total += seg.length;
            }
        }
        None
    }
}

/// A struct representing a collection of `LineString` objects.
#[pyclass]
#[derive(Debug, Clone)]
pub struct LineStringFan {
    #[pyo3(get)]
    linestrings: Vec<LineString>,
    optimized: bool,
}

#[pymethods]
impl LineStringFan {
    /// Creates a new `LineStringFan` from a given vector of `LineString` objects.
    #[new]
    fn new(linestrings: Vec<LineString>) -> Self {
        LineStringFan {
            linestrings,
            optimized: false,
        }
    }

    /// Constructs a `LineStringFan` from a nested vector of floating point numbers,
    /// where each sublist represents x and y coordinate vectors.
    ///
    /// # Arguments
    /// * `points_lists` - A vector of sublists, where each sublist contains two vectors
    ///   representing x and y coordinates.
    ///
    /// # Returns
    /// A `PyResult<LineStringFan>` instance.
    ///
    /// # Errors
    /// Returns an error if the sublists do not contain exactly two vectors or if the x and y
    /// vectors are of different lengths.
    #[staticmethod]
    pub fn from_vectors(points_lists: Vec<Vec<Vec<f64>>>) -> PyResult<LineStringFan> {
        let mut linestrings = Vec::new();
        for points_list in points_lists {
            if points_list.len() != 2 {
                return Err(PyValueError::new_err(
                    "Each sublist must contain two vectors (x and y coordinates).",
                ));
            }

            let xs = &points_list[0];
            let ys = &points_list[1];

            if xs.len() != ys.len() {
                return Err(PyValueError::new_err(
                    "x and y vectors must have the same length.",
                ));
            }

            let points = points_from_arrays(xs, ys);
            let segments = line_segments_from_points(&points);
            linestrings.push(LineString {
                segments,
                pre_length: 0.0,
            });
        }
        Ok(LineStringFan {
            linestrings,
            optimized: false,
        })
    }

    /// Returns the points of all `LineString` objects as tuples of x and y coordinate vectors.
    fn points_as_xy_vectors(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        self.linestrings
            .iter()
            .map(|linestring| linestring.points_as_xy_vector())
            .collect()
    }

    /// Projects a given point onto the first `LineString` and shortens all `LineString` objects accordingly.
    /// This will result in to shorten `LineString` that start with the given ´point´.
    ///
    /// # Arguments
    /// * `point` - A `Point` to project and use for cutting.
    pub fn project_and_cut(&mut self, point: Point) {
        if self.linestrings.is_empty() || self.linestrings[0].segments.is_empty() {
            return;
        }
        if self.optimized {
            // TODO if already optimized pre_length must be shortend.
            todo!("This is not implemented yet. This will result in wrong lengths in later calculations");
        }
        let mut cut_index = 0;
        let mut proj_point = point;
        for (i, segment) in self.linestrings[0].segments.iter().enumerate() {
            let projected = segment.project(&point).0;
            if projected != segment.p2 {
                cut_index = i;
                proj_point = projected;
                break;
            }
        }
        // if all linestrings same from the first points it is assumed that the first x segemnts are also the same
        if let Some(first_point) = self.linestrings[0].segments.first().map(|s| s.p1) {
            if self
                .linestrings
                .iter()
                .all(|ls| ls.segments.first().map_or(false, |s| s.p1 == first_point))
            {
                for linestring in &mut self.linestrings {
                    if cut_index < linestring.segments.len() {
                        linestring.segments.drain(..cut_index);
                        linestring.segments[0].p1 = proj_point;
                    }
                }
            }
        }
    }

    /// Checks for intersections between this `LineStringFan` and another `LineStringFan`.
    ///
    /// # Arguments
    /// * `other` - Another `LineStringFan` to compare against.
    /// * `sorted` - Optional boolean indicating if the results should be sorted.
    ///
    /// # Returns
    /// A vector of tuples containing intersection points and their distances along each `LineString`.
    #[pyo3(signature = (other, sorted=false))]
    fn compare_intersection(
        &self,
        other: &LineStringFan,
        sorted: Option<bool>,
    ) -> Vec<(Point, f64, f64)> {
        let mut intersections = Vec::new();

        for line_self in &self.linestrings {
            for line_other in &other.linestrings {
                let inter = line_self.intersection(line_other);
                if !inter.is_none() {
                    intersections.extend(inter.unwrap());
                }
            }
        }
        match sorted {
            Some(true) => {
                intersections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let mut intersections_nd = intersections.clone();
                // sort intersections according to the second linestring and append it to the first the return
                intersections_nd.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                intersections.extend(intersections_nd);
                return intersections;
            }
            Some(false) | None => {
                return intersections;
            }
        }
    }

    /// Checks for the first proximity between the `LineStringFan` and another `LineStringFan`.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another `LineStringFan`.
    /// * `threshold` - A floating-point threshold value for proximity filtering.
    /// * `sorted` - An optional boolean indicating whether the results should be sorted by distance.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing:
    /// - The projected point on `self`
    /// - The projected point on `other`
    /// - The cumulative distance along `self` to the projected point
    /// - The cumulative distance along `other` to the projected point
    /// - Angle between the `Linessegments` where the points were projected on.
    ///
    #[pyo3(signature = (other, threshold, sorted=false))]
    pub fn compare_first_proximity(
        &self,
        other: &LineStringFan,
        threshold: f64,
        sorted: Option<bool>,
    ) -> Vec<(Point, Point, f64, f64, f64)> {
        let mut min_distances = Vec::new();

        for line_self in &self.linestrings {
            for line_other in &other.linestrings {
                let first_proximity_info = line_self.first_proximity(line_other, threshold);
                if !first_proximity_info.is_none() {
                    min_distances.extend(first_proximity_info);
                }
            }
        }
        match sorted {
            Some(true) => {
                min_distances.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                return min_distances;
            }
            Some(false) | None => {
                return min_distances;
            }
        }
    }

    /// Finds the closest point on any `LineString` in the `LineStringFan` to a given point,
    /// provided it is within the given threshold distance.
    ///
    /// # Arguments
    ///
    /// * `point` - A reference to the point to compare.
    /// * `threshold` - A floating-point threshold value for proximity filtering.
    ///
    /// # Returns
    ///
    /// An `Option` containing a tuple:
    /// - The projected point on the closest `LineString`.
    /// - The distance between the input point and the projected point.
    /// - The angle (cf. x-axis) of the first segment of the closest `LineString`.
    ///
    /// Returns `None` if no point is within the threshold.
    pub fn compare_first_proximity_point(
        &self,
        point: &Point,
        threshold: f64,
    ) -> Option<(Point, f64, f64)> {
        let mut min_distance = f64::INFINITY;
        let mut proj_point = *point;
        let mut current_angle = 0.0;

        for ls in &self.linestrings {
            if let Some((candidate_point, distance, angle)) =
                ls.first_proximity_point(point, threshold)
            {
                if distance < min_distance {
                    min_distance = distance;
                    proj_point = candidate_point;
                    current_angle = angle;
                }
            }
        }
        if min_distance == f64::INFINITY {
            return None;
        }
        Some((proj_point, min_distance, current_angle))
    }

    /// Optimizes the `LineString` objects by reducing redundant points.
    /// Overlapping elements of multiple `LineString` are removed. Unique intervals of `Point`s are grouped into new `LineString`.
    ///
    /// # Arguments
    /// * `threshold` - Optional threshold for point merging.
    ///
    /// # Returns
    /// A vector of optimized and unique `LineString` objects.
    #[pyo3(signature = (threshold=0.05))]
    pub fn optimize_linestrings(&mut self, threshold: Option<f64>) -> Vec<LineString> {
        if self.optimized {
            println!("The Linestrings are already optimized - returned them unchanged.");
            return self.linestrings.clone();
        }
        self.optimized = true;
        let mut meta_linestring_list: Vec<LineString> = Vec::new();
        let result = recursion_fn(
            0,
            &self.linestrings,
            &mut meta_linestring_list,
            0.0,
            threshold,
        );
        meta_linestring_list.push(result);
        self.linestrings = meta_linestring_list.clone();
        meta_linestring_list
    }
}

/// Helper functions

fn group_close_points(points: &[Point], threshold: f64) -> Vec<Vec<usize>> {
    let mut grouped_indices = Vec::new();
    let mut used = vec![false; points.len()];

    for i in 0..points.len() {
        if used[i] {
            continue;
        }
        let mut cluster = vec![i];
        used[i] = true;

        for j in (i + 1)..points.len() {
            if used[j] {
                continue;
            }

            if points[i].almost_equal(&points[j], threshold) {
                cluster.push(j);
                used[j] = true;
            }
        }
        grouped_indices.push(cluster);
    }
    grouped_indices
}

fn recursion_fn(
    start_idx: usize,
    linestring_pool: &[LineString],
    meta_linestring_list: &mut Vec<LineString>,
    current_length: f64,
    threshold: Option<f64>,
) -> LineString {
    let max_length = linestring_pool
        .iter()
        .map(|x| x.segments.len())
        .max()
        .unwrap_or(0);
    let mut new_current_length = current_length;
    let mut segment_list = Vec::with_capacity(max_length - start_idx);

    for i in start_idx..max_length {
        let points_at_cur_idx: Vec<Point> = linestring_pool
            .iter()
            .filter(|ls| ls.segments.len() > i)
            .map(|ls| ls.segments[i].p2.clone())
            .collect();

        let point_clusters = group_close_points(&points_at_cur_idx, threshold.unwrap_or(0.05));

        if point_clusters.len() == 1 {
            let segment = &linestring_pool[0].segments[i];
            new_current_length += segment.length;
            segment_list.push(segment.clone());
        } else {
            for linestring_idx_group in point_clusters {
                let new_linestring_pool: Vec<LineString> = linestring_idx_group
                    .iter()
                    .map(|&idx| linestring_pool[idx].clone())
                    .collect();

                let result = recursion_fn(
                    i,
                    &new_linestring_pool,
                    meta_linestring_list,
                    new_current_length,
                    threshold,
                );
                meta_linestring_list.push(result);
            }
            break;
        }
    }

    LineString::from_segments(segment_list, Some(current_length))
}

fn points_from_arrays(xs: &[f64], ys: &[f64]) -> Vec<Point> {
    xs.iter()
        .zip(ys.iter())
        .map(|(&x, &y)| Point { x, y })
        .collect()
}

fn line_segments_from_points(points: &[Point]) -> Vec<LineSegment> {
    points
        .windows(2)
        .map(|w| LineSegment::new(w[0], w[1]))
        .collect()
}
