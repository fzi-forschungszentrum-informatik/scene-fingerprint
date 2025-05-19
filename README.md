# OnCrit

This repository provides a package for the **On**line calculation of **Crit**icality measures on object lists for the analysis of road data. Criticality values are (currently) calculated exclusively on a scene, i.e. based on the current information. No retrospective calculations are carried out.
The package has been developed in a hybrid manner, utilising Rust code for intensive computing tasks and employing Python bindings. It also offers helpers for the operation with given Python libraries, such as [Lanelet](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) for road map logic.

The input data format is based on extant data formats such as [inD](https://levelxdata.com/ind-dataset/), [INTERACTION](https://interaction-dataset.com/) and [TAF-BW](https://github.com/fzi-forschungszentrum-informatik/test-area-autonomous-driving-dataset) data. However, it should be noted that all data formats which provide the poses and speeds of traffic participants and the associated map information (currently only in lanelet format) can be used. 

The fundamental principle of OnCrit is the calculation of so-called '_future paths_', which represent all continuing paths along the current carriageway for each individual vehicle within a scene. This process enables a scene in Cartesian space to be mapped into a kind of Frenet space, thereby facilitating the implementation of many of the classic criticality metrics.
![futurepaths](./docs/scenario_comp_lq.gif)

The calculation of the metrics does not provide for an explicit ego vehicle; rather, it calculates the respective criticality values between all active traffic participants in the scene, if possible.

## Getting Started

_A virtual environment is recommended._

```bash
# Install dependencies
pip install .
pip install jupyter ipykernel wheel  # optional
```

Each basic workflow of the repository is explained step-by-step by jupyter notebooks.

* [Quickstart notebook](docs/tutorial_quick_start.ipynb)
* [Future paths notebook](docs/tutorial_future_paths_lanelet.ipynb)
* [Test your data notebook](docs/tutorial_test_your_data.ipynb)
* [s-t-metrics notebook](TODO)


## Criticality Metrics

Currently the following criticality metrics are implemented:

* EuclideanDistanceSimple
* EuclideanDistanceApproximated
* TTC
* Headway
* Gaptime
* Clearance
* Spacing
* PPET

See [Metric Explanation](docs/metrics_explanation.md) for more detail.

## DependeNcies

* Rust
* Python
* [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)

Optional for Tutorials:
* [Jupyter](https://docs.jupyter.org/en/stable/install.html)

## Building Rust

This package uses [PyO3 and Maturin](https://github.com/PyO3/maturin) to create and build python/rust bindings.

run following lines in the package dict to compile rust bindings
```
maturin develop
```
or for release
```
maturin develop --release
```

## Testing the Package

```
python -m unittest tests/utils_test.py 
```

## Upcoming Roadmap
- [ ] Explanation of the metrics [here](docs/metrics_explanation.md)
- [ ] Tooling for loading/creating own maps
- [ ] Helpers for nuscene data  
- [ ] move visualize lanelet in viz tools
- [ ] add the check for projection at (find_closest_proximity)


