# Scenario Fingerprint Tool

Analyses traffic scenarios by custom metrics to determine its criticality

## Paper

If you use the Scenario Fingerprint tool, please cite our paper.

_Fingerprint of a Traffic Scene: an Approach for a Generic and Independent Scene Assessment_<br>
Barbara Schütt, Maximilian Zipfl, J. Marius Zöllner, Eric Sax
[[PDF](https://arxiv.org/pdf/2211.13683.pdf)]
```
@inproceedings{schutt2022fingerprint,
  title={Fingerprint of a Traffic Scene: an Approach for a Generic and Independent Scene Assessment},
  author={Sch{\"u}tt, Barbara and Zipfl, Maximilian and Z{\"o}llner, J Marius and Sax, Eric},
  booktitle={2022 International Conference on Electrical, Computer, Communications and Mechatronics Engineering (ICECCME)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```


##  Requirements
The tool was tested with the following python versions:
* 3.8.10
* 3.8.16 
* 3.9.16

You also need to install the [csv_object_list_dataset_loader](https://github.com/fzi-forschungszentrum-informatik/CSV_Motion_Dataset_Loader)
and follow its installation steps. The csv-file format corresponds with the format
of the [INTERACTION](https://interaction-dataset.com/), [highD](https://www.ind-dataset.com/), or [TAF](https://github.com/fzi-forschungszentrum-informatik/test-area-autonomous-driving-dataset) dataset.

## Installation
The module can be installed with pip for usage in other packages.
Also the requirements have to be installed (Python version 3.8 or 3.9)
```bash
git clone $LINK_TO_REPO
cd scenario_criticality/
pip install -r requirements.txt
pip install . # install this module to pip if needed
```

## Usage

### Quick Test

```bash
cd src/
export PYTHONPATH=$PYONPANTH:$(pwd) # make sure python knows where the modules is, if it's not already installed
cd ..
python ./tests/test_metrics.py

```



### Implementation

1. Load a csv file
2. Initialize a metric object
3. Calculate metric


The timestamp has to be a timestamp from the dataset.

```python
from scenario_criticality.binary_metrics.post_encroachment_time.et import ET

dataset_loader = Loader()
dataset_loader.load_dataset("test_001.csv")
scenario = dataset_loader.return_scenario("test_001.csv") 

timestamp = 100

et = ET(scenario, timestamp)
et.calculate_metric()
print(et.results_matrix)
```

## Authors
Barbara Schütt (schuett@fzi.de)  
Maximilian Zipfl (zipfl@fzi.de)

## Dependencies

| Package/Library | License |
|---|---|
| numpy | [Numpy License (BSD 3-Clause)](https://numpy.org/doc/stable/license.html) |
| pandas | [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| ray | [Apache License Version 2.0](https://github.com/ray-project/ray/blob/master/LICENSE)|
| protobuf | [BSD 3-Clause](https://ptolemy.berkeley.edu/ptolemyII/ptII11.0/ptII/lib/protobuf-license.htm)|
| click | [BSD 3-Clause](https://click.palletsprojects.com/en/5.x/license/)|
| Shapely | [BSD 3-Clause](https://github.com/shapely/shapely/blob/main/LICENSE.txt)|
| plotly | [BSD 3-Clause](https://github.com/plotly/plotly.py/blob/master/LICENSE.txt)|
| pyproj | [BSD 3-Clause](https://github.com/pyproj4/pyproj/blob/main/LICENSE)|
| tqdm | [BSD 3-Clause](https://github.com/tqdm/tqdm/blob/master/LICENCE)|
| matplotlib| [Matplotlib-Lizenz](https://matplotlib.org/stable/users/project/license.html)
