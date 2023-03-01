from setuptools import setup, find_packages

setup(
    name='scenario_criticality',
    version='1.0.0',
    description='Analyses scenarios by custom metrics.',
    author='Barbara Schütt, Max Zipfl',
    author_email='schuett@fzi.de, zipfl@fzi.de',
    install_requires=['pandas', 'numpy'],
    package_dir={"": "src"},
    packages=find_packages(where="src") +
             find_packages(where="./binary_metrics") +
             find_packages(where="./binary_metrics/distance") +
             find_packages(where="./binary_metrics/gap_time") +
             find_packages(where="./binary_metrics/post_encroachment_time") +
             find_packages(where="./binary_metrics/potential_time_to_collision") +
             find_packages(where="./binary_metrics/time_to_collision") +
             find_packages(where="./binary_metrics/trajectory_distance") +
             find_packages(where="./binary_metrics/worst_time_to_collision") +
             find_packages(where="./safety_potential") +
             find_packages(where="./traffic_quality"),
    python_requires=">=3.8",
    zip_safe=False
)
#
