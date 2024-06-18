# firebench

FireBench is the largest high-fidelity wildfire simulation dataset, enabling the generation of ensembles of fire evolution scenarios and combining observational environmental data with high-fidelity simulations. From development of fire propagation models, to investigations of fire and atmospheric dynamics, or even construction of novel machine learning models related to turbulent multiphase fluid flows, the applications of FireBench are exciting! 

In our initial GitHub repo, there are two sample scripts that demonstrate how to access and process the Firebench dataset.  The first is a notebook that presents a simple UI to browse the Firebench dataset. The script in the notebook scans the simulations in the public Firebench dataset to populate its UI dropdowns. Additionally, it can plot a simple slice of the data given the selection's in the UI. The second script is an Apache Beam pipeline that demonstrates how to postprocess the Firebench dataset. Each simulation is several TBs of data, so we use Apache Beam to run post-processing code in parallel on multiple machines. This sample script computes a time series of mean, minimum and maximum variables of all variables for a given simulation. 

[![Unittests](https://github.com/google-research/firebench/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/firebench/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/firebench.svg)](https://badge.fury.io/py/firebench)

*This is not an officially supported Google product.*
