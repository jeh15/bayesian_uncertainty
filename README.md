# Bayesian Uncertainty Quantification:
A repository to store homework / projects for the Bayesian Uncertainty Quantification class.

# How to run:
You need python3 to run the code contained in this repository. 
(Tested with python3.11 but will probably work with previous python3 versions as well)

Clone the repository and navigate to it,
```
git clone git@github.com:jeh15/bayesian_uncertianty.git
cd bayesian_uncertainty
```

Download dependencies (it is recommended to use a virtual enviornment),
```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install pymcmcstat
pip install numpy
pip install scipy
pip install matplotlib
pip install drake
pip install osqp
pip install --upgrade "jax[cpu]"
```