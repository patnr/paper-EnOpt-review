# Python code scripts (supplementary materials) for paper

Prerequisite: Python 3.9

Other versions may also be compliant.
I recommend using a python virtual environment
before installing the requirements as follows:

`pip install -r path/to/requirements.txt`

To run the experiments, first create the directory `$HOME/data/robust-grad/` . Then do

`python path/to/robust-grad.py`

But you may want to first decrease the amount of experiments run by
reducing the ranges of parameters in the `coords` dict,
for example `seed = range(1 * 10**2)`.

Following completion of the script, you can print and plot the results by

`python path/to/robust-grad-results.py`
