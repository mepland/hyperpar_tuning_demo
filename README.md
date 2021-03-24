# Hyperparameter Tuning Demo
Matthew Epland, PhD  
[Komodo Health](https://www.komodohealth.com/)  
[phy.duke.edu/~mbe9](http://www.phy.duke.edu/~mbe9)  

## Summary
A demonstration of hyperparameter tuning methods for XGBoost BDTs
on [UCI's Polish companies bankruptcy data set](http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data).
[Scikit-learn](https://scikit-learn.org/stable/) was used to run random (`RandomizedSearchCV`),
and grid searches (`GridSearchCV`) over the following hyperparameters:
`max_depth`, `learning_rate`, `gamma`, `reg_alpha`, `reg_lambda`.
Bayesian optimization was performed with [Scikit-optimize](https://scikit-optimize.github.io/) using
Gaussian process, random forest, and gradient boosted tree surrogate functions.
Lastly, [Hyperopt's](https://hyperopt.github.io/hyperopt/) Tree-Structured Parzen Estimator (TPE) and
[gentun's](https://github.com/gmontamat/gentun) genetic algorithm optimizers were also included.

Note that additional work running gentun in parallel is required to achieve an equivalent performance to the other methods. TODO

The results are provided in `output/` and collected into slides in [TODO](TODO).

TODO results
The toy classification problem on this dataset TODO



For a further discussion of the theoretical underpinnings of
Bayesian optimization and other methods see my notes on data science [here](https://github.com/mepland/data_science_notes).

## Cloning the Repository
ssh  
```bash
git clone --recurse-submodules git@github.com:mepland/hyperpar_tuning_demo.git
```

https  
```bash
git clone --recurse-submodules https://github.com/mepland/hyperpar_tuning_demo.git
```

## Installing Dependencies
It is recommended to work in a [python virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to avoid clashes with other installed software. If you do not wish to use a virtual environment you can just run the first few cells of the notebook - useful when running on cloud-based virtual machines.
```bash
python -m venv ~/.venvs/newenv
source ~/.venvs/newenv/bin/activate
pip install -r requirements.txt
cd gentun
pip install -r requirements.txt
python setup.py install
```

## Running the Notebook

```bash
jupyter lab hyperpar_tuning_demo.ipynb
```
