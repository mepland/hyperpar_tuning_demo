# Hyperparameter Tuning Demo
Matthew Epland, PhD  
[Komodo Health](https://www.komodohealth.com/)  
[phy.duke.edu/~mbe9](http://www.phy.duke.edu/~mbe9)  

A demonstration of hyperparameter tuning with XGBoost and Scikit Learn.  

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
It is recommended to work in a [python virtual environment](https://realpython.com/python-virtual-environments-a-primer/) to avoid clashes with other installed software.
```bash
python -m venv ~/.venvs/newenv
source ~/.venvs/newenv/bin/activate
pip install -r requirements.txt
cd gentun
python setup.py install
```

## Running the Notebook

```bash
jupyter lab hyperpar_tuning_demo.ipynb
```
