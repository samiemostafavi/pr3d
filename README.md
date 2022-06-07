# Pr3D - PreDicting Delay probability Density

Implementation of two conditional density estimation methods with parametric neural networks in Python using Tensorflow

* Conventional and non-conditional mixture density network with Gaussian mixture model (GaussianMM)
* Novel conventional and non-conditional mixture density network with Gamma and extreme value mixture model (GammaEVM)

# Setting up development

1. Clone the repository, create a Python 3.9.0 virtual environment (requires [virtualenv](https://pypi.org/project/virtualenv/)), and activate it:

    ``` bash
    $ git clone git@github.com:samiemostafavi/pr3d.git
    
    $ cd pr3d

    $ python -m virtualenv --python=python3.9.0 ./venv

    $ source ./venv/bin/activate

    (venv) $ 
    ```
    
2. Install the required packages by `requirements.txt`: `pip install -Ur requirements.txt`.

# Using the package

    pip install git+https://github.com/samiemostafavi/pr3d.git
