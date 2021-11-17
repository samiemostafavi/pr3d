# conditional-latency-probability-prodiction

Implementation of two conditional density estimation methods with parametric neural networks in Python:

* Conventional Mixture Density Network with Gaussian Mixture Model (GMM)
* Novel Mixture Density Network with Gaussian and Extreme Value Mixture Model (EMM)

# Setting up development

1. Clone the repository, create a Python 3.6.0 virtual environment (requires [virtualenv](https://pypi.org/project/virtualenv/)), and activate it:

    ``` bash
    $ git clone git@github.com:samiemostafavi/conditional-latency-probability-prodiction.git
    
    $ cd conditional-latency-probability-prediction

    $ python -m virtualenv --python=python3.6.0 ./venv

    $ source ./venv/bin/activate

    (venv) $ 
    ```
    
2. Install the required packages by `requirements.txt`: `pip install -Ur requirements.txt`.

This implementation is based on the repository [here](https://github.com/freelunchtheorem/Conditional_Density_Estimation).
