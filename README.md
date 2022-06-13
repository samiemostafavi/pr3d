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


# Use Tensorflow with GPU and Ubuntu 18.04

According to [here](https://www.tensorflow.org/install/source#gpu), `GCC 7.3.1`, `CUDA 11.2`, and `cuDNN 8.1` is needed.
Verify the system has a cuda-capable gpu, download and install the nvidia cuda toolkit and cudnn, setup environmental variables and verify the installation.

If you have previous installation remove it first.

    sudo apt-get purge nvidia*
    sudo apt remove nvidia-*
    sudo rm /etc/apt/sources.list.d/cuda*
    sudo apt-get autoremove && sudo apt-get autoclean
    sudo rm -rf /usr/local/cuda*

System update

    sudo apt-get update
    sudo apt-get upgrade


Install other import packages

    sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev


First get the PPA repository driver

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
    sudo apt-get update

Install CUDA-11.2

    sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-2 cuda-drivers


Setup your paths

    echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    sudo ldconfig

Install cuDNN v8.1: to do that, you must create an account in https://developer.nvidia.com/rdp/cudnn-archive and download `cudnn-11.2-linux-x64-v8.1.1.33.tgz`. Unzip it 

    tar -xzvf ~/Downloads/cudnn-11.2-linux-x64-v8.1.1.33.tgz


Copy the following files into the cuda toolkit directory.

    sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.2/include
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64/
    sudo chmod a+r /usr/local/cuda-11.2/lib64/libcudnn*

Finally, to verify the installation, check

    nvidia-smi
    nvcc -V

Install Tensorflow and verify GPU capabalities

    pip install tensorflow==2.8.0
    pip install protobuf==3.20.*
    python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"



# Helpful readings

[1] https://keras.io/examples/keras_recipes/bayesian_neural_networks/

[2] https://towardsdatascience.com/modeling-uncertainty-in-neural-networks-with-tensorflow-probability-a706c2274d12

[3] https://nnart.org/understanding-a-bayesian-neural-network-a-tutorial/

[4] https://towardsdatascience.com/bayesian-neural-networks-with-tensorflow-probability-fbce27d6ef6

[5] https://towardsdatascience.com/data-formats-for-training-in-tensorflow-parquet-petastorm-feather-and-more-e55179eeeb72

[6] https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseVariational

[7] https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf

[8] https://www.youtube.com/watch?v=VFEOskzhhbc




