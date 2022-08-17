Contributing
==============


Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

# Types of Contributions

## Report Bugs

Report bugs at https://github.com/samiemostafavi/pr3d/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

## Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it. Those that are
tagged with "first-timers-only" is suitable for those getting started in open-source software.

## Write Documentation

`PR3D` could always use more documentation, whether as part of the
official docs, in docstrings, or even on the web in blog posts,
articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue at https://github.com/samiemostafavi/pr3d/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

# Get Started!

Ready to contribute? Here's how to set up `pr3d` for local development.

1. Fork the `pr3d` repo on GitHub.
2. Clone your fork locally:

        $ git clone git@github.com:your_name_here/pr3d.git

3. Install your local copy into a virtualenv. Assuming you have virtualenv installed, this is how you set up your fork for local development:

        $ cd pr3d/
        $ python -m virtualenv --python=python3.9.0 ./venv
        $ pip install -e ".[dev]"
        $ pip wheel . -w wheels

4. Create a branch for local development:

        $ git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

5. When you're done making changes, check that your changes pass *black*, *codespell*, *flake8*, and *isort* checks. To do that, add pre-commit hooks:

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

Flake8 with proper args could be set for VSCode if you use `.vscode` settings.

6. Commit your changes and push your branch to GitHub:

        $ git add .
        $ git commit -m "Your detailed description of your changes."
        $ git push origin name-of-your-bugfix-or-feature

   In brief, commit messages should follow these conventions:
       
   * Always contain a subject line which briefly describes the changes made. For example "Update CONTRIBUTING.rst".
   * Subject lines should not exceed 50 characters.
   * The commit body should contain context about the change - how the code worked before, how it works now and why you decided to solve the issue in the way you did.

   More detail on commit guidelines can be found at https://chris.beams.io/posts/git-commit

7. Submit a pull request through the GitHub website.

# Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for 3.9 and above. Check
   https://travis-ci.com/samiemostafavi/pr3d/pull_requests
   and make sure that the tests pass for all supported Python versions.


# Enable GPU Tensorflow 2.8.0 on Ubuntu 18.04

According to [here](https://www.tensorflow.org/install/source#gpu), we need `GCC 7.3.1`, `CUDA 11.2`, and `cuDNN 8.1` packages to enable GPU processing.
Verify the system has a cuda-capable gpu, download and install the nvidia cuda toolkit and cudnn, setup environmental variables and verify the installation.

1. If you have previous installation remove it first.

        $ sudo apt-get purge nvidia*
        $ sudo apt remove nvidia-*
        $ sudo rm /etc/apt/sources.list.d/cuda*
        $ sudo apt-get autoremove && sudo apt-get autoclean
        $ sudo rm -rf /usr/local/cuda*

2. System update

        $ sudo apt-get update
        $ sudo apt-get upgrade


3. Install dependencies

        $ sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev


4. Get the PPA repository driver

        $ sudo add-apt-repository ppa:graphics-drivers/ppa
        $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
        $ echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
        $ sudo apt-get update

5. Install CUDA-11.2

        $ sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-2 cuda-drivers


6. Setup your paths

        $ echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
        $ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        $ source ~/.bashrc
        $ sudo ldconfig

7. Install cuDNN v8.1: to do that, you must create an account in https://developer.nvidia.com/rdp/cudnn-archive and download `cudnn-11.2-linux-x64-v8.1.1.33.tgz`. Unzip it 

        $ tar -xzvf ~/Downloads/cudnn-11.2-linux-x64-v8.1.1.33.tgz

8. Copy the following files into the cuda toolkit directory.

        $ sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.2/include
        $ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64/
        $ sudo chmod a+r /usr/local/cuda-11.2/lib64/libcudnn*

9. Finally, to verify the installation, check

        $ nvidia-smi
        $ nvcc -V

10. Install Tensorflow and verify GPU capabalities

        $ pip install tensorflow==2.8.0
        $ pip install protobuf==3.20.*
        $ python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
        $ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"