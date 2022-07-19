
# Step 1: Make sure [Docker](https://docs.docker.com/engine/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) are installed on the host

Verify that there is an nvidia GPU

        lspci | grep -i nvidia

Verify running `nvidia-smi`

Check if the host supports `nvidia-docker`

        docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

or

        docker run --gpus all --rm nvidia/cuda nvidia-smi

if no, install as mentioned [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).


# Step 2: Choose nvidia/cuda image

Choose an nvidia cuda image by checking the supported tags [here](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md)

        https://hub.docker.com/r/nvidia/cuda

For example for the base Docker image we chose `11.6.2-cudnn8-devel-ubuntu18.04`.
Imoprtant: 
1. `devel` images contain cuDNN by default.
2. Cuda version is important. `11.7` did not work and I switched to `11.6.2` to fix it.

# Step 3: Build Docker images

Start with building the Docker images:
        
        cd pr3d
        docker build --file docker/Dockerfile.base -t samiemostafavi/pr3d:base .
        docker build --file docker/Dockerfile.pr3d -t samiemostafavi/pr3d:pr3d .
        
        docker push samiemostafavi/pr3d:base
        docker push samiemostafavi/pr3d:pr3d

Run containers:

        docker run -it --rm --gpus all --entrypoint /bin/bash samiemostafavi/pr3d:base
        docker run -it --rm --gpus all --entrypoint /bin/bash samiemostafavi/pr3d:pr3d

# Step 4: Pull and Run Images

