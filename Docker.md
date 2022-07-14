
Start with building the Docker images:

        docker build --file Dockerfile.base -t samiemostafavi/pr3d:base .
        docker build --file Dockerfile.pr3d -t samiemostafavi/pr3d:pr3d .
        
        docker push samiemostafavi/pr3d:pr3d
        docker push samiemostafavi/pr3d:base

