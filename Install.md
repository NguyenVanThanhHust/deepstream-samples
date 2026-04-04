# INSTALL
Currently, this repo is tested with deepstream 7.0 Dockerfile. 

For laptop, I'm using CUDA runtime 12.4, CUDA driver 12.6 as host.   
For Jetson, I'm using Jetpack 6.1 and everything packed with it. 
## Build docker image
For deepstream python on laptop/server.
Deepstream 7.0 is supported on Jetson.
```
docker build -t ds70_img -f dockers/ds70.Dockerfile .
```
Newest version is 9.0
```
docker build -t ds90_img -f dockers/ds90.Dockerfile .
```


## Build docker container
```
docker run --rm -it --name deepstream_ctn --gpus=all --shm-size 8G --network host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e CUDA_CACHE_DISABLE=0 --env="QT_X11_NO_MITSHM=1" --volume="$PWD:/workspace/" -w /workspace/ ds70_img:latest /bin/bash
```
Open other terminal, expose to show output video. Warning, from what I read, this isn't secure. But I don't know other way, so do it at your own risk. 
```
docker ps | grep "deepstream_ctn" | awk '{ print $1 }' | xargs -I {} sh -c "xhost +local:{}"
```

Or run app
```
docker run --rm --name deepstream_ctn --gpus=all --shm-size 8G --network host  --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="$PWD:/workspace/" ds_71_py:latest python3 your_application.py your_arguments
```

There are 2 dockers here: one to convert model from pytorch to TensorRT, other to run your deepstream app.

Build docker image to convert model from pytorch
```
docker build -t dev_img -f docker/pytorch_dev.Dockerfile .
```

## Reference
Install update-rtpmanager-sh-in-deepstream-7-0: 
https://forums.developer.nvidia.com/t/can-t-install-update-rtpmanager-sh-in-deepstream-7-0-docker/297573/4