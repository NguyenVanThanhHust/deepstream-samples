# Video pipeline with Pytorch
Source: https://paulbridger.com/posts/video-analytics-pytorch-pipeline/

## Setup env
Build docker image 
```
docker build -t ds70_pt_img -f dockers/ds70_pt.Dockerfile ./dockers/
```

## How to run
### Start docker
Start docker container
```
docker run --rm -it --name deepstream_ctn --gpus '"device=0" --shm-size 8G --network host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e CUDA_CACHE_DISABLE=0 --env="QT_X11_NO_MITSHM=1" --volume="$PWD:/workspace/" -w /workspace/ ds70_pt_img:latest /bin/bash
```
### First check
Validate if the pipeline is working
```
gst-launch-1.0 filesrc location=videos/scenario_1/scenario1_cam1.mkv ! decodebin ! progressreport update-freq=1 ! fakesink sync=true
```
The result should look like
```
(gst-plugin-scanner:157): GStreamer-WARNING **: 03:11:38.981: Failed to load plugin '/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream/libnvdsgst_udp.so': librivermax.so.1: cannot open shared object file: No such file or directory
Fontconfig error: failed reading config file: /etc/fonts/conf.avail/53-monospace-lcd-filter.conf: Bad file descriptor (errno 9)
Setting pipeline to PAUSED ...
Pipeline is PREROLLING ...
Pipeline is PREROLLED ...
Setting pipeline to PLAYING ...
Redistribute latency...
New clock: GstSystemClock
progressreport0 (00:00:01): 0 / 103 seconds ( 0.0 %)
progressreport0 (00:00:02): 1 / 103 seconds ( 1.0 %)
...
progressreport0 (00:01:43): 102 / 103 seconds (99.0 %)
progressreport0 (00:01:43): 103 / 103 seconds (100.0 %)
Got EOS from element "pipeline0".
Execution ended after 0:01:43.287305015
Setting pipeline to NULL ...
Freeing pipeline ...
```

Or run without sync
```
gst-launch-1.0 filesrc location=videos/scenario_1/scenario1_cam1.mkv ! decodebin ! progressreport update-freq=1 ! fakesink sync=false
```
Output should be much faster
```
(gst-plugin-scanner:185): GStreamer-WARNING **: 03:14:33.598: Failed to load plugin '/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream/libnvdsgst_udp.so': librivermax.so.1: cannot open shared object file: No such file or directory
Setting pipeline to PAUSED ...
Pipeline is PREROLLING ...
progressreport0 (00:00:01): 0 / 103 seconds ( 0.0 %)
Pipeline is PREROLLED ...
Setting pipeline to PLAYING ...
Redistribute latency...
New clock: GstSystemClock
progressreport0 (00:00:02): 22 / 103 seconds (21.4 %)
progressreport0 (00:00:03): 45 / 103 seconds (43.7 %)
progressreport0 (00:00:04): 68 / 103 seconds (66.0 %)
progressreport0 (00:00:05): 91 / 103 seconds (88.3 %)
progressreport0 (00:00:05): 103 / 103 seconds (100.0 %)
Got EOS from element "pipeline0".
Execution ended after 0:00:04.485233407
Setting pipeline to NULL ...
Freeing pipeline ...
```

### How to run with pytorch
