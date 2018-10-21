sudo nvidia-docker run \
    -v /home/kidzik/workspace/DensePose/DensePoseData/UV_data:/densepose/DensePoseData/UV_data \
    -v /home/kidzik/workspace/DensePose/DensePoseData/models:/densepose/DensePoseData/models \
    -v /media/kidzik/SECURED/clinical-videos/data/frames:/densepose/frames \
    -v /media/kidzik/SECURED/clinical-videos/data/densepose-out:/densepose/out \
    -v /home/kidzik/workspace/DensePose/scripts:/densepose/scripts \
    -it densepose:c2-cuda9-cudnn7 \
    bash scripts/mytest.sh
