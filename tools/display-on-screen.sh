gst-launch-1.0 -v \
v4l2src device=/dev/video0 \
! video/x-raw, width=1920, height=1080 \
! videoconvert \
! kmssink bus-id=fd4a0000.zynqmp-display fullscreen-overlay=1 sync=false
