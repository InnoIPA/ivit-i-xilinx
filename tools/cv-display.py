import cv2, time

def main_loop(cap, writer, expect_fps=60):

    count, fps = 0, expect_fps

    t_expect = (1/expect_fps)
    fps_pool = []
    
    print('Start Capture Loop ... \n')
        
    while(True):
        
        t_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret: break

        count = count + 1
        
        t_write = time.perf_counter()
        writer.write(frame)
        
        print(f"\rID: {count}, FPS: {fps:<5}, WRTIE RTSP: {round((time.perf_counter()-t_write),3):<5}", end='')

        t_cur = time.perf_counter() - t_start
        # if t_cur < t_expect:
        #     time.sleep(t_expect-t_cur)
        # time.sleep(1/59)

        fps_pool.append( int(1/(t_cur)) )
        if len(fps_pool)>=10:
            fps = sum(fps_pool)//len(fps_pool); fps_pool.clear()

print(
    """\n Press Ctrl+C to exit !!! \n"""
)

# cap = cv2.VideoCapture('/dev/video0')
cap = cv2.VideoCapture('./data/car.mp4', cv2.CAP_FFMPEG)
print('Initialized Video Capture')

ret, frame = cap.read()
(src_hei, src_wid), src_fps = frame.shape[:2], cap.get(cv2.CAP_PROP_FPS)

gst_pipeline = \
    'appsrc ' + \
    f'caps=video/x-raw,format=BGR,width={src_wid},height={src_hei},framerate={src_fps}/1 ' + \
    '! videoconvert ' + \
    '! kmssink bus-id=fd4a0000.zynqmp-display fullscreen-overlay=1 sync=false'


print('\n', gst_pipeline, '\n')

writer = cv2.VideoWriter(  gst_pipeline, cv2.CAP_GSTREAMER, 0, 
                        src_fps, (src_wid, src_hei), True )

# assert not writer.isOpened(), 'Could not open video writer ...'

try:
    main_loop(cap, writer)

except KeyboardInterrupt:
    pass

except Exception as e:
    print(e)

finally:
    print('\nQuit')
    cap.release()
