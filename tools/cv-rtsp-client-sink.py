import cv2, time, threading

class CamThread():

    def __init__(self, input, fps=60):
        self.input = input
        self.cap = cv2.VideoCapture(self.input)
        self.fps = fps
        self.is_stop = False
        self.frame = None

        # Start thread
        self.t = threading.Thread(
            target = self.loop, daemon=True
        )

        self.t.start()
        time.sleep(1)

    def dynamic_sleep(self, start_time):
        if(1/self.fps-(time.time()-start_time))>0:
            time.sleep(abs(1/self.fps-(time.time()-start_time)))

    def loop(self):

        while( not self.is_stop):
            
            start_time = time.time()

            ret, frame = self.cap.read()

            if not ret: 
                self.cap = cv2.VideoCapture(self.input)
                time.sleep(1); continue

            self.frame = frame

            self.dynamic_sleep(start_time)

        print('Out of thread ...')

    def read(self):
        return True, self.frame
    
    def release(self):
        self.is_stop = True
        self.t.join()
        self.cap.release()
        
    def __del__(self):
        self.release()

def main_loop(cap, writer, expect_fps):

    count, fps = 0, expect_fps
    
    print('Start Capture Loop ... \n')
        
    while(True):
        
        start_time = time.time()

        ret, frame = cap.read()

        if not ret: break

        count = count + 1
        
        t_write = time.time()
        writer.write(frame)
        
        print(f"\rID: {count}, FPS: {fps:<5}, WRTIE RTSP: {round((time.time()-t_write),3):<5}", end='')

        if(1/expect_fps-(time.time()-start_time))>0:
            time.sleep(abs(1/expect_fps-(time.time()-start_time)))
        else:              
            continue

        t_cur = time.time() - start_time
        fps = int(1/t_cur)

print(
    """\n Press Ctrl+C to exit !!! \n"""
)

# cap = cv2.VideoCapture('/dev/video0')

INTPUT = './data/car.mp4'
FPS = 30

cap = CamThread(input=INTPUT, fps=30)

ret, frame = cap.read()
(src_hei, src_wid), src_fps = frame.shape[:2], FPS

gst_pipeline = \
    'videomixer name=mix sink_0::xpos=0 sink_0::ypos=0 ' + \
    '! omxh264enc prefetch-buffer=true ' + \
    'control-rate=2 target-bitrate=1000 ' + \
    'filler-data=false constrained-intra-prediction=true ' + \
    'periodicity-idr=120 gop-mode=low-delay-p aspect-ratio=3 ' + \
    'low-bandwidth=true default-roi-quality=4 ' + \
    '! video/x-h264,alignment=au ' + \
    '! rtspclientsink location=rtsp://localhost:8554/test ' + \
    'appsrc is-live=true block=true format=GST_FORMAT_TIME ' + \
    f'caps=video/x-raw,format=BGR,width={src_wid},height={src_hei},framerate={src_fps}/1 ' + \
    '! videoconvert ! mix.sink_0'

print('\n', gst_pipeline, '\n')

writer = cv2.VideoWriter(  gst_pipeline, cv2.CAP_GSTREAMER, 0, 
                        src_fps, (src_wid, src_hei), True )


# assert not writer.isOpened(), 'Could not open video writer ...'

try:
    main_loop(cap, writer, src_fps)

except KeyboardInterrupt:
    pass

except Exception as e:
    print(e)

finally:
    print('\nQuit')
    writer.release()
    cap.release()
