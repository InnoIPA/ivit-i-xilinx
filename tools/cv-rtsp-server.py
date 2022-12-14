import cv2
import time
import threading
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer,GObject
# from human_detection.model import YOLOv4_human_detection

class WebcamVideoStream():
    def __init__(self, uri, latency):
        ############################################################################
        # Work
        # Original0
        gst_str = ('multifilesrc location=./data/car.mp4 loop=true ' + \
                    '! decodebin3 ' + \
                    '! videoconvert ' + \
                    '! appsink').format(uri, latency)
        # gst_str = ('multifilesrc location=./data/car.mp4 loop=true ' + \
        #             '! queue ! decodebin3 ' + \
        #             '! queue ! videoconvert ' + \
        #             '! appsink').format(uri, latency)
        # gst_str = ('multifilesrc location=./data/car.mp4 loop=true ' + \
        #             '! h264parse ! omxh264dec ! queue ' + \
        #             '! videoconvert ! queue ' + \
        #             '! appsink').format(uri, latency) 
        ############################################################################
        """        
        加了 ip-mode, op-mode 會無法讀到 frame
        gst_str = ('multifilesrc location=./data/car.h264 loop=true ' + \
                    '! h264parse ! omxh264dec ip-mode=1 op-mode=0 ' + \
                    '! videoconvert ' + \
                    '! appsink').format(uri, latency) 
        """
        # gst_str = ('multifilesrc location=./data/car.h264 loop=true ' + \
        #             '! videoparse format=nv12 width=1280 height=720 framerate=30/1 '
        #             '! qtdemux name=demux demux.video_0 ! h264parse ! queue max-size-bytes=0 ! omxh264dec ip-mode=1 op-mode=0 ! queue max-size-bytes=0 ' + \
        #             '! appsink').format(uri, latency) 
        # gst_str = ('v4l2src device=/dev/video0 ! ' + \
        #             'h264parse ! omxh264dec ! ' + \
        #             'videoconvert ! ' + \
        #             'appsink').format(uri, latency)
        ############################################################################
        # Run
        # self.stream = cv2.VideoCapture(
        #     gst_str, cv2.CAP_GSTREAMER
        # )
        self.stream = cv2.VideoCapture(
            './data/car.mp4'
        )

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.finish = False
    def start(self):
        self.update()
    def update(self):
        temp_frame = None
        while True:
            start_time=time.time()
            if self.stopped:
                self.stream.release()
                break
            (self.grabbed, self.frame) = self.stream.read()
            # print(temp_frame is self.frame)
            # self.frame = cv2.resize(self.frame, (1920,1080), interpolation=cv2.INTER_AREA)
            # cv2.imshow("img",self.frame)
            if cv2.waitKey=='q':
                self.stopped = True
            self.finish=True
            if((1/30)-(time.time()-start_time))>0:
                time.sleep(abs((1/30)-(time.time()-start_time)))
            else:              
                continue
            temp_frame = self.frame
    def read(self):
        return self.grabbed, self.frame.copy()
    def check(self):
        return self.finish    
    def stop(self):
        self.stopped = True
    def result(self):
        while True:
            start_time=time.time()
            if self.finish==True and self.analtysis.ready==True:
                ret,self.result_frame=self.read()
                classes,confidences,boxes=self.analtysis.read()
                self.result_frame=self.model.human_detection_outputs_frame(self.result_frame,classes,confidences,boxes)
                if((1/30)-(time.time()-start_time))>0:
                    time.sleep(abs((1/30)-(time.time()-start_time)))
                else:              
                    continue
    def set_model(self,analtysis,model):
        self.model=model
        self.analtysis=analtysis
    def get(self):
        return self.grabbed,self.result_frame

class webAnaltysis():
    def __init__(self,webcam,model):
        self.webcam=webcam
        self.model=model
        self.ready=False
    def start(self):
        while True:
            start_time=time.time()
            ret,frame=self.webcam.read()
            if ret:
                self.classes,self.confidences,self.boxes=self.model.detect(frame)
                self.ready=True 
            if(0.07-(time.time()-start_time))>0:
                time.sleep(abs(0.07-(time.time()-start_time)))
            else:              
                continue
    def read(self):
        return self.classes,self.confidences,self.boxes

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
    def on_need_data(self, src, lenght):
        ret,frame=self.webcam.read()  
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = int(timestamp)
        buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames += 1
        retval = src.emit('push-buffer', buf)
        if retval != Gst.FlowReturn.OK:
            print(retval)
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        rtsp_media.set_shared(True)
        self.appsrc = rtsp_media.get_element().get_child_by_name('source')
        self.appsrc.connect('need-data', self.on_need_data)
    def set_frame(self,webcam):
        self.webcam=webcam
        ret,frame=self.webcam.read()
        width=frame.shape[1]
        height=frame.shape[0]

        
        self.launch_string = 'videomixer name=mix sink_0::xpos=0 sink_0::ypos=0 ' \
                            '! omxh264enc prefetch-buffer=true ' \
                            'control-rate=2 target-bitrate=1000 ' \
                            'filler-data=false constrained-intra-prediction=true ' \
                            'periodicity-idr=120 gop-mode=low-delay-p aspect-ratio=3 ' \
                            'low-bandwidth=true default-roi-quality=4 ' \
                            '! video/x-h264,alignment=au ' \
                            '! h264parse ' \
                            '! rtph264pay name=pay0 pt=96 ' \
                            'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                            'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                            '! videoconvert ! mix.sink_0 '.format(width,height,self.fps )

class GstServer():
    def __init__(self,webcam):
        self.server=GstRtspServer.RTSPOnvifServer.new()
        # auth = GstRtspServer.RTSPAuth()
        # token = GstRtspServer.RTSPToken()
        # token.set_string('media.factory.role', "user")
        # basic = GstRtspServer.RTSPAuth.make_basic("admin", "wec")
        # auth.add_basic(basic, token)
        # self.server.set_auth(auth)
        self.port="8999"
        self.server.set_service(self.port)
        self.server.connect("client-connected",self.client_connected)
        self.server.attach(None)
        factory=SensorFactory()
        factory.set_frame(webcam)
        # permissions = GstRtspServer.RTSPPermissions()
        # permissions.add_permission_for_role("user", "media.factory.access", True)
        # permissions.add_permission_for_role("user", "media.factory.construct", True)
        # factory.set_permissions(permissions)
        factory.set_shared(True)
        self.server.get_mount_points().add_factory("/test", factory)
        #  start serving
        print ("stream ready at rtsp://127.0.0.1:" + self.port + "/test")
        #  this com IP address
        # print ("stream ready at rtsp://192.168.1.157:" + self.port+ "/test")
    def client_connected(self, arg1, arg2):      
        print('Client connected')

if __name__ == '__main__':
    GObject.threads_init()
    Gst.init(None)
    src=f''
    # model=YOLOv4_human_detection()
    webcam=WebcamVideoStream(src,0)
    # analtysis=webAnaltysis(webcam,model)
    # webcam.set_model(analtysis,model)
    webcam_thread=threading.Thread(target = webcam.start)
    webcam_thread.start()
    while True:
        if webcam.check()==True:
            break
        time.sleep(0.01)
    # analtysis_thread=threading.Thread(target = analtysis.start)
    # analtysis_thread.start()
    # result_thread=threading.Thread(target = webcam.result)
    # result_thread.start()
    # while True:
    #     if analtysis.ready==True:
    #         break
    #     time.sleep(0.01)
    RtspServer=GstServer(webcam)
    loop = GLib.MainLoop()
    loop.run()