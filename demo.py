import os, sys, cv2, logging, argparse, time, copy

# ivit
sys.path.append(os.getcwd())
from ivit_i.common import api
from ivit_i.common.pipeline import Pipeline

from ivit_i.utils.parser import load_json
from ivit_i.utils.draw_tools import draw_fps
from ivit_i.utils.err_handler import handle_exception
from ivit_i.app.handler import get_application

FULL_SCREEN = True
WAIT_KEY_TIME   = 1
SERV    = 'server'
RTSP    = 'rtsp'
GUI     = 'gui'

def init_cv_win():
    logging.info('Init Display Window')
    cv2.namedWindow( CV_WIN, cv2.WND_PROP_FULLSCREEN )
    # cv2.setWindowProperty( CV_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN )

def fullscreen_toggle():
    global FULL_SCREEN
    cv2.setWindowProperty( 
        CV_WIN, cv2.WND_PROP_FULLSCREEN, 
        cv2.WINDOW_FULLSCREEN if FULL_SCREEN else cv2.WINDOW_NORMAL )
    FULL_SCREEN = not FULL_SCREEN

def display(frame, t_wait_key):

    exit_flag = False

    cv2.imshow(CV_WIN, frame)            
    
    key = cv2.waitKey(t_wait_key)
    if key in {ord('q'), ord('Q'), 27}:
        exit_flag = True
    elif key in { ord('a'), 201, 206, 210, 214 }:
        fullscreen_toggle()

    return exit_flag

def get_running_mode(args):
    if(args.server): return SERV
    elif(args.rtsp): return RTSP
    else: return GUI

def check_info(info):
    if info is None: return False
    if info['detections']==[]: return False
    
    return True


def main(args):

    # Get Mode
    mode = get_running_mode(args)
    t_wait_key  = 0 if args.debug else WAIT_KEY_TIME

    # Load and combine configuration
    app_conf = load_json(args.config)                       
    model_conf = load_json(app_conf['prim']['model_json']) 
    total_conf = model_conf.copy()
    total_conf.update(app_conf)

    # Get the target API and load model
    trg = api.get(total_conf)
    
    # Load Model
    trg.load_model(total_conf, int(args.mode))
    
    # # Setup Async Mode
    if args.mode == 1: trg.set_async_mode()
        
    # Get source object
    src = Pipeline(total_conf['source'], total_conf['source_type'])
    src.start()
    (src_hei, src_wid), src_fps = src.get_shape(), src.get_fps()

    # Concate RTSP pipeline
    if mode==RTSP:

        gst_pipeline = \
            'videomixer name=mix sink_0::xpos=0 sink_0::ypos=0 ! omxh264enc prefetch-buffer=true ' + \
            'control-rate=2 target-bitrate=3000 filler-data=false constrained-intra-prediction=true ' + \
            'periodicity-idr=120 gop-mode=low-delay-p aspect-ratio=3 low-bandwidth=true default-roi-quality=4 ' + \
            '! video/x-h264,alignment=au ' + \
            f'! rtspclientsink location=rtsp://{args.ip}:{args.port}{args.name} ' + \
            'appsrc ' + \
            f'caps=video/x-raw,format=BGR,width={src_wid},height={src_hei},framerate={src_fps}/1 ' + \
            '! videoconvert ! mix.sink_0'

        out = cv2.VideoWriter(  gst_pipeline, cv2.CAP_GSTREAMER, 0, 
                                src_fps, (src_wid, src_hei), True )


        logging.info(f'Define Gstreamer Pipeline: {gst_pipeline}')

        if not out.isOpened():
            raise Exception("can't open video writer")

    # Setting Application
    try:
        application = get_application(total_conf)
        if total_conf["application"]["name"] != "default":
            application.set_area(frame=src.get_first_frame() if mode==GUI else None)
    except Exception as e:
        handle_exception(error=e, title="Could not load application ... set app to None", exit=False)
    
    # Start inference
    if mode==GUI: init_cv_win()

    # Infer Parameters
    temp_info, cur_info    = None, None
    cur_fps , temp_fps     = 30, 30
    fps_buf = []

    try:
        while True:
            
            # Get current frame
            t_start = time.time()
            success, frame = src.read()
            
            # Check frame
            if not success:
                if src.get_type() == 'v4l2':
                    break
                else:
                    application.reset()
                    src.reload()
                    continue

            draw = frame.copy()
            
            # Inference
            cur_info = trg.inference( frame )
            if(check_info(cur_info)):
                temp_info, cur_fps = cur_info, temp_fps
            
            # # Drawing result using application and FPS
            if(temp_info):
                draw, app_info = application(draw, temp_info)
                draw = draw_fps( draw, cur_fps )
                
            # Display draw
            if mode==GUI:
                exit_win = display(draw, t_wait_key)
                if exit_win: break

            elif mode==RTSP:
                out.write(draw)

            # Log
            if(cur_info): logging.info(cur_info['detections'])

            # Delay inferenece to fix in 30 fps
            t_cost, t_expect = (time.time()-t_start), (1/src.get_fps())
            time.sleep( (t_expect-t_cost) if( t_cost<t_expect ) else 1e-6 )
            
            # Calculate FPS
            if(check_info(cur_info)):
                fps_buf.append(int(1/(time.time()-t_start)))
                if(len(fps_buf)>10): fps_buf.pop(0)
                temp_fps = sum(fps_buf)/len(fps_buf)

    except Exception as e:
        print(handle_exception(e))

    finally:
        trg.release()            
        src.release()

        if mode==RTSP:
            out.release()
        
        del trg

    logging.warning('Quit')
    sys.exit()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help = "The path of application config")
    parser.add_argument('-s', '--server', action="store_true", help = "Server mode, not to display the opencv windows")
    parser.add_argument('-r', '--rtsp', action="store_true", help = "RTSP mode, not to display the opencv windows")
    parser.add_argument('-d', '--debug', action="store_true", help = "Debug mode")
    parser.add_argument('-m', '--mode', type=int, default = 1, help = "Select sync mode or async mode{ 0: sync, 1: async }")
    parser.add_argument('-i', '--ip', type=str, default = 'localhost', help = "The ip address of RTSP uri")
    parser.add_argument('-p', '--port', type=str, default = '8554', help = "The port number of RTSP uri")
    parser.add_argument('-n', '--name', type=str, default = '/mystream', help = "The name of RTSP uri")

    args = parser.parse_args()

    if not ('/' in args.name):
        args.name = f'/{args.name}'

    main(args)
    