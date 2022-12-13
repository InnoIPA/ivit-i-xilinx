import cv2, time

def main_loop(cap):
    count, fps = 0, cap.get(cv2.CAP_PROP_FPS)
    fps_pool = []
    
    print('Start Capture Loop ... \n')
    
    while(True):
        
        t_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret: break

        count = count + 1
        print(f"\rID: {count}, FPS: {fps:<5}", end='')

        fps_pool.append( int(1/(time.perf_counter() - t_start)) )
        if len(fps_pool)>=10:
            fps = sum(fps_pool)//len(fps_pool); fps_pool.clear()

print(
    """\n Press Ctrl+C to exit !!! \n"""
)

cap = cv2.VideoCapture('/dev/video0')
cap = cv2.VideoCapture('./data/car.mp4')

print('Initialized Video Capture')

try:
    main_loop(cap)

except KeyboardInterrupt:
    pass

except Exception as e:
    print(e)

finally:
    print('\nQuit')
    cap.release()
