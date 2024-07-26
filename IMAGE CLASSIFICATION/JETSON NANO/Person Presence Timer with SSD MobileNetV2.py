#Person Presence Timer with SSD MobileNetV2
import jetson.inference
import jetson.utils
import time

cam = jetson.utils.gstCamera(640, 480, '/dev/video0')
disp = jetson.utils.glDisplay()
font = jetson.utils.cudaFont()
net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.5)

timer_started = False
start_time = time.time()
total_time = 0


while disp.IsOpen():
    frame, width, height = cam.CaptureRGBA()
    detections = net.Detect(frame, width, height)
    classID_1_detected = any(detection.ClassID == 1 for detection in detections)

    if classID_1_detected:
        if timer_started:
            elapsed_time = time.time() - start_time
            timer_started = False
            total_time += elapsed_time
            print(f"Timer stopped at {total_time} seconds")
    else:
        if not timer_started:
            start_time = time.time()
            timer_started = True
            print("Timer started")

    for detection in detections:
        item = net.GetClassDesc(detection.ClassID == 1 )
        if detection.ClassID != 1:
          font.OverlayText(frame, width, height, f'unlabeled', int(detection.Left), int(detection.Top), font.White , font.Black)
        font.OverlayText(frame, width, height, f'{total_time}', 5, 5, font.Yellow, font.Blue)
    disp.RenderOnce(frame, width, height)
