import jetson.inference
import jetson.utils
# Set up camera
cam = jetson.utils.gstCamera(640, 480, '/dev/video0')
# Set up display
disp = jetson.utils.glDisplay()
# Set up font
font = jetson.utils.cudaFont()
# Load the SSD-Mobilenet-v2 model
net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.5)
# Main loop
while disp.IsOpen():
    # Capture a frame from the camera
    frame, width, height = cam.CaptureRGBA()
    # Detect objects in the frame
    detections = net.Detect(frame, width, height)
    # Render the frame with detected objects
    for detection in detections:
        item = net.GetClassDesc(detection.ClassID)
        font.OverlayText(frame, width, height, item, int(detection.Left), int(detection.Top), font.Black, font.White)
    # Display the frame
    disp.RenderOnce(frame, width, height)
