import matplotlib.pyplot as plt
import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
#from gpiozero import LED,Device
import gpiod
import time
import torch
import requests
from gpiod.line import Direction, Value

with gpiod.Chip("/dev/gpiochip0") as chip:
    info = chip.get_info()
    print(f"{info.name} [{info.label}] ({info.num_lines} lines)")



LINE = 10
LINE1 = 9
LINE2 = 11

with gpiod.request_lines(
    "/dev/gpiochip4",
    consumer="blink-example",
    config={
        LINE: gpiod.LineSettings(
            direction=Direction.OUTPUT, output_value=Value.ACTIVE
        ),
        LINE1: gpiod.LineSettings(
            direction=Direction.OUTPUT, output_value=Value.ACTIVE
        ),
        LINE2: gpiod.LineSettings(
            direction=Direction.OUTPUT, output_value=Value.ACTIVE
        )
    },
) as request:
    while True:
        request.set_value(LINE, Value.ACTIVE)
        request.set_value(LINE1, Value.ACTIVE)
        request.set_value(LINE2, Value.ACTIVE)
        time.sleep(1)
        print("test")
        request.set_value(LINE, Value.INACTIVE)
        request.set_value(LINE1, Value.INACTIVE)
        request.set_value(LINE2, Value.INACTIVE)
        time.sleep(1)

'''
c1=LED(16)
c2=LED(20)
c3=LED(21)
c1.off()
c2.off()
c3.off()
'''


# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
#picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888',"size":(640,480)}))
picam2.start()

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

path= '20240806_15h02m41s_grim.png'
image=cv2.imread(path)
print("test1")


def draw_grid(frame):
    h, w, _ = frame.shape
    dx = 320  # Width of each rectangle
    for x in range(0, w, dx):
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)
    return frame

def get_grid_cells_with_objects(results, frame_shape=(1080, 1920)):
    h, w = frame_shape
    dx = 320  # Width of each rectangle
    grid_cells = [0] * 6  # Six rectangles

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2
            grid_x = int(cx // dx)
            if grid_x < len(grid_cells):
                grid_cells[grid_x] = 1  # Mark cell as occupied

    return grid_cells

def setup_leds(grid_cells):
    
    if grid_cells == [0,0,0,0,0,0]:  # No obstacles in this cell
        light_up_leds(0)
    elif grid_cells == [1,0,0,0,0,0]: # c1
        light_up_leds(5)
    elif grid_cells == [0,1,0,0,0,0]: # c2
        light_up_leds(5)
    elif grid_cells == [0,0,1,0,0,0]: # c3
        light_up_leds(6) 
    elif grid_cells == [0,0,0,1,0,0]: # c4
        light_up_leds(1)
    elif grid_cells == [0,0,0,0,1,0]: # c5
        light_up_leds(2) 
    elif grid_cells == [0,0,0,0,0,1]: # c6
        light_up_leds(2) 
    elif grid_cells == [1,1,0,0,0,0]: # c1 and c2
        light_up_leds(5)
    elif grid_cells == [1,0,1,0,0,0]: # c1 and c3
        light_up_leds(6)
    elif grid_cells == [1,0,0,1,0,0]: # c1 and c4
        light_up_leds(6)
    elif grid_cells == [1,0,0,0,1,0]: # c1 and c5
        light_up_leds(3)
    elif grid_cells == [1,0,0,0,0,1]: # c1 and c6
        light_up_leds(3)
    elif grid_cells == [0,1,1,0,0,0]: # c2 and c3
        light_up_leds(6)
    elif grid_cells == [0,1,0,1,0,0]: # c2 and c4
        light_up_leds(6)
    elif grid_cells == [0,1,0,0,1,0]: # c2 and c5
        light_up_leds(3)
    elif grid_cells == [0,1,0,0,0,1]: # c2 and c6
        light_up_leds(4)
    elif grid_cells == [0,0,1,1,0,0]: # c3 and c4
        light_up_leds(1)
    elif grid_cells == [0,0,1,0,1,0]: # c3 and c5
        light_up_leds(1)
    elif grid_cells == [0,0,1,0,0,1]: # c3 and c6
        light_up_leds(1)
    elif grid_cells == [0,0,0,1,1,0]: # c4 and c5
        light_up_leds(1)
    elif grid_cells == [0,0,0,1,0,1]: # c4 and c6
        light_up_leds(1)
    elif grid_cells == [0,0,0,0,1,1]: # c5 and c6
        light_up_leds(2)
    elif grid_cells == [1,1,1,0,0,0]: # c1,c2,c3
        light_up_leds(6)
    elif grid_cells == [1,1,0,1,0,0]: # c1,c2,c4
        light_up_leds(6)
    elif grid_cells == [1,1,0,0,1,0]: # c1,c2,c5
        light_up_leds(6)
    elif grid_cells == [1,1,0,0,0,1]: # c1,c2,c6
        light_up_leds(4)
    elif grid_cells == [1,0,1,1,0,0]: # c1,c3,c4
        light_up_leds(6)
    elif grid_cells == [1,0,1,0,1,0]: # c1,c3,c5
        light_up_leds(6)
    elif grid_cells == [1,0,1,0,0,1]: # c1,c3,c6
        light_up_leds(4)
    elif grid_cells == [1,0,0,1,1,0]: # c1,c4,c5
        light_up_leds(2)
    elif grid_cells == [1,0,0,1,0,1]: # c1,c4,c6
        light_up_leds(2)
    elif grid_cells == [1,0,0,0,1,1]: # c1,c5,c6
        light_up_leds(3)
    elif grid_cells == [1,0,0,0,1,1]: # c2,c3,c4
        light_up_leds(3)
    elif grid_cells == [1,0,0,0,1,1]: # c2,c3,c5
        light_up_leds(3)
    elif grid_cells == [1,0,0,0,1,1]: # c2,c3,c6
        light_up_leds(3)
    elif grid_cells == [1,0,0,0,1,1]: # c2,c4,c5
        light_up_leds(3)
    elif grid_cells == [1,0,0,0,1,1]: # c2,c4,c6
        light_up_leds(3)
    elif grid_cells == [1,0,0,0,1,1]: # c2,c5,c6
        light_up_leds(3)
    elif grid_cells == [0,0,1,1,1,1]: # c3,c4,c5
        light_up_leds(1)
    elif grid_cells == [0,0,1,1,0,1]: # c3,c4,c6
        light_up_leds(1)
    elif grid_cells == [0,0,1,0,1,1]: # c3,c5,c6
        light_up_leds(1)
    elif grid_cells == [1,0,0,0,1,1]: # c4,c5,c6
        light_up_leds(3)
    elif grid_cells == [1,1,1,1,0,0]: # c1,c2,c3,c4
        light_up_leds(6)
    elif grid_cells == [1,1,1,0,1,0]: # c1,c2,c3,c5
        light_up_leds(6)
    elif grid_cells == [1,1,1,0,0,1]: # c1,c2,c3,c6
        light_up_leds(5)
    elif grid_cells == [1,1,0,1,1,0]: # c1,c2,c4,c5
        light_up_leds(6)
    elif grid_cells == [1,1,0,1,0,1]: # c1,c2,c4,c6
        light_up_leds(7)
    elif grid_cells == [1,1,0,0,1,1]: # c1,c2,c5,c6
        light_up_leds(3)
    elif grid_cells == [1,0,1,1,1,0]: # c1,c3,c4,c5
        light_up_leds(6)
    elif grid_cells == [1,0,1,1,0,1]: # c1,c3,c4,c6
        light_up_leds(7)
    elif grid_cells == [1,0,1,0,1,1]: # c1,c3,c5,c6
        light_up_leds(7)
    elif grid_cells == [1,0,0,1,1,1]: # c1,c4,c5,c6
        light_up_leds(2)
    elif grid_cells == [0,1,1,1,1,0]: # c2,c3,c4,c5
        light_up_leds(1)
    elif grid_cells == [0,1,1,1,0,1]: # c2,c3,c4,c6
        light_up_leds(1)
    elif grid_cells == [0,1,1,0,1,1]: # c2,c3,c5,c6
        light_up_leds(1)
    elif grid_cells == [0,1,0,1,1,1]: # c2,c4,c5,c6
        light_up_leds(1)
    elif grid_cells == [0,0,1,1,1,1]: # c3,c4,c5,c6
        light_up_leds(1)
    elif grid_cells == [1,1,1,1,1,0]: # c1,c2,c3,c4,c5
        light_up_leds(6)
    elif grid_cells == [1,1,1,1,0,1]: # c1,c2,c3,c4,c6
        light_up_leds(7)
    elif grid_cells == [1,1,1,0,1,1]: # c1,c2,c3,c5,c6
        light_up_leds(7)
    elif grid_cells == [1,1,0,1,1,1]: # c1,c2,c4,c5,c6
        light_up_leds(7)
    elif grid_cells == [1,0,1,1,1,1]: # c1,c3,c4,c5,c6
        light_up_leds(7)
    elif grid_cells == [0,1,1,1,1,1]: # c2,c3,c4,c5,c6
        light_up_leds(1)
    elif grid_cells == [1,1,1,1,1,1]: # c1,c2,c3,c4,c5,c6
        light_up_leds(7)
'''
def light_up_leds(x):
    if(x==0):
        c1.off()
        c2.off()
        c3.off()
        print("turning on none")
    elif(x==1):
        c1.on()
        c2.off()
        c3.off()
        print("turning on one")
    elif(x==2):
        c1.off()
        c2.on()
        c3.off()
        print("turning on two")
    elif(x==3):
        c1.on()
        c2.on()
        c3.off()
        print("turning on one and two")
    elif(x==4):
        c1.off()
        c2.off()
        c3.on()
        print("turning on three")
    elif(x==5):
        c1.on()
        c2.off()
        c3.on()
        print("turning on one and three")
    elif(x==6):
        c1.off()
        c2.on()
        c3.on()
        print("turning on two and three")
    elif(x==7):
        print("Stop where you are")
'''
while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()
    print(frame.shape)
    # Run YOLOv8 inference on the frame
    results = model(frame)
    print("test2")
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    grid_frame = draw_grid(annotated_frame)
    cell_with_object = get_grid_cells_with_objects(results)
    print(cell_with_object)
    print("cell number")
    plt.imshow(cv2.cvtColor(grid_frame,cv2.COLOR_BGR2RGB))
    plt.show(block=False)
    setup_leds(cell_with_object)
    plt.pause(5)
    plt.close()
# Release resources and close windows
cv2.destroyAllWindows()
