import cv2
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'Utils/hand_landmarker.task'                           # Path to the Hand Landmarker model, it's pretrained and in this codebase... interesting. Hard to build off

BaseOptions = mp.tasks.BaseOptions                                  # I do not know what this is
HandLandmarker = mp.tasks.vision.HandLandmarker                     # Create an object for the Hand Landmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions       # I'm guessing this is to do with storing the options for the initial object
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult         # And then this can transform the results then
VisionRunningMode = mp.tasks.vision.RunningMode                     # Holy OOP abuse, I'm guessing they created so many objects for the sake of modularity? It feels like it's needlessly complex though


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    
# Define all the options or something
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),          # Configuring...
    running_mode=VisionRunningMode.LIVE_STREAM,                     # Configuring...
    result_callback=print_result)                                   # Configuring...


 #########################################
 #- Now we actually start getting video -#
 #########################################
 
Camera = cv2.VideoCapture(0)                                        # Define the videocapture/livestream

with HandLandmarker.create_from_options(options) as landmarker:    
    while True:
        ret, image = Camera.read()
        
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)         #data = numpy_frame_from_opencv
        #landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        rgb_frame = cv2.cvtColor(mp_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frame", rgb_frame)
        
        k = cv2.waitKey(1)
        if k == ord('x'):
            break
    
Camera.release()
cv2.destroyAllWindows()
