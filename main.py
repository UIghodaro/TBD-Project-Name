import cv2                                    # opencv-python, for video capture
import numpy as np                            # mediapipe uses numpy frames/arrays heavily
import time                                   # For use with mediapipe


import mediapipe as mp                        # mediapipe, for hand tracking and command implementation
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = 'Utils/hand_landmarker.task'                           # Path to the Hand Landmarker model, it's pretrained and in this codebase... interesting. Hard to build off

BaseOptions = mp.tasks.BaseOptions                                  # I do not know what this is
HandLandmarker = mp.tasks.vision.HandLandmarker                     # Create an object for the Hand Landmarker
latest_result = None

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions       # I'm guessing this is to do with storing the options for the initial object
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult         # And then this can transform the results then
VisionRunningMode = mp.tasks.vision.RunningMode                     # Holy OOP abuse, I'm guessing they created so many objects for the sake of modularity? It feels like it's needlessly complex though


# Create a hand landmarker instance with the live stream mode:
#def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#    print('hand landmarker result: {}'.format(result))

latest_result = None

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result
    
# Define all the options or something
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),          # Configuring...
    running_mode=VisionRunningMode.LIVE_STREAM,                     # Configuring...
    num_hands=2,
    result_callback=print_result)                                   # Configuring...

########################################################
# Visualisation module created by the brothers at google
########################################################
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
    
 #########################################
 #- Now we actually start getting video -#
 #########################################
 
Camera = cv2.VideoCapture(0)                                        # Define the videocapture/livestream

with HandLandmarker.create_from_options(options) as landmarker:    
    while Camera.isOpened():
        ret, image = Camera.read()                                                    # Get the current webcam image (continuous cause 'while True)

        if not ret:                                                                   # Making sure?
           print('empty camera frame')
           continue

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                            # Convert the camera frame to RGB
        
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)         # data = numpy_frame_from_opencv
       
        timestamp_ms = int(time.time() * 1000)                                        # Live tracking needs the time for some reason
        print(f"-----\n {timestamp_ms} \n---------")
        landmarker.detect_async(mp_image, timestamp_ms)                               # Given a converted image, detect everything I guess

        #landmarker_result = options.result_callback(HandLandmarkerResult, mp_image, timestamp_ms)
        
        if latest_result:
            image = draw_landmarks_on_image(rgb_frame, latest_result)
        cv2.imshow("Frame", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        k = cv2.waitKey(1)
        if k == ord('x'):
            break
    
Camera.release()
cv2.destroyAllWindows()
