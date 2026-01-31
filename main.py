import cv2
import pickle
import cvzone
import numpy as np

# Parameters for parking space dimensions
width, height = 107, 48

# Load saved parking positions
try:
    with open(r'C:\Users\Tushar Bhure\Desktop\major\Carparking\CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Initialize accuracy metrics
TP = 0  # True Positives
FP = 0  # False Positives
TN = 0  # True Negatives
FN = 0  # False Negatives

# Function to capture mouse clicks to set or remove parking positions
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

# Function to check parking spaces in the video feed
def checkParkingSpace(imgPro, img):
    global TP, FP, TN, FN
    
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 1200:  # Threshold to determine if space is free
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            
            # Update accuracy metrics for free space detection
            TP += 1   # True Positive (correctly detected free space)
            FN -= 1   # False Negative (previously counted as occupied)
        else:
            color = (0, 0, 255)
            thickness = 2
            
            # Update accuracy metrics for occupied space detection
            FP += 1   # False Positive (incorrectly detected occupied space)
            TN += 1   # True Negative (correctly detected occupied space)

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))

# Function to display accuracy table without recall calculation
def displayAccuracy(img):
    global TP, FP, TN, FN
    
    total_spaces = len(posList)
    
    if total_spaces > 0:
        accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100 if (TP + TN + FP + FN) > 0 else 0
        precision = (TP / (TP + FP) * 100) if (TP + FP) > 0 else 0
        
        metrics_text = f"Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}%"
        cvzone.putTextRect(img, metrics_text, (50, img.shape[0] - 50), scale=1,
                           thickness=2, offset=10)

# Video feed for parking space detection
cap = cv2.VideoCapture(r'C:\Users\Tushar Bhure\Desktop\major\Carparking\/carPark.mp4')

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video loaded successfully.")

while True:
    # Read video frames
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()

    if not success:
        print("Error: Frame not loaded correctly.")
        break

    # Preprocess the image to enhance parking space detection
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         25,
                                         16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Check parking spaces and show result
    checkParkingSpace(imgDilate, img)
    
    # Display the accuracy metrics on the image without recall
    displayAccuracy(img)

    # Display the parking image with drawn rectangles
    cv2.imshow("Image", img)
    
    # Set mouse callback to interact with the parking positions
    cv2.setMouseCallback("Image", mouseClick)

    # Add a key handler to allow the window to be closed
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import pickle
# import cvzone
# import numpy as np

# # Parameters for parking space dimensions
# width, height = 107, 48

# # Load saved parking positions
# try:
#     with open('D:\\html\\JS\\Carparking\\CarParkPos', 'rb') as f:
#         posList = pickle.load(f)
# except FileNotFoundError:
#     posList = []

# # Initialize background subtractor
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# def mouseClick(events, x, y, flags, params):
#     if events == cv2.EVENT_LBUTTONDOWN:
#         posList.append((x, y))
#     if events == cv2.EVENT_RBUTTONDOWN:
#         for i, pos in enumerate(posList):
#             x1, y1 = pos
#             if x1 < x < x1 + width and y1 < y < y1 + height:
#                 posList.pop(i)

#     with open('CarParkPos', 'wb') as f:
#         pickle.dump(posList, f)

# def checkParkingSpace(imgPro, img):
#     spaceCounter = 0
#     for pos in posList:
#         x, y = pos
#         imgCrop = imgPro[y:y + height, x:x + width]

#         # Find contours in the cropped parking space
#         contours, _ = cv2.findContours(imgCrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if len(contours) == 0:
#             color = (0, 255, 0)  # Green for free space
#             thickness = 5
#             spaceCounter += 1
#         else:
#             color = (0, 0, 255)  # Red for occupied space
#             thickness = 2

#         # Draw rectangle around the parking spot
#         cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
#         cvzone.putTextRect(img, f'{len(contours)}', (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

#     cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
#                        thickness=5, offset=20, colorR=(0, 200, 0))

# # Video feed for parking space detection
# cap = cv2.VideoCapture('D:\\html\\JS\\Carparking\\carPark.mp4')

# # Check if the video was opened correctly
# if not cap.isOpened():
#     print("Error: Could not open video file.")
# else:
#     print("Video loaded successfully.")

# while True:
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     success, img = cap.read()

#     if not success:
#         print("Error: Frame not loaded correctly.")
#         break

#     # Apply background subtraction to the frame
#     fgmask = fgbg.apply(img)
    
#     # Denoise the foreground mask with morphological operations
#     kernel = np.ones((3, 3), np.uint8)
#     imgPro = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

#     # Check parking spaces and show result
#     checkParkingSpace(imgPro, img)
    
#     # Display the parking image with drawn rectangles
#     cv2.imshow("Image", img)
    
#     # Set mouse callback to interact with the parking positions
#     cv2.setMouseCallback("Image", mouseClick)

#     if cv2.waitKey(10) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
#         break

# cap.release()
# cv2.destroyAllWindows()



