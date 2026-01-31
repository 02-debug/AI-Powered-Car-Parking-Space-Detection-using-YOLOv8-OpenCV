import cv2
import pickle

# Load the image
img = cv2.imread('carParkImg.png')
height, width, _ = img.shape

# List to hold parking space positions
parkPos = []

def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        parkPos.append((x, y))
        cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouseClick)

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Save the positions
with open('CarParkPos', 'wb') as f:
    pickle.dump(parkPos, f)