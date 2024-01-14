import cv2
import numpy as np

##########################################################
# Canny and processing here
def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 12)
    img_canny = cv2.Canny(img_blur, 10, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def get_contours(img, img_original):
    img_contours = img_original.copy()
    contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), -1) 

    return img_contours

######################################################
# Everything starts here
cap = cv2.VideoCapture("data/test10.mp4") 

########################################################
# Thresholds here
threshold_value = 215                  
threshold_color = (0, 0, 200)
time_threshold = 5

#######################################################
success, img1 = cap.read()
heat_map = np.zeros(img1.shape[:-1])
success, img2 = cap.read()
frame_counter = 0

# Initialize the video writer
frame_width, frame_height = img1.shape[1], img1.shape[0]
fourcc = cv2.VideoWriter_fourcc(*'VP90')
out = cv2.VideoWriter('output_video.webm', fourcc, 30, (frame_width, frame_height))

while success:
    frame_counter += 1  # Increment the frame counter

    if frame_counter == 10:  # Check if 10 frames have passed
        frame_counter = 0
    diff = cv2.absdiff(img1, img2)
    img_contours = get_contours(process(diff), img1)

    heat_map[np.all(img_contours == [0, 255, 0], 2)] += 3.2
    heat_map[np.any(img_contours != [0, 255, 0], 2)] -= 6
    heat_map[heat_map < 0] = 0
    heat_map[heat_map > 255] = 255

    img_mapped = cv2.applyColorMap(heat_map.astype('uint8'), cv2.COLORMAP_JET)
    threshold_mask = heat_map > threshold_value
    img_mapped[threshold_mask] = threshold_color

    out.write(img_mapped)  # Write the frame to the video file

    cv2.putText(img_mapped, f"test", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.imshow("Original", img1)
    # cv2.imshow("Heat Map", img_mapped)

    img1 = img2
    success, img2 = cap.read()
    print('converting...')
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video writer and close OpenCV windows
out.release()
cv2.destroyAllWindows()
