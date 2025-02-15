import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
    
    frame = cv2.flip(frame, 1)
    cv2.imshow("Video Frame", frame)
    
    key_pressed = cv2.waitKey(1) 
    
    if key_pressed == 13 or cv2.getWindowProperty('Video Frame', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()