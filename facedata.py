import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "./face_dataset/"

name = input("Enter the name of the person : ")

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
    
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #returns top left coordinates of face along with width and height
    faces = face_cascade.detectMultiScale(grayframe, 1.3, 5) #framename, scale factor, number of neighbours, min size of face
    
    if len(faces) == 0:
        print("No faces detected in the current frame.")
        continue
    else:
        print(f"Detected {len(faces)} face(s).")
    
    k = 1
    
    #sorting the faces according to largest area ie. faces returns an array(x,y,w,h) with indexes 0,1,2,3 so we are multiplying w and h, according to descending order
    faces = sorted(faces, key = lambda x : x[2]*x[3], reverse = True) 
    
    
    for face in faces[:1]: #the first detected face
        x,y,w,h = face
        
        offset = 5
        face_offset = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_selection = cv2.resize(face_offset, (100,100))
        
        skip+=1
        if skip%10 == 0:
            face_data.append(face_selection)

        
        cv2.imshow(f"Face {k}", face_selection)
        k+=1
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
        
    cv2.imshow("Video", frame)
    
    key_pressed = cv2.waitKey(1) 
    
    if key_pressed == 13 or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        break  
    
cap.release()
cv2.destroyAllWindows()    
        
if len(face_data) > 0:
    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    print(f"Dataset shape: {face_data.shape}")
    np.save(dataset_path + name, face_data)
    print(f"Dataset saved at: {dataset_path + name}.npy")
else:
    print("No face data collected. Please ensure proper positioning and lighting and try again.")

