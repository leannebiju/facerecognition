# Face Recognition

  ## facedata.py
  This code is used to enter the face to the dataset. It takes the name as input. And sorts the faces in order of the area each face in the screen takes and takes pictures of the face with the largest area and stores the pictures as a numpy array with the given name in a folder called *face_dataset* until the window is closed.

  ## face_recognition.py
  This code opens the webcam, checks for faces and classifies each face according to the entries in the face_dataset folder according to the KNN classification.


# Running the code

First run the facedata.py to store the facedata in face_dataset folder. 

Then run the face_recognition to recogonise the faces and label accordingly.

# Credits

This project is done following a [youtube video](https://www.youtube.com/watch?v=vA-JiuYX--Y)
