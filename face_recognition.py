import numpy as np #numerical operations
import cv2 #opencv tasks
import os #for interaction with os (filepaths etc...)

#KNN code
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum()) #Eucledian distance

def knn(train, test, k = 5): #training data with last coloum label, test - point to be classified 
    dist = []
    
    for i in range(train.shape[0]): #.shape returns the dimensions (no of rows, no of columns) as a tuple. here we use [0] to access only no of rows
        ix = train[i, :-1] #get feature vector selects the entire row but not the last element
        iy = train[i, -1]  #get label i.e. last column
        d = distance(test, ix) #computing distance btwn test and training point
        dist.append([d, iy]) #appending the distance and the label as a pair to the list
    #sort based on distance from each tuple entered in [] to the dist array to get the top k neighbours
    dk = sorted(dist, key = lambda x:x[0])[:k]
    #retrieve only the labels by converting to numpy array i.e each tuple a row then extracting the last column
    labels = np.array(dk)[:, -1]
    
    #get frequencies of labels
    output = np.unique(labels, return_counts = True)
    #find max freq and corresponding label
    index = np.argmax(output[1])#output[1] has count of labels
    return output[0][index]#output[0] has labels

#KNN code ends

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

dataset_path = "./face_dataset/"

face_data = [] #stores face images as numpy
labels = []  #stores corresponding labels
class_id = 0 #unique id for every given file
names = {} #dictionary that maps name to class id

#dataset prep
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4] #taking all characters except last 4 chars that are '.npy'
        #load the face embeddings and store to data item
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        
        target = class_id * np.ones((data_item.shape[0],))#creates an array of ones with the same size as no of images and assigns unique label, eg each picture of a person, say bob, will have label the same as class id
        class_id += 1 #next person
        labels.append(target)

#converting tuples and labels to columns
face_dataset = np.concatenate(face_data, axis = 0) #concatenating by increasing number of rows
face_labels = np.concatenate(labels, axis = 0).reshape((-1,1))#concatenating and transforming so that it is in a single column
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis = 1) #concatenating by adding columns
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

#display

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
    
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(grayframe, 1.3, 5) #framename, scale factor, number of neighbours, min size of face
    
    for face in faces:
        x,y,w,h = face
        
        offset = 5
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))
        
        out = knn(trainset, face_section.flatten())
        
        cv2.putText(frame, names[int(out)],(x,y-10), font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255),2)
                
    cv2.imshow("Video", frame)
    
    key_pressed = cv2.waitKey(1) 
    
    if key_pressed == 13 or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        break  
    
cap.release()
cv2.destroyAllWindows()    