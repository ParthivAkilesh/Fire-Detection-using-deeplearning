from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

clsf = load_model("model.h5")

res_labels = ["DANGER FIRE DETECTED!", "SAFE NO FIRE DETECTED"]

cap = cv2.VideoCapture("G:\Projects and Related\Project\Projects Learning\Fire Detection\FireTesting.mp4")


while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (196, 196))
    np_fm = np.asarray(frame)
    # print("shape: ", np_fm.shape)
    img = np_fm.reshape(-1,196, 196,3)
    prediction = clsf.predict(img)[0]
    frame = cv2.resize(frame, (800, 500))
    
    font = cv2.FONT_HERSHEY_DUPLEX
    
    if prediction>0.5:
        p_frame = cv2.putText(frame, res_labels[0], (0, 490), font, 2, (0, 0, 255), 3)
    else:
         p_frame = cv2.putText(frame, res_labels[1], (0, 490), font, 2, (0, 255, 0), 3)
    cv2.imshow("Frame ", p_frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    
    