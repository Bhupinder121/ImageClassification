import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

vid = cv2.VideoCapture(0)
# path = os.path.join("D:", "Projects", "models", "imageclassifier.h5")
path = "D:\Projects\Machine Learning\ImageClassification\models\\fiveImageclassifier.h5"
new_model = load_model(path)
emotion = ["angry", "fearful", "happy", "sad", "surpirse"]
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    resize = tf.image.resize(frame, (256,256))
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    prediction = yhat[0]
    text = ""
    max_prediction_text = ""
    max_prediction = 0
    if(prediction[0] > 0.7):
        if (prediction[0] > max_prediction):
            max_prediction_text = "Angry"
        text += "Angry "
    if(prediction[1] > 0.7):
        if (prediction[1] > max_prediction):
            max_prediction_text = "Fear"
        text += "fear "
    if(prediction[2] > 0.7):
        if (prediction[2] > max_prediction):
            max_prediction_text = "happy"
        text += "happy "
    if(prediction[3] > 0.7):
        if (prediction[3] > max_prediction):
            max_prediction_text = "Sad"
        text += "sad "
    if(prediction[4] > 0.7):
        if (prediction[4] > max_prediction):
            max_prediction_text = "Suprise"
        text += "suprise "
    for i in range(len(emotion)):
        print(f"{emotion[i]}: {prediction[i]}")
    cv2.putText(frame, max_prediction_text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()