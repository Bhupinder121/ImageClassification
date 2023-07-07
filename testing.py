import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

vid = cv2.VideoCapture(0)
# path = os.path.join("D:", "Projects", "models", "imageclassifier.h5")
path = "D:\Projects\Machine Learning\ImageClassification\models\imageclassifier.h5"
new_model = load_model(path)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    resize = tf.image.resize(frame, (256,256))
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    if(yhat > 0.5):
        text = "Sad"
    else:
        text = "Happy"
    cv2.putText(frame, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
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