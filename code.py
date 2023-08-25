from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model('hand_gesture_classifier_model.h5',compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])

vid = cv2.VideoCapture(0)

res=180
tsd=np.zeros((1,res,res,3))

while True:

    ret, frame = vid.read()

    length = 250
    top_left = (0, 0)  
    bottom_right = (length, length)  
    frame=cv2.rectangle(frame, top_left, bottom_right,(0,0,255),2)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    img = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(res,res))
    tsd[0]=img

    y_pred=model.predict(tsd)
    y_pred=np.argmax(y_pred,axis=1)
    # print(y_pred)
    if y_pred==0:
        print("1")
        frame=cv2.putText(frame,'1',(85,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)
    elif y_pred==1:
        print("2")
        frame=cv2.putText(frame,'2',(85,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)
    elif y_pred==2:
        print("3")
        frame=cv2.putText(frame,'3',(85,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)
    elif y_pred==3:
        print("4")
        frame=cv2.putText(frame,'4',(85,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)
    elif y_pred==4:
        print("5")
        frame=cv2.putText(frame,'5',(85,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)
    else:
        print("0")
        frame=cv2.putText(frame,'0',(85,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)

    cv2.imshow('Webcam', frame)

vid.release()
cv2.destroyAllWindows()
