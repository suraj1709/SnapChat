from flask import Flask, Response, send_file,render_template
import numpy as np
import cv2
import skimage.io as imshow
from keras.models import load_model

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')



        
@app.route("/")
def route():
    model = load_model('model.hdf5')
    FACE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    EYE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3,640) # set Width
    cap.set(4,480)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CLASSIFIER.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            img_copy=np.copy(img)
            width_original = roi_gray.shape[1] 
            height_original = roi_gray.shape[0] 
            img_gray = cv2.resize(roi_gray, (96, 96))
         
            img_gray = img_gray/255
            img_model = np.reshape(img_gray, (1,96,96,1))
            keypoints = model.predict(img_model)[0]         # Predict keypoints for the current input
            
            # Keypoints are saved as (x1, y1, x2, y2, ......)
            x_coords = keypoints[0::2]      # Read alternate elements starting from index 0
            y_coords = keypoints[1::2]      # Read alternate elements starting from index 1
            
            x_coords_denormalized = (x_coords+0.5)*width_original       # Denormalize x-coordinate
            y_coords_denormalized = (y_coords+0.5)*height_original      # Denormalize y-coordinate
            
            for i in range(len(x_coords)):          # Plot the keypoints at the x and y coordinates
                cv2.circle(roi_color, (x_coords_denormalized[i], y_coords_denormalized[i]), 2, (255,255,0), -1)
            
            # Particular keypoints for scaling and positioning of the filter
            left_lip_coords = (int(x_coords_denormalized[11]), int(y_coords_denormalized[11]))
            right_lip_coords = (int(x_coords_denormalized[12]), int(y_coords_denormalized[12]))
            top_lip_coords = (int(x_coords_denormalized[13]), int(y_coords_denormalized[13]))
            bottom_lip_coords = (int(x_coords_denormalized[14]), int(y_coords_denormalized[14]))
            left_eye_coords = (int(x_coords_denormalized[3]), int(y_coords_denormalized[3]))
            right_eye_coords = (int(x_coords_denormalized[5]), int(y_coords_denormalized[5]))
            brow_coords = (int(x_coords_denormalized[6]), int(y_coords_denormalized[6]))
            
            # Scale filter according to keypoint coordinates
            beard_width = right_lip_coords[0] - left_lip_coords[0]
            glasses_width = right_eye_coords[0] - left_eye_coords[0]
            
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)
            santa_filter = cv2.imread('filters/santa_filter.png', -1)
            santa_filter = cv2.resize(santa_filter, (beard_width*3,100))
            sw,sh,sc = santa_filter.shape
            
            #for i in range(0,sw):       # Overlay the filter based on the alpha channel
                #for j in range(0,sh):
                    #if santa_filter[i,j][3] != 0:
                        #img_copy[top_lip_coords[1]+i+y-20, left_lip_coords[0]+j+x-60] = santa_filter[i,j]
            glasses = cv2.imread('filters/glasses.png', -1)
            glasses = cv2.resize(glasses, (glasses_width*2,150))
            gw,gh,gc = glasses.shape
        
            for i in range(0,gw):       # Overlay the filter based on the alpha channel
                for j in range(0,gh):
                    if glasses[i,j][3] != 0:
                        img_copy[brow_coords[1]+i+y-50, left_eye_coords[0]+j+x-60] = glasses[i,j]
            cv2.imwrite('at.jpg', img)
            cv2.imwrite('pt.jpg', img_copy)
            cv2.imwrite('pt.jpg', img_copy)
            yield (b'--frame\r\n' 
                   b' Content-Type: image/jpeg\r\n\r\n' + open('pt.jpg', 'rb').read() + b'\r\n')
@app.route("/stream")
def stream():
    return Response(route(), mimetype="multipart/x-mixed-replace; boundary=frame")


if(__name__ == "__main__"):
    app.run()