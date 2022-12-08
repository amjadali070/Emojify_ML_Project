from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)

camera = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not camera.isOpened():
    raise IOError("Cannot open webcam")


def generate_frames():
    while True:
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceDetector.detectMultiScale(gray_img, 1.1,5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)     
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def get_output():
    while True:
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
            print(result['dominant_emotion'])
            emotion = result['dominant_emotion']
            
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceDetector.detectMultiScale(gray_img, 1.1,5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)      
            
            emoji_dict = {"angry": "emojis/angry.png", "disgust": "emojis/disgusted.png", "fear": "emojis/fearful.png", 
                           "happy": "emojis/happy.png","neutral": "emojis/neutral.png", "sad": "emojis/sad.png", 
                           "surprise": "emojis/surprised.png"}
            
            img = open(emoji_dict[emotion], "rb").read()
            

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stream")
def stream():
    return Response(get_output(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)