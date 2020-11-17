#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import face_recognition
from flask_bootstrap import Bootstrap
from realtime_detection import gen_frames

#Initialize the Flask app
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')