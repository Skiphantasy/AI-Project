#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
from flask_bootstrap import Bootstrap

#Initialize the Flask app
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/opencv')
def opencv():
    return render_template('opencv.html')

@app.route('/server')
def server():
    return render_template('server.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')