import os
import glob
from src.human.constants import *
from flask import Flask, render_template,request
from src.human.pipeline.prediction_pipeline import SinglePrediction

app = Flask(__name__)

@app.route('/',methods= ['GET'])
def land():
    return render_template("landing_page.html")

@app.route('/index',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        upload_file =request.files['fileup']
        filename = upload_file.filename
        upload_audio_path = os.path.join(os.getcwd(),STATIC_DIR,UPLOAD_SUB_DIR)
        os.makedirs(upload_audio_path,exist_ok=True)

        # Remove all files in dir before saving upload image
        filelist = glob.glob(os.path.join(upload_audio_path, "*"))
        for f in filelist:
            os.remove(f)
 
        upload_audio_path = os.path.join(os.getcwd(),STATIC_DIR,UPLOAD_SUB_DIR,filename)
        upload_file.save(upload_audio_path)
        prediction = SinglePrediction()
        prediction.predict(filename)

    return render_template("index.html", upload_image = filename, upload = True)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)