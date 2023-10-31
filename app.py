from flask import Flask, Request, jsonify, render_template, Response, session
import torch
import numpy as np
from network import C3D_modelv1
import cv2

torch.backends.cudnn.benchmark = True
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta

app = Flask(__name__)

app.config['SECRET_KEY'] = 'abc'
app.config['UPLOAD_FOLDER'] = 'static/UCF_fighting'
last_email_time = None
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


torch.backends.cudnn.benchmark = True

email_address = 'phamnguyen24303@gmail.com'
email_password = '******************'
recipient_email = 'nguyenpttse170162@fpt.edu.vn'


def send_email(frame):
    global last_email_time
    time = datetime.now()
    if last_email_time is None or (time - last_email_time) >= timedelta(minutes=2):
        subject = 'Alert: Fight Detected in Public Area'
        recipient_name = recipient_email
        your_name = 'Thao Nguyen'
        organization = 'FPT Software'

        message = f'Dear {recipient_name},\n\n' \
                  f'Our monitoring program has detected a fight in the area of *** at {time}.\n' \
                  f'Stay safe.\n\n' \
                  f'Sincerely,\n' \
                  f'{your_name}\n' \
                  f'{organization}'

        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))
        # Attach the frame as an image
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return "Error converting frame to image format"

        frame_bytes = buffer.tobytes()
        img = MIMEImage(frame_bytes)
        msg.attach(img)

        try:
            # Establish an SMTP connection
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email_address, email_password)

            # Send the email
            server.sendmail(email_address, recipient_email, msg.as_string())
            server.quit()

            print("Email sent successfully.")
            last_email_time = time
        except Exception as e:
            print("Email could not be sent:", str(e))


def video_detection(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/2class.txt', 'r') as f:
        class_names = f.readlines()

    model = C3D_modelv1.C3D(num_classes=2)
    checkpoint = torch.load('run_new/run_10/models/C3D-traindata_epoch-19.pth.tar',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Initialize the camera feed (default camera, you can specify a camera index)
    cap = cv2.VideoCapture(path)  # Use camera index 0 for the default camera

    clip = []
    y = 0
    n = 600

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if path != 0:
            frame = cv2.resize(frame, (1920, 1080))

        tmp_ = cv2.resize(frame, (112, 112))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            if probs[0][1].detach().cpu().item() > 0.65:
                y += 1
                n = 0

            else:
                n += 1

            if probs[0][1].detach().cpu().item() > 0.5:
                colour = (0, 0, 255)
            else:
                colour = (0, 255, 0)

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        colour, 2)
            cv2.putText(frame, "%.4f" % probs[0][label], (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        colour, 2)
            if n < 250:
                cv2.putText(frame, "Warning", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 255), 2)
            if y > 35 and n < 60:
                cv2.putText(frame, "Warning: Violence", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 225), 2)
            clip.pop(0)

            if probs[0][1].detach().cpu().item() > 0.95:
                send_email(frame)
        yield frame

def gen_frame(path_x=''):
    output = video_detection(path_x)
    for detect in output:
        ref, buffer = cv2.imencode('.jpg', detect)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def gen_frame_cam(path):
#     output = video_detection(path)
#     for detect in output:
#         ref, buffer = cv2.imencode('.jpg', detect)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename).replace("\\", "/")))

        session['video_path'] = os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                         secure_filename(file.filename).replace("\\", "/"))
        )
    return render_template('video.html', form=form)


# Define the API endpoint for video classification
@app.route('/video')
def video():
    return Response(gen_frame(path_x=session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Rendering the Webcam Rage
# Now lets make a Webcam page for the application
# Use 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')


@app.route('/webapp')
def webapp():
    return Response(gen_frame(0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
