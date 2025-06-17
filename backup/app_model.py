from flask import Flask, render_template, Response, request, jsonify, url_for, session, redirect, flash
import cv2
import numpy as np
import torch
import pickle
import os
import time
import logging
from scipy.spatial import distance
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import model ST-GCN BiLSTM (bạn cần đặt file model_stgcn_bilstm.py trong cùng thư mục với app.py)
from model_stgcn_bilstm import STGCN_BiLSTM

# Thiết lập logging
LOGGER.setLevel(logging.ERROR)

# === SETTINGS ===
MODEL_PATH = "models/stgcn_bilstm_model.pth"
LABEL_MAP_PATH = "data/label_mapping.pkl"
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# === COCO Skeleton ===
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
]

# === ACTION MAPPING ===
ACTION_NAME_MAP = {
    (0, 4): "Walking", (0, 21): "Transitioning State", (0, 28): "Laying down", (0, 31): "Accessing Tool",
    (0, 33): "Enter working area", (0, 34): "Leaving work area", (0, 35): "Carrying Object",
    (0, 36): "Work in progress", (0, 37): "Falling", (22, 31): "Picking up Object",
    (23, 31): "Putting down Object", (24, 31): "Operating Machine"
}

# === HELPER FUNCTIONS ===
def create_edge_index(skeleton, num_keypoints=17):
    adj = np.zeros((num_keypoints, num_keypoints))
    for i, j in skeleton:
        adj[i, j] = 1
        adj[j, i] = 1
    edge_index = np.array(np.where(adj))
    return torch.tensor(edge_index, dtype=torch.long)

def draw_skeleton(frame, keypoints, connections, bbox=None, person_id="1", action_label="Unknown"):
    for start_idx, end_idx in connections:
        if keypoints[start_idx, 2] > 0.1 and keypoints[end_idx, 2] > 0.1:
            p1 = tuple(map(int, keypoints[start_idx, :2]))
            p2 = tuple(map(int, keypoints[end_idx, :2]))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for x, y, c in keypoints:
        if c > 0.1:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    if bbox is not None:
        x_min, y_min, w, h = bbox
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_min + w), int(y_min + h)), (255, 0, 0), 2)
        label = f"ID: {person_id}, Action: {action_label}"
        cv2.putText(frame, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def normalize_keypoints(keypoints, bbox):
    x_min, y_min, width, height = bbox
    normalized = []
    for x, y, c in keypoints:
        x_norm = (x - x_min) / width if width > 1 else x / IMAGE_WIDTH
        y_norm = (y - y_min) / height if height > 1 else y / IMAGE_HEIGHT
        normalized.append([x_norm, y_norm, c])
    return np.array(normalized)

def assign_id(bbox, prev_centroids, max_distance=100):
    global next_id
    centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if not prev_centroids:
        prev_centroids.append((next_id, centroid))
        return str(next_id), prev_centroids
    distances = [distance.euclidean(centroid, prev[1]) for prev in prev_centroids]
    if min(distances) < max_distance:
        idx = np.argmin(distances)
        prev_centroids[idx] = (prev_centroids[idx][0], centroid)
        return str(prev_centroids[idx][0]), prev_centroids
    prev_centroids.append((next_id, centroid))
    next_id += 1
    return str(next_id - 1), prev_centroids

def send_email(subject, body, to_email):
    from_email = "your_email@gmail.com"  # Thay bằng email của bạn
    password = "your_password"  # Thay bằng mật khẩu hoặc mật khẩu ứng dụng

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# === MODEL LOAD ===
yolo_model = YOLO("yolov8l.pt", task='detect').to(DEVICE)
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(DEVICE)
pose_model.eval()

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)
action_to_name = {v: ACTION_NAME_MAP.get(k, "Unknown") for k, v in label_map.items()}
label_list = sorted(set(label_map.values()))

edge_index = create_edge_index(SKELETON_CONNECTIONS).to(DEVICE)
model = STGCN_BiLSTM(num_keypoints=NUM_KEYPOINTS, num_classes=len(label_list), edge_index=edge_index).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === FLASK APP SETUP ===
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
camera = None
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

users = {}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Từ điển ngôn ngữ (đã sửa lỗi cú pháp)
LANGUAGES = {
    'english': {
        'title': 'AI Worker Behavior Recognition',
        'sidebar_title': 'AI Worker',
        'dashboard': 'Dashboard',
        'live_camera': 'Live Camera',
        'upload_video': 'Upload Video',
        'event_history': 'Event History',
        'settings': 'Settings',
        'welcome': 'Welcome',
        'logout': 'Logout',
        'login': 'Login',
        'notifications': 'Notifications',
        'no_hazardous': 'No hazardous behaviors detected.',
        'clear_notifications': 'Clear Notifications',
        'new_hazardous': 'New hazardous behavior detected! at',
        'status': 'STATUS',
        'recognition': 'RECOGNITION',
        'start': 'Start',
        'pause': 'Pause',
        'export': 'Export',
        'camera_display': 'Camera Display',
        'select': 'Select',
        'refresh': 'Refresh',
        'on_camera': 'On Camera',
        'off_camera': 'Off Camera',
        'detection_controls': 'Detection Controls',
        'start_detection': 'Start Detection',
        'stop_detection': 'Stop Detection',
        'no_camera_or_detection': 'No camera or detection active.',
        'detected_behaviors': 'Detected Behaviors',
        'falling': 'Falling',
        'lay_down': 'Lay down',
        'last_detected': 'Last detected',
        'camera_id': 'Camera',
        'is_not_available_now': 'is not available now.',
        'is_on': 'is On',
        'is_off': 'is Off',
        'detection_is_off': 'Detection is Off',
        'detection_is_on': 'Detection is On',
        'detection_is_starting': 'Detection is Starting',
        'detection_is_stopping': 'Detection is Stopping',
        'refreshing_camera': 'Refreshing Camera',
        'feed': 'feed',
        'behavior_detected': 'Behavior detected',
        'detected_video': 'Detected Video',
        'start_detection_upload': 'Start Detection',
        'please_select_video': 'Please select a video file to upload.',
        'error_during_detection': 'Error during detection',
        'no_log_data': 'No log data to save.',
        'save_log_to_excel': 'Save Log to Excel',
        'clear_log': 'Clear Log',
        'event_history': 'Event History',
        'camera_detection_log': 'Camera Detection Log',
        'time': 'Time',
        'behavior': 'Behavior',
        'camera_id_label': 'Camera ID',
        'accuracy': 'Accuracy',
        'username_email': 'Username',
        'password': 'Password',
        'remember_me': 'Remember me',
        'forgot_password': 'Forgot password?',
        'or': 'or',
        'continue_with_google': 'Continue with Google',
        'continue_with_microsoft': 'Continue with Microsoft Account',
        'dont_have_account': "Don't have an account?",
        'sign_up': 'Sign up',
        'invalid_credentials': 'Invalid username or password',
        'username_exists': 'Username already exists',
        'settings_title': 'Settings',
        'alert_notification': 'Alert & Notification',
        'sound_alert': 'Sound Alert',
        'email_notif': 'Notification Email',
        'camera_video': 'Camera & Video',
        'auto_record': 'Auto Record',
        'account': 'Account',
        'username': 'Username',
        'change_password': 'Change Password',
        'language_save': 'Language & Save Option',
        'language': 'Language',
        'data_retention': 'Data Retention Period',
        'save_settings': 'Save Settings',
        'english': 'English',
        'korean': 'Korean',
        '3_days': '3 Days',
        '7_days': '7 Days',
        '30_days': '30 Days',
        'settings_saved': 'Settings have been saved!',
        'change_password_not_implemented': 'Change Password feature is not implemented yet. Please implement the logic to change the password.',
        'hazard_detected': 'Hazard Detected!',
        'dismiss': 'Dismiss',
    },
    'korean': {
        'title': 'AI 작업자 행동 인식',
        'sidebar_title': 'AI 작업자',
        'dashboard': '대시보드',
        'live_camera': '라이브 카메라',
        'upload_video': '비디오 업로드',
        'event_history': '이벤트 기록',
        'settings': '설정',
        'welcome': '환영합니다',
        'logout': '로그아웃',
        'login': '로그인',
        'notifications': '알림',
        'no_hazardous': '위험 행동이 감지되지 않았습니다.',
        'clear_notifications': '알림 지우기',
        'new_hazardous': '새로운 위험 행동이 감지되었습니다! 시간:',
        'status': '상태',
        'recognition': '인식',
        'start': '시작',
        'pause': '일시 정지',
        'export': '내보내기',
        'camera_display': '카메라 디스플레이',
        'select': '선택',
        'refresh': '새로고침',
        'on_camera': '카메라 켜기',
        'off_camera': '카메라 끄기',
        'detection_controls': '탐지 제어',
        'start_detection': '탐지 시작',
        'stop_detection': '탐지 중지',
        'no_camera_or_detection': '카메라 또는 탐지가 활성화되지 않았습니다.',
        'detected_behaviors': '탐지된 행동',
        'falling': '떨어짐',
        'lay_down': '누움',
        'last_detected': '최근 탐지',
        'camera_id': '카메라',
        'is_not_available_now': '을 사용할 수 없습니다.',
        'is_on': '켜짐',
        'is_off': '꺼짐',
        'detection_is_off': '탐지가 꺼져 있습니다.',
        'detection_is_on': '탐지가 켜져 있습니다.',
        'detection_is_starting': '탐지가 시작 중입니다.',
        'detection_is_stopping': '탐지가 중지 중입니다.',
        'refreshing_camera': '카메라 새로고침 중',
        'feed': '피드',
        'behavior_detected': '행동이 감지되었습니다',
        'detected_video': '탐지된 비디오',
        'start_detection_upload': '탐지 시작',
        'please_select_video': '업로드할 비디오 파일을 선택해 주세요.',
        'error_during_detection': '탐지 중 오류가 발생했습니다',
        'no_log_data': '저장할 로그 데이터가 없습니다.',
        'save_log_to_excel': '로그를 엑셀로 저장',
        'clear_log': '로그 지우기',
        'event_history': '이벤트 기록',
        'camera_detection_log': '카메라 탐지 로그',
        'time': '시간',
        'behavior': '행동',
        'camera_id_label': '카메라 ID',
        'accuracy': '정확도',
        'username_email': '사용자 이름',
        'password': '비밀번호',
        'remember_me': '로그인 유지',
        'forgot_password': '비밀번호를 잊으셨나요?',
        'or': '또는',
        'continue_with_google': 'Google로 계속',
        'continue_with_microsoft': 'Microsoft 계정으로 계속',
        'dont_have_account': '계정이 없으신가요?',
        'sign_up': '회원 가입',
        'invalid_credentials': '잘못된 사용자 이름 또는 비밀번호',
        'username_exists': '사용자 이름이 이미 존재합니다',
        'settings_title': '설정',
        'alert_notification': '알림 및 경고',
        'sound_alert': '소리 경고',
        'email_notif': '알림 이메일',
        'camera_video': '카메라 및 비디오',
        'auto_record': '자동 녹화',
        'account': '계정',
        'username': '사용자 이름',
        'change_password': '비밀번호 변경',
        'language_save': '언어 및 저장 옵션',
        'language': '언어',
        'data_retention': '데이터 보관 기간',
        'save_settings': '설정 저장',
        'english': '영어',
        'korean': '한국어',
        '3_days': '3일',
        '7_days': '7일',
        '30_days': '30일',
        'settings_saved': '설정이 저장되었습니다!',
        'change_password_not_implemented': '비밀번호 변경 기능은 아직 구현되지 않았습니다. 비밀번호 변경 로직을 triển khai해 주세요.',
        'hazard_detected': '위험 감지됨!',
        'dismiss': '닫기',
    }
}

# Khởi tạo biến toàn cục cho model
sequence_buffer = []
prev_centroids = []
next_id = 1
detection_running = False

def get_language():
    return session.get('language', 'english')

@app.route('/')
def index():
    if not session.get('username'):
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if not session.get('username'):
        return redirect(url_for('login'))
    language = get_language()
    return render_template("dashboard_layout.html", language=LANGUAGES[language])

@app.route('/live_camera')
def live_camera():
    if not session.get('username'):
        return redirect(url_for('login'))
    language = get_language()
    return render_template("live_camera.html", language=LANGUAGES[language])

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global camera
    if not session.get('username'):
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    camera_id = int(request.form['camera_id'])
    temp = cv2.VideoCapture(camera_id)
    if not temp.isOpened():
        return jsonify({'success': False, 'message': 'This camera is not available now'})
    camera = temp
    return jsonify({'success': True})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global camera, sequence_buffer, prev_centroids, next_id
    if not session.get('username'):
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    if camera:
        camera.release()
        camera = None
        sequence_buffer = []  # Reset buffer
        prev_centroids = []
        next_id = 1
    return jsonify({'success': True})

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_running
    if not session.get('username'):
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    detection_running = True
    return jsonify({'success': True})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running
    if not session.get('username'):
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    detection_running = False
    return jsonify({'success': True})

def gen_frames():
    global camera, sequence_buffer, prev_centroids, next_id, detection_running
    while True:
        if camera:
            success, frame = camera.read()
            if not success:
                break

            if detection_running:
                # Xử lý khung hình với model nhận diện hành vi
                rgb = frame[:, :, ::-1]
                results = yolo_model(rgb)[0]

                person_boxes_xyxy = [box[:4].tolist() for box in results.boxes.data if int(box[5]) == 0]
                if person_boxes_xyxy:
                    person_boxes_xywh = np.array(person_boxes_xyxy)
                    person_boxes_xywh[:, 2] -= person_boxes_xywh[:, 0]
                    person_boxes_xywh[:, 3] -= person_boxes_xywh[:, 1]

                    inputs = pose_processor(rgb, boxes=[person_boxes_xywh], return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        outputs = pose_model(**inputs)
                        results_pose = pose_processor.post_process_pose_estimation(outputs, boxes=[person_boxes_xywh])[0]

                    for i, pose_result in enumerate(results_pose):
                        keypoints = pose_result["keypoints"].cpu().numpy()
                        scores = pose_result["scores"].cpu().numpy()
                        keypoints_with_conf = np.concatenate([keypoints, scores[..., None]], axis=-1)
                        bbox = person_boxes_xywh[i]
                        person_id, prev_centroids = assign_id(bbox, prev_centroids)
                        norm_keypoints = normalize_keypoints(keypoints_with_conf, bbox)
                        sequence_buffer.append(norm_keypoints.flatten())

                        action_label = "Unknown"
                        if len(sequence_buffer) == SEQUENCE_LENGTH:
                            input_data = np.array(sequence_buffer).reshape(1, SEQUENCE_LENGTH, NUM_KEYPOINTS * 3)
                            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
                            with torch.no_grad():
                                prediction = model(input_tensor)
                                class_id = prediction.argmax(dim=1).item()
                                action_label = action_to_name.get(label_list[class_id], "Unknown")
                            sequence_buffer.pop(0)

                            # Gửi thông báo nếu phát hiện hành vi bất thường
                            if action_label in ["Falling", "Laying down"]:
                                to_email = session.get('email_notif', 'default@example.com')
                                if to_email:
                                    subject = "Hazardous Behavior Detected!"
                                    body = f"Warning: {action_label} detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!"
                                    send_email(subject, body, to_email)

                        frame = draw_skeleton(frame, keypoints_with_conf, SKELETON_CONNECTIONS, bbox, person_id, action_label)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video')
def upload_video():
    if not session.get('username'):
        return redirect(url_for('login'))
    language = get_language()
    return render_template('upload_video.html', language=LANGUAGES[language])

@app.route('/upload_and_detect', methods=['POST'])
def upload_and_detect():
    if not session.get('username'):
        return redirect(url_for('login'))

    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video file'}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        detected_filename = f"detected_{filename}"
        detected_filepath = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
        out = cv2.VideoWriter(detected_filepath, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return jsonify({'error': 'Failed to create output video file (codec issue)'}), 500

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, "Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)

        cap.release()
        out.release()

        if not os.path.exists(detected_filepath):
            return jsonify({'error': 'Failed to save detected video'}), 500

        video_url = url_for('static', filename='uploads/' + detected_filename, _external=True)
        logs = [
            {'behavior': 'Detected', 'accuracy': 95, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ]
        return jsonify({'video_url': video_url, 'logs': logs})

@app.route('/event_history')
def event_history():
    if not session.get('username'):
        return redirect(url_for('login'))
    language = get_language()
    return render_template('event_history.html', language=LANGUAGES[language])

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        session['sound_alert'] = request.form.get('sound_alert') == 'true'
        session['email_notif'] = request.form.get('email_notif', 'user@example.com')
        session['auto_record'] = request.form.get('auto_record') == 'true'
        session['language'] = request.form.get('language', 'english')
        session['data_retention'] = request.form.get('data_retention', '7')
        flash('Settings updated successfully!')
        return redirect(url_for('settings'))
    
    return render_template('settings.html', language=LANGUAGES[session.get('language', 'english')])

@app.route('/login', methods=['GET', 'POST'])
def login():
    language = get_language()
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = 'remember' in request.form

        if username in users and users[username] == password:
            session['username'] = username
            if remember:
                session.permanent = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password", language=LANGUAGES[language])

    return render_template('login.html', language=LANGUAGES[language])

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    language = get_language()
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            return render_template('signup.html', error="Username already exists", language=LANGUAGES[language])
        users[username] = password
        session['username'] = username
        return redirect(url_for('dashboard'))

    return render_template('signup.html', language=LANGUAGES[language])

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    session.pop('language', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)