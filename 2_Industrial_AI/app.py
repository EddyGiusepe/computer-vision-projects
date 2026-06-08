import math
import io
import os
import struct
import time
import warnings
import wave

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import cv2
import cvzone
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_NAME = "yolo26x.pt" # "yolov8s.pt"
CAMERA_INDEXES = [0, 1]
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PERSON_CONFIDENCE_THRESHOLD = 0.50
WARNING_TRIGGER_SECONDS = 1.5
DANGER_TRIGGER_SECONDS = 4.0
CLEAR_ALERT_SECONDS = 1.0
MAX_CONSECUTIVE_READ_FAILURES = 10
ALARM_FILENAMES = ["alarm.mp3", "alarm.wav", "alarm.ogg", "alarm.mpeg"]
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]

SAFE_STATE = "safe"
WARNING_STATE = "warning"
DANGER_STATE = "danger"


def open_camera(camera_indexes):
    for camera_index in camera_indexes:
        device_path = f"/dev/video{camera_index}"
        if not os.path.exists(device_path):
            continue

        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            continue

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return camera, camera_index

    return None, None


def get_alert_state(elapsed_time):
    if elapsed_time >= DANGER_TRIGGER_SECONDS:
        return DANGER_STATE
    if elapsed_time >= WARNING_TRIGGER_SECONDS:
        return WARNING_STATE
    return SAFE_STATE


def find_alarm_file():
    search_directories = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)

    for directory in (current_dir, project_dir, os.getcwd()):
        if directory not in search_directories:
            search_directories.append(directory)

    for directory in search_directories:
        for filename in ALARM_FILENAMES:
            candidate = os.path.join(directory, filename)
            if os.path.exists(candidate):
                return candidate

    return None


def create_fallback_alarm_sound():
    sample_rate = 22050
    duration_seconds = 0.45
    amplitude = 12000
    frequencies = (880, 660)
    total_samples = int(sample_rate * duration_seconds)

    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            frames = bytearray()
            for sample_index in range(total_samples):
                time_position = sample_index / sample_rate
                frequency = frequencies[(sample_index // (sample_rate // 6)) % len(frequencies)]
                sample_value = int(
                    amplitude * math.sin(2 * math.pi * frequency * time_position)
                )
                frames.extend(struct.pack("<h", sample_value))

            wav_file.writeframes(bytes(frames))

        wav_buffer.seek(0)
        return pygame.mixer.Sound(file=wav_buffer)


def load_pillow_font(size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_font_candidates = [
        os.path.join(current_dir, "assets", "fonts", "DejaVuSans-Bold.ttf"),
        os.path.join(current_dir, "assets", "fonts", "DejaVuSans.ttf"),
    ]

    for font_path in [*local_font_candidates, *FONT_CANDIDATES]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue

    return ImageFont.load_default()


def get_text_size(font, text):
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def clamp_point(point, frame_width, frame_height):
    x, y = point
    return max(0, min(x, frame_width - 1)), max(0, min(y, frame_height - 1))


def add_text_overlay(
    overlays,
    text,
    position,
    font,
    text_color,
    background_color=None,
    padding=(12, 8),
):
    overlays.append(
        {
            "text": text,
            "position": position,
            "font": font,
            "text_color": text_color,
            "background_color": background_color,
            "padding": padding,
        }
    )


def draw_label_background(img, overlay):
    text = overlay["text"]
    font = overlay["font"]
    position = overlay["position"]
    background_color = overlay["background_color"]
    padding_x, padding_y = overlay["padding"]
    text_width, text_height = get_text_size(font, text)
    frame_height, frame_width = img.shape[:2]

    x, y = clamp_point(position, frame_width, frame_height)
    x2 = min(frame_width - 1, x + text_width + (padding_x * 2))
    y2 = min(frame_height - 1, y + text_height + (padding_y * 2))

    cv2.rectangle(img, (x, y), (x2, y2), background_color, -1)
    overlay["text_position"] = (x + padding_x, y + padding_y)


def render_text_overlays(img, overlays):
    if not overlays:
        return img

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)
    frame_height, frame_width = img.shape[:2]

    for overlay in overlays:
        x, y = clamp_point(overlay["position"], frame_width, frame_height)
        text_position = overlay.get("text_position", (x, y))
        draw.text(
            text_position,
            overlay["text"],
            font=overlay["font"],
            fill=overlay["text_color"],
        )

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


cap, active_camera_index = open_camera(CAMERA_INDEXES)
if cap is None:
    checked_indexes = ", ".join(str(index) for index in CAMERA_INDEXES)
    raise SystemExit(
        f"Unable to open any configured camera. Checked indexes: {checked_indexes}."
    )

model = YOLO(MODEL_NAME)
LABEL_FONT = load_pillow_font(size=28)
BANNER_FONT = load_pillow_font(size=30)

# --- DETECTORS ---
face_cascade_smart = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)
face_cascade_profile = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# --- AUDIO SETUP ---
audio_available = False
alarm_file_path = find_alarm_file()
alarm_sound_mode = None
fallback_alarm_sound = None
try:
    pygame.mixer.init()
    if alarm_file_path:
        pygame.mixer.music.load(alarm_file_path)
        audio_available = True
        alarm_sound_mode = "music"
    else:
        fallback_alarm_sound = create_fallback_alarm_sound()
        audio_available = True
        alarm_sound_mode = "fallback"
except pygame.error as exc:
    if alarm_file_path:
        try:
            fallback_alarm_sound = create_fallback_alarm_sound()
            audio_available = True
            alarm_sound_mode = "fallback"
        except pygame.error:
            print(f"Audio alarm unavailable: {exc}")
    else:
        print(f"Audio alarm unavailable: {exc}")

# --- ALERT STATE ---
anomaly_start_time = None
anomaly_clear_start_time = None
alert_state = SAFE_STATE
held_alert_box = None
consecutive_read_failures = 0

print(
    "Starting PAD System "
    f"(camera {active_camera_index}, warning={WARNING_TRIGGER_SECONDS:.1f}s, "
    f"danger={DANGER_TRIGGER_SECONDS:.1f}s, clear={CLEAR_ALERT_SECONDS:.1f}s)..."
)

while True:
    success, img = cap.read()
    if not success:
        consecutive_read_failures += 1
        if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
            print(
                "Camera read failed repeatedly. "
                "Check the selected device or try another camera index."
            )
            break
        continue

    consecutive_read_failures = 0
    now = time.time()
    results = model(img, stream=True, classes=[0], verbose=False)

    current_frame_has_anomaly = False
    current_frame_box = None
    current_anomaly_box = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3]),
            )
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf <= PERSON_CONFIDENCE_THRESHOLD:
                continue

            if current_frame_box is None:
                current_frame_box = (x1, y1, w, h)

            y1_c, y2_c = max(0, y1), min(img.shape[0], y2)
            x1_c, x2_c = max(0, x1), min(img.shape[1], x2)
            person_roi = img[y1_c:y2_c, x1_c:x2_c]

            if person_roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

            faces_front = face_cascade_smart.detectMultiScale(
                gray_roi, 1.1, 5, minSize=(50, 50)
            )
            faces_profile = face_cascade_profile.detectMultiScale(
                gray_roi, 1.1, 5, minSize=(50, 50)
            )
            flipped_gray = cv2.flip(gray_roi, 1)
            faces_profile_flipped = face_cascade_profile.detectMultiScale(
                flipped_gray, 1.1, 5, minSize=(50, 50)
            )

            is_face_visible = (
                len(faces_front) > 0
                or len(faces_profile) > 0
                or len(faces_profile_flipped) > 0
            )

            if not is_face_visible:
                current_frame_has_anomaly = True
                if current_anomaly_box is None:
                    current_anomaly_box = (x1, y1, w, h)

    if current_frame_has_anomaly:
        if anomaly_start_time is None:
            anomaly_start_time = now
        anomaly_clear_start_time = None
        held_alert_box = current_anomaly_box or held_alert_box
        elapsed_time = now - anomaly_start_time
    elif anomaly_start_time is not None:
        if anomaly_clear_start_time is None:
            anomaly_clear_start_time = now

        elapsed_time = anomaly_clear_start_time - anomaly_start_time
        clear_elapsed = now - anomaly_clear_start_time

        if clear_elapsed >= CLEAR_ALERT_SECONDS:
            anomaly_start_time = None
            anomaly_clear_start_time = None
            held_alert_box = None
            elapsed_time = 0.0
    else:
        elapsed_time = 0.0

    alert_state = get_alert_state(elapsed_time)
    alarm_active = alert_state == DANGER_STATE
    text_overlays = []

    if alert_state == DANGER_STATE:
        target_box = current_anomaly_box or held_alert_box
        if target_box:
            x1, y1, w, h = target_box
            cvzone.cornerRect(
                img,
                (x1, y1, w, h),
                l=30,
                t=5,
                rt=1,
                colorR=(0, 0, 255),
                colorC=(0, 0, 255),
            )
            add_text_overlay(
                text_overlays,
                text="VIOLAÇÃO DE SEGURANÇA",
                position=(max(0, x1), max(35, y1)),
                font=LABEL_FONT,
                text_color=(255, 255, 255),
                background_color=(0, 0, 255),
                padding=(12, 8),
            )
            draw_label_background(img, text_overlays[-1])

        cv2.rectangle(img, (0, 0), (FRAME_WIDTH, 50), (0, 0, 255), -1)
        add_text_overlay(
            text_overlays,
            text="ALARME: ROSTO COBERTO DETECTADO",
            position=(300, 10),
            font=BANNER_FONT,
            text_color=(255, 255, 255),
        )
    elif alert_state == WARNING_STATE:
        target_box = current_anomaly_box or held_alert_box
        time_left = max(0, math.ceil(DANGER_TRIGGER_SECONDS - elapsed_time))

        if target_box:
            x1, y1, w, h = target_box
            cvzone.cornerRect(
                img,
                (x1, y1, w, h),
                l=30,
                t=5,
                rt=1,
                colorR=(0, 255, 255),
                colorC=(0, 255, 255),
            )
            add_text_overlay(
                text_overlays,
                text=f"ATENÇÃO: Mostre o rosto ({time_left}s)",
                position=(max(0, x1), max(35, y1)),
                font=LABEL_FONT,
                text_color=(0, 0, 0),
                background_color=(0, 255, 255),
                padding=(12, 8),
            )
            draw_label_background(img, text_overlays[-1])

        cv2.rectangle(img, (0, 0), (FRAME_WIDTH, 50), (0, 255, 255), -1)
        add_text_overlay(
            text_overlays,
            text=f"ATENÇÃO: Mostre o rosto ({time_left}s)",
            position=(300, 10),
            font=BANNER_FONT,
            text_color=(0, 0, 0),
        )
    else:
        if current_frame_box:
            x1, y1, w, h = current_frame_box
            cvzone.cornerRect(
                img,
                (x1, y1, w, h),
                l=30,
                t=5,
                rt=1,
                colorR=(0, 255, 0),
                colorC=(0, 255, 0),
            )
            add_text_overlay(
                text_overlays,
                text="SEGURO: Rosto verificado",
                position=(max(0, x1), max(35, y1)),
                font=LABEL_FONT,
                text_color=(255, 255, 255),
                background_color=(0, 255, 0),
                padding=(12, 8),
            )
            draw_label_background(img, text_overlays[-1])

    if audio_available:
        if alarm_active:
            if alarm_sound_mode == "music":
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
                    print("Alarm sounding.")
            elif alarm_sound_mode == "fallback" and fallback_alarm_sound is not None:
                if pygame.mixer.get_busy() is False:
                    fallback_alarm_sound.play(loops=-1)
                    print("Alarm sounding.")
        elif alarm_sound_mode == "music":
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        elif alarm_sound_mode == "fallback":
            pygame.mixer.stop()

    img = render_text_overlays(img, text_overlays)
    cv2.imshow("Vigilancia PAD - Logica Temporal", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
