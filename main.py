import os
import sys

# Base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Redirect sys.stderr and sys.stdout if they are None
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')

# Configure logging to a file
import logging

log_file = os.path.join(base_dir, 'app.log')
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_file,
    filemode='w',  # Overwrite the log file on each run
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure Kivy window properties using Config.set before importing Kivy modules
from kivy.config import Config
Config.set('graphics', 'width', '750')
Config.set('graphics', 'height', '820')
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'position', 'custom')
Config.set('graphics', 'top', '50')
Config.set('graphics', 'left', '50')

# Set the icon using Config
icon = os.path.join(base_dir, "Static", "icon", "cctv.png")
Config.set('kivy', 'window_icon', icon)

# Now import Kivy modules
from kivy.app import App
from kivy.core.window import Window

# Configure Kivy's Logger to write to a file
from kivy.logger import Logger
Logger.handlers = []  # Remove default handlers

from logging import FileHandler
kivy_log_file = os.path.join(base_dir, 'kivy.log')
file_handler = FileHandler(kivy_log_file, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
Logger.addHandler(file_handler)

# Set the window to always be on top (if needed)
Window.topmost = True
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture

# ... rest of your imports (non-Kivy modules)
from Static.extra.license_detect import ObjectDetector as Ort
import threading
import cv2
from datetime import datetime
import queue
import time
import pyodbc
import requests
import logging

# The rest of your code...

# Global variables for RTSP URLs and license address
rtsp_urls = {}
license_address = ""
entry_door = ""
exit1_door = ""
exit2_door = ""
video_streams = {}
latest_detections = []
door_status = {
    'Entry': False,
    'Exit_1': False,
    'Exit_2': False
}

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
os.environ["OPENCV_FFMPEG_DEBUG"] = "1"

base_dir = os.path.dirname(__file__)

# File paths using os.path.join
icon = os.path.join(base_dir,"Static","icon","cctv.ico")

# Set the icon using Config before Kivy initializes


onnx_model1 = os.path.join(base_dir, "Static", "models", "plate.onnx")
onnx_model2 = os.path.join(base_dir, "Static", "models", "yolov8m_text.onnx")

class_names = ['-']
class_names2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

plate_detector = Ort(onnx_model1, class_names, conf_thres=0.6, iou_thres=0.8)
text_detector = Ort(onnx_model2, class_names2, conf_thres=0.65, iou_thres=0.8)

detected_classes = []
detected_classes_lock = threading.Lock()

print(f"Icon path: {icon}")
print(f"Does icon exist? {os.path.exists(icon)}")

# Define VideoCapture class to handle video capture and detection
class VideoCapture:
    def __init__(self, rtsp_url, license_address, timeout=5):  # Add door_control
        self.camera_name = self.get_key_from_value(rtsp_urls, rtsp_url)
        self.plate_detector = plate_detector
        self.text_detector = text_detector
        self.license_address = license_address
        self.rtsp_url = rtsp_url
        self.cap = None
        self.q = queue.Queue()
        self.running = True
        self.timeout = timeout  # Timeout for connection

        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def get_key_from_value(self, dictionary, search_value):
        try:
            for key, value in dictionary.items():
                if value == search_value:
                    return key
            return None
        except Exception as e:
            logging.error(f"Error in get_key_from_value: {e}")
            return None

    def _reader(self):
        start_time = time.time()  # Start time for timeout
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            logging.debug(f"Attempting to connect to {self.rtsp_url}")

            while not self.cap.isOpened():
                if time.time() - start_time > self.timeout:
                    logging.error(f"Timeout: Could not connect to {self.rtsp_url} within {self.timeout} seconds.")
                    return

                time.sleep(1)

            logging.debug(f"Connected to {self.rtsp_url}")

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error(f"Failed to read frame from {self.rtsp_url}.")
                    return

                if not self.q.empty():
                    self.q.get_nowait()  # Remove the previous frame if queue is full
                self.q.put(frame)

        except Exception as e:
            logging.error(f"Error initializing VideoCapture for {self.rtsp_url}: {e}")
        finally:
            if self.cap is not None:
                self.cap.release()
                logging.debug(f"VideoCapture for {self.rtsp_url} released")

    def read(self):
        try:
            return self.q.get()
        except Exception as e:
            logging.error(f"Error in read: {e}")
            return None

    def stop(self):
        self.running = False  # Signal to stop the loop in the thread
        if self.cap is not None:
            self.cap.release()

    # Perform object detection on the frame
    def detect_objects(self, frame):
        try:
            # Run plate and text detection on the frame
            frame = self._run_plate_detection(frame)
        except Exception as e:
            logging.error(f"Error in detect_objects: {e}")

        return frame

    # License plate and text detection logic
    def _run_plate_detection(self, frame):
        try:
            # Get frame dimensions and define region of interest (ROI)
            height, width, _ = frame.shape
            roi_x, roi_y, roi_width, roi_height = width // 4, height // 4, width // 2, height // 2

            # Crop the frame to focus on the region of interest
            cropped_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 1)

            # Detect plates in the cropped frame
            boxes, scores, class_ids = self.plate_detector(cropped_frame)

            for j, box in enumerate(boxes):
                x, y, w, h = box
                # Draw rectangle around detected plate
                cv2.rectangle(frame, (int(x) + roi_x, int(y) + roi_y), (int(x + w) + roi_x, int(y + h) + roi_y), (0, 255, 0), 1)

                # Extract the plate image from the cropped frame
                plate_image = cropped_frame[max(0, int(y - 10)):int(h + 10), max(0, int(x - 10)):int(w + 10)]
                if plate_image.size == 0:
                    logging.error("Empty plate image, skipping detection.")
                    continue

                # Convert the plate image to grayscale for text detection
                try:
                    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                    # Detect text on the plate
                    boxes_text, scores_text, class_ids_text = self.text_detector(plate_gray)
                    sorted_preds = sorted(zip(boxes_text, class_ids_text), key=lambda x: x[0][0])
                    license_number = ''.join([class_names2[class_id] for _, class_id in sorted_preds])

                    # Process and store detection if the license number is not already detected
                    with detected_classes_lock:
                        if license_number not in detected_classes:
                            detected_classes.append(license_number)
                            detection_time = datetime.now().strftime("%I:%M %p %d:%m:%Y")
                            logging.info(f"Detected plate: {license_number}")

                            # Add detection data to latest detections
                            camera_name = {0: "Entry", 1: "Exit_1", 2: "Exit_2"}.get(self.camera_name, "")
                            data = {
                                'Carno': license_number,
                                'channel_state': camera_name,
                                'timestamp': detection_time
                            }
                            latest_detections.append(data)

                            # Limit to last 5 detections
                            if len(latest_detections) > 1:
                                latest_detections.pop(0)

                            # Save the plate image and call controling logic
                            save_dir = os.path.join("Static", "predict")
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{license_number}_{detection_time}.jpg")
                            cv2.putText(plate_gray, license_number, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            # Use the door_control instance to call the control logic
                            headers = {'Content-Type': 'application/json'}
                            threading.Thread(target=controling, args=(save_path, plate_gray, self.license_address, data, headers)).start()
                            detected_classes.clear()

                except Exception as e:
                    logging.error(f"Failed to process plate detection: {e}")

        except Exception as e:
            logging.error(f"Error in plate detection: {e}")

        return frame

def connection():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"        # Use localhost as the server name
        "DATABASE=license_num;"     # Database name
        "Trusted_Connection=yes;"   # Use Windows Authentication
    )
    return pyodbc.connect(conn_str)

def controling(save_path, plate_gray, addr, data, headers):
    """
    Function to handle the logic when a license plate is detected.
    It checks the license in the database and sends an open signal to the appropriate door.
    """
    url_map = {
        'Entry': entry_door,
        'Exit_1': exit1_door,
        'Exit_2': exit2_door
    }

    channel = data['channel_state']
    try:
        # Connect to the database
        conn = connection()
        cursor = conn.cursor()

        license_number = data['Carno']
        query = "SELECT COUNT(1) FROM license WHERE license_num = ?"
        cursor.execute(query, (license_number,))
        result = cursor.fetchone()

        # If the door is already open, we skip further actions
        if door_status.get(channel):
            print(f"{channel} gate is already open. Skipping further actions.")
            return
        else:
            if result[0] == 1:
                # License number exists in the database, proceed with API posting
                print(f"License {license_number} exists in the database. Posting to API.")
                door_open(data, url_map, channel)
                response = requests.post(addr, json=data, headers=headers)
                response.raise_for_status()
            else:
                print(f"License {license_number} not found in the database.")

        cursor.close()
        conn.close()

    except pyodbc.Error as e:
        print(f"Database error: {e}")
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def door_open(data, url_map, channel):
    """
    Function to send a signal to open the door.
    """
    global door_status  # Use the global door_status variable to track door state
    try:
        open_signal = {'open': '1'}  # The signal used to open the gate

        # Mark the door as open in the door_status dictionary
        door_status[channel] = True
        print(f"Sending open signal to {channel} gate.")
        
        # Create a new thread to send the door signal
        threading.Thread(target=send_door_signal, args=(url_map[channel], open_signal, channel)).start()

    except requests.RequestException as e:
        print(f"Error in sending signal to {channel}: {e}")

def send_door_signal(door_url, open_signal, channel_state):
    """
    Function to send the signal to open the door and reset the door status after 10 seconds.
    """
    global door_status  # Access the global door_status
    try:
        # Send the open signal to the door URL
        response = requests.post(door_url, json=open_signal, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        print(f"{channel_state} gate opened.")

        # Wait for 10 seconds before resetting the door status
        if response.status_code == 200:
            for remaining in range(10, 0, -1):
                print(f"{channel_state}: Waiting... {remaining} seconds remaining", end="\r")
                time.sleep(1)
            print(f"\n{channel_state}: Ready for the next signal.")

        # Reset the door status after waiting
        door_status[channel_state] = False

    except requests.RequestException as e:
        print(f"Error in sending signal to {channel_state}: {e}")
        door_status[channel_state] = False


class StreamScreen(Screen):
    def __init__(self, **kwargs):
        super(StreamScreen, self).__init__(**kwargs)
        
        self.frame_counter = 0
        # Main layout (BoxLayout with vertical orientation)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Top layout (GridLayout with 2 columns: one for the button, one for the label)
        self.top_layout = GridLayout(cols=2, size_hint=(1, 0.1), padding=10, spacing=10)
        
        # Add a button to go to the settings page
        self.settings_button = Button(text="Go to Settings", size_hint=(None, None), size=(200, 50), pos_hint={'x': 0.2, 'center_y': 0.5})
        self.settings_button.bind(on_press=self.go_to_settings)
        self.top_layout.add_widget(self.settings_button)

        # Info label placed on the top right of the layout
        self.info_label = Label(
            text="",
            size_hint=(1, 1),  # Make sure it fills the available space in the second column
            #pos_hint={'x': 1, 'center_y': 0.5}
            halign="right",      # Horizontal alignment
            #valign="top"        # Vertical alignment
        )
        self.info_label.bind(size=self.info_label.setter('text_size'))  # Ensure proper text wrapping and alignment
        self.top_layout.add_widget(self.info_label)
        
        # Stream layout (for video streams, 2 columns)
        self.stream_layout = GridLayout(cols=2, padding=10, spacing=10, size_hint=(1, 0.7))

        # Add the widgets to the main layout
        self.layout.add_widget(self.top_layout)   # Top section with the settings button and info label
        self.layout.add_widget(self.stream_layout)  # Middle section with streams
        
        # Add the main layout to the screen
        self.add_widget(self.layout)

    def go_to_settings(self, instance):
        # Switch to the settings page
        self.manager.current = 'settings'

    def start_program(self):
        try:
            self.stream_layout.clear_widgets()

            for name, rtsp_url in rtsp_urls.items():
                # Create a label to show "Loading" and replace it with a video stream later
                loading_label = Label(
                    text=f"Loading {name} stream...",
                    size_hint=(None, None),
                    size=(320, 320),
                    halign='center',
                    valign='middle'
                )
                loading_label.bind(size=loading_label.setter('text_size'))
                self.stream_layout.add_widget(loading_label)

                # Start each video stream in a new thread
                threading.Thread(
                    target=self.start_video_stream,
                    args=(name, rtsp_url, loading_label),
                    daemon=True
                ).start()

                self.update_info_text()

        except Exception as e:
            logging.error(f"Error in starting the program: {e}")

    def start_video_stream(self, name, rtsp_url, loading_label):
        video_capture = VideoCapture(rtsp_url, license_address)
        Clock.schedule_once(lambda dt: self.create_video_widget(video_capture, loading_label), 0)

    def create_video_widget(self, video_capture, loading_label):
        stream_image = Image(size=(320, 320), size_hint=(None, None))
        self.stream_layout.remove_widget(loading_label)
        self.stream_layout.add_widget(stream_image)
        Clock.schedule_interval(lambda dt: self.update_stream(video_capture, stream_image), 1.0 / 30.0)

    '''def update_stream(self, video_capture, stream_image):
        frame = video_capture.read()
        if frame is not None:
            # Run object detection on the frame
            frame = video_capture.detect_objects(frame)
            frame = cv2.resize(frame, (320, 320))
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(320, 320), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            stream_image.texture = texture

            # Check if any detections and control doors
            if latest_detections:
                detection = latest_detections[-1]  # Get the most recent detection
                
            self.update_info_text()'''

    def update_stream(self, video_capture, stream_image):
        frame = video_capture.read()
        if frame is not None:
            # Only detect objects every 5 frames for performance boost
            if self.frame_counter % 5 == 0:
                frame = video_capture.detect_objects(frame)
            frame = cv2.resize(frame, (412, 412))  # Reduce size for smoother performance
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(412, 412), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            stream_image.texture = texture
        
        if latest_detections:
                detection = latest_detections[-1]  # Get the most recent detection
                
        self.update_info_text()
        self.frame_counter += 1

    def update_info_text(self):
        # Generate the info text from the latest detections
        info_text = "\n\n".join([
            f"license number: {d['Carno']} \ntime: {d['timestamp']}\nchannel: {d['channel_state']}"
            for d in latest_detections
        ])

        # Update the info_label with the generated text
        self.info_label.text = info_text

    def on_stop(self):
        for video_capture in video_streams.values():
            video_capture.stop()

class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super(SettingsScreen, self).__init__(**kwargs)

        # Main layout (BoxLayout to split the screen vertically)
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Top layout for input fields (RTSP URLs and License API)
        layout = GridLayout(cols=1, padding=10, spacing=10, size_hint=(1, 0.7))

        # Entry fields for RTSP and License Plate API
        self.entry_rtsp_entry = TextInput(hint_text="Entry Camera", multiline=False)
        layout.add_widget(self.entry_rtsp_entry)

        self.exit1_rtsp_entry = TextInput(hint_text="Exit 1 Camera", multiline=False)
        layout.add_widget(self.exit1_rtsp_entry)

        self.exit2_rtsp_entry = TextInput(hint_text="Exit 2 Camera", multiline=False)
        layout.add_widget(self.exit2_rtsp_entry)

        self.license_api_entry = TextInput(hint_text="License Plate API", multiline=False)
        layout.add_widget(self.license_api_entry)

        self.door_control_entry = TextInput(hint_text="Door Contrl Entry", multiline=False)
        layout.add_widget(self.door_control_entry)

        self.door_control_exit1 = TextInput(hint_text="Door_Contrl_Exit1", multiline=False)
        layout.add_widget(self.door_control_exit1)

        self.door_control_exit2 = TextInput(hint_text="Door_Contrl_Exit2", multiline=False)
        layout.add_widget(self.door_control_exit2)

        # Bottom layout for buttons (Submit and Auto Scan)
        layout2 = GridLayout(cols=2, padding=10, spacing=10, size_hint=(1, 0.3))

        # Submit button to switch back to the stream screen
        self.back_button = Button(text="Submit", size_hint=(None, None), size=(200, 50))
        self.back_button.bind(on_press=self.go_to_stream)

        # Auto-scan button for triggering scan functionality
        self.auto_scan = Button(text="Auto Scan", size_hint=(None, None), size=(200, 50))
        self.auto_scan.bind(on_press=self.auto_scan_action)  # Define functionality for auto scan

        # Add buttons to the button layout (bottom section)
        layout2.add_widget(self.back_button)
        layout2.add_widget(self.auto_scan)

        # Add both layouts (top for inputs, bottom for buttons) to the main layout
        main_layout.add_widget(layout)  # Top section for input fields
        main_layout.add_widget(layout2)  # Bottom section for buttons

        # Add the main layout to the screen
        self.add_widget(main_layout)
    
    def auto_scan_action(self, instance):
        print("Auto scan triggered")
        try:
            file_path = os.path.join(base_dir, "static", "Setting.txt")
            with open(file_path, "r") as file:
                lines = file.readlines()

                if len(lines) >= 7:
                    self.entry_rtsp_entry.text = lines[0].strip()
                    if self.entry_rtsp_entry.text.isdigit():
                        print("Entry RTSP URL is a digit; it should be a valid URL.")
                    
                    self.exit1_rtsp_entry.text = lines[1].strip()
                    if self.exit1_rtsp_entry.text.isdigit():
                        print("Exit1 RTSP URL is a digit; it should be a valid URL.")

                    self.exit2_rtsp_entry.text = lines[2].strip()
                    if self.exit2_rtsp_entry.text.isdigit():
                        print("Exit2 RTSP URL is a digit; it should be a valid URL.")

                    # Read the remaining fields as strings
                    self.license_api_entry.text = lines[3].strip()
                    self.door_control_entry.text = lines[4].strip()
                    self.door_control_exit1.text = lines[5].strip()
                    self.door_control_exit2.text = lines[6].strip()

                    self.go_to_stream(None)
                else:
                    print("Error: Not enough lines in settings.txt file")

        except Exception as e:
            print(f"Error reading the settings.txt file: {e}")

    def go_to_stream(self, instance):
        global rtsp_urls, license_address, entry_door, exit1_door, exit2_door

        rtsp_urls[0] = self.entry_rtsp_entry.text
        rtsp_urls[1] = self.exit1_rtsp_entry.text
        rtsp_urls[2] = self.exit2_rtsp_entry.text

        # Convert to integer if it's a digit; otherwise, keep it as a string
        for i in range(3):
            if rtsp_urls[i].isdigit():
                rtsp_urls[i] = int(rtsp_urls[i])
            # No else needed since rtsp_urls[i] is already a string

        license_address = self.license_api_entry.text
        entry_door = self.door_control_entry.text
        exit1_door = self.door_control_exit1.text
        exit2_door = self.door_control_exit2.text

        if not entry_door or not exit1_door or not exit2_door:
            print("Error: One or more door control URLs are missing.")
            return

        self.manager.current = 'stream'
        self.manager.get_screen('stream').start_program()

'''
    def go_to_stream(self, instance):
        # Update global RTSP URLs and door control URLs
        global rtsp_urls, license_address, entry_door, exit1_door, exit2_door
        rtsp_urls[0] = self.entry_rtsp_entry.text
        rtsp_urls[1] = self.exit1_rtsp_entry.text
        rtsp_urls[2] = self.exit2_rtsp_entry.text
        #rtsp_urls[0] = int(entry_rtsp_entry.get().strip()) if entry_rtsp_entry.get().strip().isdigit() else entry_rtsp_entry.get().strip()
        #rtsp_urls[1] = int(entry_rtsp_exit1.get().strip()) if entry_rtsp_exit1.get().strip().isdigit() else entry_rtsp_exit1.get().strip()
        #rtsp_urls[2] = int(entry_rtsp_exit2.get().strip()) if entry_rtsp_exit2.get().strip().isdigit() else entry_rtsp_exit2.get().strip()
        license_address = self.license_api_entry.text
        entry_door = self.door_control_entry.text
        exit1_door = self.door_control_exit1.text
        exit2_door = self.door_control_exit2.text

        # Ensure URLs are valid and non-empty before proceeding
        if not entry_door or not exit1_door or not exit2_door:
            print("Error: One or more door control URLs are missing.")
            return

        # Switch to the stream page
        self.manager.current = 'stream'
        self.manager.get_screen('stream').start_program()

    def auto_scan_action(self, instance):
        # Logic to read from settings.txt and auto-fill the fields
        print("Auto scan triggered")
        try:
            file_path = os.path.join(base_dir, "static", "Setting.txt")
            with open(file_path, "r") as file:
                lines = file.readlines()

                if len(lines) >= 7:
                    self.entry_rtsp_entry.text = lines[0].strip()
                if self.entry_rtsp_entry.text.isdigit():
                    self.entry_rtsp_entry.text = int(self.entry_rtsp_entry.text)

                self.exit1_rtsp_entry.text = lines[1].strip()
                if self.exit1_rtsp_entry.text.isdigit():
                    self.exit1_rtsp_entry.text = int(self.exit1_rtsp_entry.text)

                self.exit2_rtsp_entry.text = lines[2].strip()
                if self.exit2_rtsp_entry.text.isdigit():
                    self.exit2_rtsp_entry.text = int(self.exit2_rtsp_entry.text)
                    self.license_api_entry.text = lines[3].strip()
                    self.door_control_entry.text = lines[4].strip()
                    self.door_control_exit1.text = lines[5].strip()
                    self.door_control_exit2.text = lines[6].strip()

                    # Call the submission logic
                    self.go_to_stream(None)
                else:
                    print("Error: Not enough lines in settings.txt file")

        except Exception as e:
            print(f"Error reading the settings.txt file: {e}")
            '''

    

class LicensePlateApp(App):
    def build(self):
  
        # Create the screen manager
        sm = ScreenManager()

        # Add the streaming and settings screens to the screen manager
        sm.add_widget(StreamScreen(name='stream'))
        sm.add_widget(SettingsScreen(name='settings'))

        # Set the default screen
        sm.current = 'stream'

        # Start the program in the stream screen
        sm.get_screen('stream').start_program()

        return sm

# Entry point of the application
if __name__ == "__main__":
    LicensePlateApp().run()
