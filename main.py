import cv2
import numpy as np
import time
from threading import Thread
import tkinter as tk
from collections import deque


class GazeControlApp:
    def __init__(self):
        # GUI Setup
        self.root = tk.Tk()
        self.root.title("Gaze Control Interface")
        self.root.configure(bg='gray')
        self.root.resizable(True, True)

        self.selected_row = None
        self.selected_col = None
        self.moved_row = None
        self.moved_col = None
        self.buttons = []

        # Gaze tracking variables
        self.current_gaze = "Center"
        self.gaze_history = deque(maxlen=30)  # Stores last 30 gaze directions (~1 sec at 30fps)
        self.last_gaze_change_time = time.time()

        # Blink detection variables
        self.eye_closed = False
        self.blink_start_time = 0
        self.blink_history = deque(maxlen=15)  # Stores last 15 eye states (~0.5 sec at 30fps)

        # Create GUI elements
        self.create_gui()

        # Video processing
        self.running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Error: Cannot open video source")
            self.root.destroy()
            return

        # Start video processing thread
        self.thread = Thread(target=self.video_loop)
        self.thread.start()

        # Start GUI main loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def create_gui(self):
        # Create 3x3 button grid
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(
                    self.root, text=f"BUTTON–{i * 3 + j + 1}", bg="deepskyblue",
                    command=lambda i=i, j=j: self.on_button_click(i, j)
                )
                btn.grid(row=i + 1, column=j + 1, padx=5, pady=5, sticky="nsew")
                row.append(btn)
            self.buttons.append(row)

        # Arrow buttons
        arrow_style = {"font": ("Arial", 12, "bold"), "width": 2, "height": 1}
        tk.Button(self.root, text="▲", command=self.move_up, **arrow_style).grid(row=0, column=2, sticky="nsew")
        tk.Button(self.root, text="▼", command=self.move_down, **arrow_style).grid(row=4, column=2, sticky="nsew")
        tk.Button(self.root, text="◄", command=self.move_left, **arrow_style).grid(row=2, column=0, sticky="nsew")
        tk.Button(self.root, text="►", command=self.move_right, **arrow_style).grid(row=2, column=4, sticky="nsew")

        # Status label
        self.status_label = tk.Label(self.root, text="Status: Tracking", bg="gray", fg="white")
        self.status_label.grid(row=5, column=0, columnspan=5, sticky="ew")

        # Expand layout
        for i in range(6):  # Added extra row for status
            self.root.grid_rowconfigure(i, weight=1)
        for j in range(5):
            self.root.grid_columnconfigure(j, weight=1)

    def update_highlight(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].configure(bg="deepskyblue")

        if self.moved_row is not None and self.moved_col is not None:
            self.buttons[self.moved_row][self.moved_col].configure(bg="lime")

        if self.selected_row is not None and self.selected_col is not None:
            self.buttons[self.selected_row][self.selected_col].configure(bg="yellow")

    # Movement functions
    def move_up(self):
        if self.moved_row is None or self.moved_col is None:
            self.moved_row, self.moved_col = 1, 1
        if self.moved_row > 0:
            self.moved_row -= 1
            self.update_highlight()

    def move_down(self):
        if self.moved_row is None or self.moved_col is None:
            self.moved_row, self.moved_col = 1, 1
        if self.moved_row < 2:
            self.moved_row += 1
            self.update_highlight()

    def move_left(self):
        if self.moved_row is None or self.moved_col is None:
            self.moved_row, self.moved_col = 1, 1
        if self.moved_col > 0:
            self.moved_col -= 1
            self.update_highlight()

    def move_right(self):
        if self.moved_row is None or self.moved_col is None:
            self.moved_row, self.moved_col = 1, 1
        if self.moved_col < 2:
            self.moved_col += 1
            self.update_highlight()

    def on_button_click(self, i, j):
        if self.moved_row == i and self.moved_col == j:
            self.selected_row = i
            self.selected_col = j
            self.update_highlight()

    def detect_blink(self, frame):
        """Detects if eyes are closed by analyzing the darkest region size"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find darkest region (similar to pupil detection)
        darkest_point = get_darkest_area(frame)
        masked = mask_outside_square(gray, darkest_point, 100)
        thresholded = apply_binary_threshold(masked, int(np.min(masked)), 30)

        # Count white pixels (potential pupil area)
        white_pixels = cv2.countNonZero(thresholded)

        print("White pixels:", white_pixels) # for debuging

        # If white area is very small, eyes are likely closed
        return white_pixels < 108000  # Threshold may need adjustment

    def check_gaze_dwell(self):
        """Checks if gaze has been consistent for 1 seconds and triggers movement"""
        if len(self.gaze_history) < 30:  # Need at least 1 second of data
            return

        # Check if all recent gazes are the same (and not Center)
        unique_gazes = set(self.gaze_history)
        if len(unique_gazes) == 1 and self.current_gaze != "Center":
            if time.time() - self.last_gaze_change_time >= 1:  # 1 seconds dwell
                # Move in opposite direction
                if self.current_gaze == "Right":
                    self.root.after(0, self.move_right)
                elif self.current_gaze == "Left":
                    self.root.after(0, self.move_left)
                elif self.current_gaze == "Up":
                    self.root.after(0, self.move_up)
                elif self.current_gaze == "Down":
                    self.root.after(0, self.move_down)

                # Reset timer
                self.last_gaze_change_time = time.time()

    def check_blink_selection(self):
        """Checks if eyes have been closed for 2 seconds to select current button"""
        if len(self.blink_history) < 15:  # Need at least 0.5 seconds of data
            return

        # Check if all recent states are "closed"
        if all(self.blink_history) and self.eye_closed:
            if time.time() - self.blink_start_time >= 1:  # 1 seconds closed
                # Select current button
                if self.moved_row is not None and self.moved_col is not None:
                    self.root.after(0, self.on_button_click, self.moved_row, self.moved_col)
                    self.status_label.config(text="Status: Selected!")

                # Reset blink timer
                self.blink_start_time = time.time()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = frame[110:350, 80:560]

            # Blink detection
            is_closed = self.detect_blink(frame)
            self.blink_history.append(is_closed)

            # Update blink state
            if is_closed and not self.eye_closed:
                self.eye_closed = True
                self.blink_start_time = time.time()
                self.status_label.config(text="Status: Eye closed")
            elif not is_closed and self.eye_closed:
                self.eye_closed = False
                self.status_label.config(text="Status: Tracking")

            # Check for blink selection
            self.check_blink_selection()

            # Only process gaze when eyes are open
            if not self.eye_closed:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                darkest_point = get_darkest_area(frame)
                masked = mask_outside_square(gray_frame, darkest_point, 100)
                thresholded_raw = apply_binary_threshold(gray_frame, int(np.min(gray_frame)), 30)
                thresholded = mask_outside_square(thresholded_raw, darkest_point, 100)

                # Process frame and get gaze direction
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(thresholded, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

                if largest_contours and len(largest_contours[0]) > 5:
                    ellipse = cv2.fitEllipse(largest_contours[0])
                    direction, _ = estimate_gaze_direction(ellipse, frame.shape)

                    # Update gaze history
                    if direction != self.current_gaze:
                        self.current_gaze = direction
                        self.last_gaze_change_time = time.time()

                    self.gaze_history.append(direction)

                    # Check for gaze dwell
                    self.check_gaze_dwell()

                    # Visual feedback
                    cv2.putText(frame, f"Gaze: {direction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Thresholded", thresholded)
            cv2.imshow("Pupil Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False

    def on_close(self):
        self.running = False
        self.root.destroy()


# === Image Processing Functions ===
def apply_binary_threshold(image, darkest_pixel_value, added_threshold):
    threshold = darkest_pixel_value + added_threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image


def get_darkest_area(image):
    ignore_bounds = 20
    image_skip = 10
    block_size = 20
    internal_skip = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = (gray.shape[1] // 2, gray.shape[0] // 2)

    for y in range(ignore_bounds, gray.shape[0] - block_size, image_skip):
        for x in range(ignore_bounds, gray.shape[1] - block_size, image_skip):
            block_sum = np.sum(gray[y:y + block_size:internal_skip, x:x + block_size:internal_skip])
            if block_sum < min_sum:
                min_sum = block_sum
                darkest_point = (x + block_size // 2, y + block_size // 2)

    return darkest_point


def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2
    mask = np.zeros_like(image)

    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)


def filter_contours_by_area_and_return_largest(contours, area_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= area_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio <= ratio_thresh and area > max_area:
                max_area = area
                largest_contour = contour

    return [largest_contour] if largest_contour is not None else []


def estimate_gaze_direction(ellipse, frame, threshold=0.15):
    pupil_center = ellipse[0]
    frame_h, frame_w = frame[:2]

    norm_x = (pupil_center[0] - frame_w / 2) / (frame_w / 2)
    norm_y = (pupil_center[1] - frame_h / 2) / (frame_h / 2)

    if abs(norm_x) > abs(norm_y):
        if norm_x < -threshold:
            return "Right", (norm_x, norm_y)
        elif norm_x > threshold:
            return "Left", (norm_x, norm_y)
    else:
        if norm_y < -threshold:
            return "Up", (norm_x, norm_y)
        elif norm_y > threshold:
            return "Down", (norm_x, norm_y)

    return "Center", (norm_x, norm_y)


if __name__ == "__main__":
    app = GazeControlApp()