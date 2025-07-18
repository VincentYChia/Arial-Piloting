import cv2, time
from typing import Tuple
from threading import Event
from .abs.robot_wrapper import RobotWrapper


class FrameReader:
    def __init__(self, cap):
        # Initialize the video capture
        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    @property
    def frame(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class VirtualRobotWrapper(RobotWrapper):
    def __init__(self):
        self.stream_on = False

        # ADD CANCELLATION SUPPORT
        self.cancellation_event = Event()
        self.movement_x_accumulator = 0
        self.movement_y_accumulator = 0
        self.rotation_accumulator = 0

    def set_cancellation_event(self, event: Event):
        """Set the cancellation event from the controller"""
        self.cancellation_event = event

    def stop_all_commands(self):
        """Stop all ongoing commands immediately"""
        self.cancellation_event.set()

    def _sleep_with_cancellation(self, duration: float):
        """Sleep while checking for cancellation every 0.1 seconds"""
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.cancellation_event.is_set():
                return False
            time.sleep(0.1)
        return True

    def keep_active(self):
        pass

    def connect(self):
        pass

    def takeoff(self) -> bool:
        return True

    def land(self):
        pass

    def start_stream(self):
        self.cap = cv2.VideoCapture(0)
        self.stream_on = True

    def stop_stream(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return FrameReader(self.cap)

    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving forward {distance} cm")

        if self.cancellation_event.is_set():
            return False, False

        self.movement_x_accumulator += distance

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving backward {distance} cm")

        if self.cancellation_event.is_set():
            return False, False

        self.movement_x_accumulator -= distance

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving left {distance} cm")

        if self.cancellation_event.is_set():
            return False, False

        self.movement_y_accumulator += distance

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving right {distance} cm")

        if self.cancellation_event.is_set():
            return False, False

        self.movement_y_accumulator -= distance

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")

        if self.cancellation_event.is_set():
            return False, False

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")

        if self.cancellation_event.is_set():
            return False, False

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")

        if self.cancellation_event.is_set():
            return False, False

        self.rotation_accumulator += degree

        if degree >= 90:
            print("-> Turning CCW over 90 degrees")
            if not self._sleep_with_cancellation(2.0):
                return False, False
            return True, False

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")

        if self.cancellation_event.is_set():
            return False, False

        self.rotation_accumulator -= degree

        if degree >= 90:
            print("-> Turning CW over 90 degrees")
            if not self._sleep_with_cancellation(2.0):
                return False, False
            return True, False

        if not self._sleep_with_cancellation(1.0):
            return False, False

        return True, False