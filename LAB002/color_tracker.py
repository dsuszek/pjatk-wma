# pylint: disable=E0401
"""
Module for color tracking in a video.

This module provides classes and functions for tracking colors in a video.
It includes:

Classes:
    ProcessingType: Enum class defining different types of processing.
    ColorTracker: Class for tracking colors in a video.
    Display: Class responsible for presenting the results to the user.
    EventHandler: Class that handles input from the user.

Functions:
    parse_arguments(): Parses command line arguments.
    main(): Main function to track colors in a video.
"""
from enum import Enum
from typing import Optional, Tuple
import random as rng
import argparse
import cv2
import numpy as np


class ProcessingType(Enum):
    """
    Enum class representing different types of processing applied to frames.

    Attributes:
        RAW: No modification.
        TRACKER: Object are tracked.
        HUE: Video is processed based on the hue values.
        SATURATION: Video is processed based on the saturation.
        VALUE: If this attribute is chosen, processing will be based on value (brightness).
        MASK: Processing based on a mask.
    """
    RAW = 0
    TRACKER = 1
    HUE = 2
    SATURATION = 3
    VALUE = 4
    MASK = 5


# MODEL
class ColorTracker:
    """
    Class for tracking colors in a video. This is the most complex class in the program.

    Attributes:
        _ht (int): Tolerance for the hue parameter.
        _st (int): Tolerance for the saturation parameter.
        _vt (int): Tolerance for the value parameter.
        _video (cv2.VideoCapture): Video capture object.
        _tracked_color (None | tuple[int, int, int]): RGB tuple representing the tracked color.
        _frame (None | np.ndarray): Current frame from the video.
        _processed_frame (None | np.ndarray): Processed frame after color tracking.
        _processing_type (ProcessingType): Type of processing applied to the frames.
        _tracked_object_position (None | np.ndarray): Position of the tracked object in the frame.

    Functions:
        set_processing_type(): Sets a processing type.
        set_reference_color_by_position(): Sets the color to be tracked.
        track_object(): Track the object in the current frame based on the tracked color.
        update_frame(): Checks if the video is available.
                        If so, then starts the processing of the frames.
        process_frame(): Processes current frame of the video.
        get_frame(): Gets the current frame from the color tracker.
        get_processed_frame(): Returns the processed frame after color tracking.
    """

    def __init__(self, video_path: str, ht: int, st: int, vt: int,
                 tracked_color: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Constructor of ColorTracker class. Initializes a ColorTracker object.

        Parameters:
            video_path (str): The path to the video file.
            ht (int): Tolerance for the hue.
            st (int): Tolerance for the saturation.
            vt (int): Tolerance for the value.
            tracked_color (Optional[Tuple[int, int, int]], optional): Tracked color info.

         Returns:
             None
         """
        self._ht = ht
        self._st = st
        self._vt = vt
        self._video = cv2.VideoCapture(video_path)
        if not self._video.isOpened():
            raise ValueError(f'Unable to open video at path {video_path}.')
        self._tracked_color = tracked_color
        self._frame: np.ndarray = np.zeros(0)
        self._processed_frame: np.ndarray = np.zeros(0)
        self._processing_type: ProcessingType = ProcessingType.RAW
        self._tracked_object_position: np.ndarray = np.zeros(0)
        self._object_detected = False

    def set_processing_type(self, ptype: ProcessingType) -> None:
        """
        Sets a processing type.

        Parameters:
            ptype (ProcessingType):The type of the processing.

        Returns:
            None
        """
        self._processing_type = ptype

    def set_reference_color_by_position(self, x_coord: int, y_coord: int) -> None:
        """
        Sets the reference color for tracking based on the position in the frame.

        Parameters:
            x_coord (int): The x-coordinate of the pixel.
            y_coord (int): The y-coordinate of the pixel.

        Returns:
            None
        """
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        self._tracked_color = hsv_frame[y_coord, x_coord, :]

    def _track_object(self) -> None:
        """
        Tracks the object in the current frame based on the tracked color.

        This method adds color tracking to the current frame for the color chosen by the user.
        In a nutshell, it converts the frame from RGB to HSV space and creates a mask.
        To do this, it uses tracked color's hue, saturation, and value considering the tolerances.
        In the next step, it identifies contours in the mask.
        Then it identifies the largest contour, representing the object.
        Finally, it draws a rectangle around it.

        Returns:
            None
        """

        # Change RGB to HSV
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)

        if self._tracked_color is None:
            raise ValueError("Tracked color is not set.")

        print('tracked color:', self._tracked_color)


        h_range, s_range, v_range = (
            [
                self._tracked_color[0] - self._ht,
                self._tracked_color[0] + self._ht
            ],
            [
                self._tracked_color[1] - self._st,
                self._tracked_color[1] + self._st,
            ],
            [
                self._tracked_color[2] - self._vt,
                self._tracked_color[2] + self._vt,
            ],
        )
        h_range[0] = h_range[0] if h_range[0] > 0 else 0
        s_range[0] = s_range[0] if s_range[0] > 0 else 0
        v_range[0] = v_range[0] if v_range[0] > 0 else 0
        h_range[1] = h_range[1] if h_range[1] < 180 else 180
        s_range[1] = s_range[1] if s_range[1] < 255 else 255
        v_range[1] = v_range[1] if v_range[1] < 255 else 255

        lower_limit = np.array([h_range[0], s_range[0], v_range[0]])
        upper_limit = np.array([h_range[1], s_range[1], v_range[1]])

        print('lower_limit: ', lower_limit)
        print('upper_limit: ', upper_limit)

        # Add a mask
        mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

        # Find contours
        contours = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]

        if len(contours) > 0:
            # And now find the biggest contour
            contour = max(contours, key=cv2.contourArea)
            # Add rectangle around the object
            min_x, min_y, max_x, max_y = cv2.boundingRect(contour)

            self._tracked_object_position = (min_x, min_y, max_x, max_y)
            self._processed_frame = cv2.rectangle(self._frame.copy(), (min_x, min_y),
                                                  (min_x + max_x, min_y + max_y), (0, 0, 0), 2)
            self._object_detected = True
        else:
            self._tracked_object_position = None
            self._object_detected = False
            self._processed_frame = self._frame

    def update_frame(self) -> bool:
        """
        Checks if the video is available. If so, then starts the processing of the frames.

        Returns:
            bool: True if the video was successfully loaded and processed, False otherwise.
        """
        read_successful, self._frame = self._video.read()
        if read_successful:
            self._process_frame()
        return read_successful

    def _process_frame(self) -> None:
        """
        Processes current frame of the video.

        Returns:
            None
        """
        if self._processing_type == ProcessingType.RAW:
            self._processed_frame = self._frame
            return
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        hue = hsv_frame[:, :, 0]
        saturation = hsv_frame[:, :, 1]
        value = hsv_frame[:, :, 2]
        if self._processing_type == ProcessingType.HUE:
            self._processed_frame = hue
        elif self._processing_type == ProcessingType.SATURATION:
            self._processed_frame = saturation
        elif self._processing_type == ProcessingType.VALUE:
            self._processed_frame = value

        if self._tracked_color is None:
            raise ValueError(
                'Attempted processing mode that requires a tracking color set without it set.'
            )
        mask = np.zeros_like(hue)
        mask[hue == self._tracked_color[0]] = 255
        if self._processing_type == ProcessingType.MASK:
            self._processed_frame = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [cv2.approxPolyDP(contour, 3, True) for contour in contours]
        bound_rect = [cv2.boundingRect(poly) for poly in contours_poly]
        centers = []
        radius = []

        for poly in contours_poly:
            center, rad = cv2.minEnclosingCircle(poly)
            centers.append(center)
            radius.append(rad)

        drawing = np.zeros_like(mask, dtype=np.uint8)

        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(bound_rect[i][0]), int(bound_rect[i][1])),
                          (int(bound_rect[i][0] + bound_rect[i][2]),
                           int(bound_rect[i][1] + bound_rect[i][3])), color, 2)
            cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

        if self._processing_type == ProcessingType.TRACKER:
            self._track_object()

    def get_frame(self) -> np.ndarray:
        """
        Gets the current frame from the color tracker.

        Returns:
            np.ndarray: The current frame.
         """
        if self._frame is None:
            raise ValueError('Attempted to get frame from uninitialized color tracker.')
        return self._frame.copy()

    def get_processed_frame(self) -> np.ndarray:
        """
        Returns the processed frame after color tracking.

        Returns:
            np.ndarray: A copy of the processed frame.
        """
        return self._processed_frame.copy()


# VIEW
class Display:
    """
        A class that represents the display.

        Functions:
        update_display(): Updates the display with a new image.
        get_window_name(): Returns the name of the window.
    """
    def __init__(self, window_name: str) -> None:
        """
        Constructor of Display class. Initializes a Display object.

        Parameters:
            window_name (str): Name of the window.
        """
        self._window = cv2.namedWindow(window_name)
        self._window_name = window_name

    def update_display(self, image: np.ndarray) -> None:
        """
        Updates the display with a new image.

            Parameters:
                image (np.ndarray): The image to display.

            Returns:
                None
        """
        cv2.imshow(self._window_name, image)

    def get_window_name(self) -> str:
        """
        Returns the name of the window.

            Returns:
                str: The name of the window.
        """
        return self._window_name


# CONTROLLER
class EventHandler:
    """
    A class representing a display window.

    Attributes:
        PROCESSING_TYPE_KEYMAP

    Functions:
        handle_mouse(): Handles mouse events such as clicking.
        handle_keys(): Handles the keys clicked by the user.
        handle_events(): Triggers handle_keys() function.
    """
    PROCESSING_TYPE_KEYMAP = {
        ord('h'): ProcessingType.HUE,
        ord('s'): ProcessingType.SATURATION,
        ord('v'): ProcessingType.VALUE,
        ord('r'): ProcessingType.RAW,
        ord('m'): ProcessingType.MASK,
        ord('t'): ProcessingType.TRACKER
    }

    def __init__(self, tracker: ColorTracker, display: Display, timeout: int) -> None:
        """
        Constructor of EventHandler class. Initializes an EventHandler object.

        Parameters:
            tracker (ColorTracker): Object of class ColorTracker.
            display (Display): Object of class Display.
            timeout (int): Amount of time after which the timeout should appear.
         """
        self._window_name = display.get_window_name()
        self._tracker = tracker
        self._timeout = timeout
        cv2.setMouseCallback(self._window_name, self._handle_mouse)

    def _handle_mouse(self, event, x_coord, y_coord, flags, param) -> None:
        """
        Handles mouse events such as clicking.

        Parameters:
            event (int): The type of mouse event.
            x_coord (int): The x-coordinate of the mouse cursor.
            y_coord (int): The y-coordinate of the mouse cursor.
            flags (int): Additional flags associated with the event.
            param: Additional parameters.

        Returns:
            None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tracker.set_reference_color_by_position(x_coord, y_coord)
            self._tracker._track_object()
            print(self._tracker._tracked_color)

    def _handle_keys(self) -> bool:
        """
        Handles the keys clicked by the user.

        Returns:
            bool: False if the user clicked on 'q' - in this case the application should be closed.
                  True otherwise.
        """
        keycode = cv2.waitKey(self._timeout)
        if keycode == ord('q') or keycode == 27:
            return False
        elif keycode in EventHandler.PROCESSING_TYPE_KEYMAP.keys():
            self._tracker.set_processing_type(EventHandler.PROCESSING_TYPE_KEYMAP[keycode])
        return True

    def handle_events(self) -> bool:
        """
        Triggers handle_keys() function.

        Returns:
            bool: True if events are successfully handled, False otherwise.
        """
        return self._handle_keys()


def parse_arguments() -> argparse.Namespace:
    """
    Parses arguments from command line.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', type=str, required=True,
                        help='Path to video that will be processed.')
    parser.add_argument('-ht', '--hue_tolerance', type=int, required=True,
                        help='Tolerance for the hue parameter.')
    parser.add_argument('-st', '--saturation_tolerance', type=int, required=True,
                        help='Tolerance for the saturation parameter.')
    parser.add_argument('-vt', '--value_tolerance', type=int, required=True,
                        help='Tolerance for the hue parameter.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function to track colors in a video.

    Parameters:
        args (argparse.Namespace): Parsed command line arguments containing video path
            and color tolerance values.

    Returns:
        None
    """
    try:
        window_name = 'Color tracker'
        wait_key_timeout = 10
        tracker = ColorTracker(
            args.video_path, args.hue_tolerance, args.saturation_tolerance, args.value_tolerance
        )
        display = Display(window_name)
        event_handler = EventHandler(tracker, display, wait_key_timeout)
        while True:
            if not tracker.update_frame():
                break
            display.update_display(tracker.get_processed_frame())
            if not event_handler.handle_events():
                break

    except ValueError as val_error:
        print(val_error)


if __name__ == '__main__':
    main(parse_arguments())
