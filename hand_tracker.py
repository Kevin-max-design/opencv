"""
Hand Tracker Module
Wraps MediaPipe Hands for real-time hand landmark detection.
Extracts key landmarks: index fingertip, thumb tip, index MCP.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Real-time hand landmark tracking using MediaPipe."""

    # Key landmark indices
    INDEX_TIP = 8
    THUMB_TIP = 4
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    WRIST = 0

    def __init__(self, max_hands=1, min_detection_conf=0.7, min_tracking_conf=0.7):
        """
        Initialize the hand tracker.

        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_conf: Minimum detection confidence threshold.
            min_tracking_conf: Minimum tracking confidence threshold.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def process_frame(self, frame):
        """
        Process a video frame and detect hand landmarks.

        Args:
            frame: BGR image from OpenCV (numpy array).

        Returns:
            results: MediaPipe hand detection results.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def get_landmarks(self, results, frame_shape):
        """
        Extract key landmark positions in pixel coordinates.

        Args:
            results: MediaPipe hand detection results.
            frame_shape: Shape of the video frame (height, width, channels).

        Returns:
            dict with landmark positions or None if no hand detected.
            Keys: 'index_tip', 'thumb_tip', 'index_mcp', 'middle_tip', 'wrist',
                  'all_landmarks' (list of all 21 landmarks in pixel coords)
        """
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame_shape

        def to_pixel(landmark):
            return (int(landmark.x * w), int(landmark.y * h))

        landmarks = {
            'index_tip': to_pixel(hand_landmarks.landmark[self.INDEX_TIP]),
            'thumb_tip': to_pixel(hand_landmarks.landmark[self.THUMB_TIP]),
            'index_mcp': to_pixel(hand_landmarks.landmark[self.INDEX_MCP]),
            'middle_tip': to_pixel(hand_landmarks.landmark[self.MIDDLE_TIP]),
            'wrist': to_pixel(hand_landmarks.landmark[self.WRIST]),
            'all_landmarks': [
                to_pixel(lm) for lm in hand_landmarks.landmark
            ],
        }

        return landmarks

    def draw_landmarks(self, frame, results):
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame: BGR image to draw on.
            results: MediaPipe hand detection results.
        """
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

    def get_hand_size(self, landmarks):
        """
        Calculate approximate hand size based on wrist-to-middle-finger distance.
        Used for dynamic threshold scaling.

        Args:
            landmarks: Dict of landmark positions from get_landmarks().

        Returns:
            float: Approximate hand size in pixels.
        """
        if landmarks is None:
            return 0

        wrist = np.array(landmarks['wrist'])
        middle_tip = np.array(landmarks['middle_tip'])
        return float(np.linalg.norm(middle_tip - wrist))

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
