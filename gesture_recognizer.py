"""
Gesture Recognizer Module
Handles pinch detection, gesture debouncing, and state machine transitions.
Converts raw hand landmarks into drawing intent.
"""

import numpy as np
from enum import Enum


class GestureState(Enum):
    """Possible gesture states."""
    IDLE = "idle"
    DRAWING = "drawing"
    UI_INTERACT = "ui_interact"


class GestureRecognizer:
    """
    Recognizes drawing gestures using pinch detection with debouncing.

    Uses a state machine with hysteresis to prevent flickering:
    IDLE -> DRAWING: pinch detected for N consecutive frames
    DRAWING -> IDLE: pinch released for N consecutive frames
    """

    def __init__(self, pinch_threshold_ratio=0.4, debounce_frames=1):
        """
        Initialize the gesture recognizer.

        Args:
            pinch_threshold_ratio: Ratio of hand size below which a pinch is detected.
                                   Lower = requires fingers closer together.
            debounce_frames: Number of consecutive frames needed to confirm state change.
        """
        self.pinch_threshold_ratio = pinch_threshold_ratio
        self.debounce_frames = debounce_frames

        # State machine
        self.current_state = GestureState.IDLE
        self._pinch_counter = 0
        self._release_counter = 0

        # For tracking pinch quality
        self._last_pinch_distance = 0
        self._last_hand_size = 0

    def update(self, landmarks, hand_size, ui_regions=None):
        """
        Update gesture state based on current landmarks.

        Args:
            landmarks: Dict of landmark positions from HandTracker.
            hand_size: Current hand size in pixels for dynamic thresholding.
            ui_regions: List of (x, y, w, h, action) tuples for UI button regions.

        Returns:
            tuple: (GestureState, pinch_position or None)
        """
        if landmarks is None or hand_size == 0:
            self._release_counter += 1
            self._pinch_counter = 0
            if self._release_counter >= self.debounce_frames:
                self.current_state = GestureState.IDLE
            return self.current_state, None

        # Calculate pinch distance
        index_tip = np.array(landmarks['index_tip'])
        thumb_tip = np.array(landmarks['thumb_tip'])
        pinch_distance = float(np.linalg.norm(index_tip - thumb_tip))

        self._last_pinch_distance = pinch_distance
        self._last_hand_size = hand_size

        # Dynamic threshold based on hand size
        threshold = hand_size * self.pinch_threshold_ratio

        # Pinch midpoint (where the drawing happens)
        pinch_pos = tuple(((index_tip + thumb_tip) / 2).astype(int))

        is_pinching = pinch_distance < threshold

        if is_pinching:
            self._pinch_counter += 1
            self._release_counter = 0

            # Check if pinch is over a UI element
            if ui_regions and self._pinch_counter >= self.debounce_frames:
                for region in ui_regions:
                    rx, ry, rw, rh, action = region
                    if (rx <= pinch_pos[0] <= rx + rw and
                            ry <= pinch_pos[1] <= ry + rh):
                        self.current_state = GestureState.UI_INTERACT
                        return self.current_state, pinch_pos

            # Transition to drawing after debounce threshold
            if self._pinch_counter >= self.debounce_frames:
                self.current_state = GestureState.DRAWING
        else:
            self._release_counter += 1
            self._pinch_counter = 0

            # Transition back to idle after debounce threshold
            if self._release_counter >= self.debounce_frames:
                self.current_state = GestureState.IDLE

        return self.current_state, pinch_pos if is_pinching else None

    def get_pinch_info(self):
        """
        Get debug info about the last pinch detection.

        Returns:
            dict with pinch distance, hand size, threshold, etc.
        """
        threshold = self._last_hand_size * self.pinch_threshold_ratio
        return {
            'distance': self._last_pinch_distance,
            'hand_size': self._last_hand_size,
            'threshold': threshold,
            'state': self.current_state.value,
            'pinch_counter': self._pinch_counter,
            'release_counter': self._release_counter,
        }

    def reset(self):
        """Reset the gesture recognizer state."""
        self.current_state = GestureState.IDLE
        self._pinch_counter = 0
        self._release_counter = 0
