"""
Drawing Engine Module
Manages the drawing canvas with temporal smoothing, stroke management,
and multi-color support.
"""

import cv2
import numpy as np
from collections import deque


class DrawingEngine:
    """
    Manages drawing strokes on a transparent canvas overlay.

    Features:
    - Temporal smoothing (EMA) for jitter-free lines
    - Stroke history for undo support
    - Anti-aliased line rendering
    - Multi-color and variable thickness support
    """

    def __init__(self, width, height, smoothing_factor=0.45):
        """
        Initialize the drawing engine.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            smoothing_factor: EMA smoothing factor (0-1). Lower = smoother but more lag.
        """
        self.width = width
        self.height = height
        self.smoothing_factor = smoothing_factor

        # Canvas (4-channel BGRA for transparency)
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)

        # Current drawing state
        self.current_color = (255, 255, 255)  # White default (BGR)
        self.current_thickness = 4
        self.is_drawing = False

        # Smoothed position
        self._smoothed_pos = None
        self._prev_smoothed_pos = None

        # Stroke history for undo
        self._strokes = []       # List of completed strokes
        self._current_stroke = []  # Points in the current stroke

        # Position history for advanced smoothing
        self._pos_history = deque(maxlen=5)

    def start_stroke(self):
        """Begin a new drawing stroke."""
        if not self.is_drawing:
            self.is_drawing = True
            self._current_stroke = []
            self._smoothed_pos = None
            self._prev_smoothed_pos = None
            self._pos_history.clear()

    def end_stroke(self):
        """End the current drawing stroke and save to history."""
        if self.is_drawing:
            self.is_drawing = False
            if len(self._current_stroke) > 1:
                self._strokes.append({
                    'points': self._current_stroke.copy(),
                    'color': self.current_color,
                    'thickness': self.current_thickness,
                })
            self._current_stroke = []
            self._smoothed_pos = None
            self._prev_smoothed_pos = None
            self._pos_history.clear()

    def add_point(self, raw_pos):
        """
        Add a new point to the current stroke with temporal smoothing.

        Args:
            raw_pos: Tuple (x, y) of the raw fingertip position.
        """
        if not self.is_drawing:
            return

        x, y = raw_pos

        # Clamp to canvas bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))

        # Add to position history for weighted average
        self._pos_history.append((x, y))

        # Apply temporal smoothing (Exponential Moving Average)
        if self._smoothed_pos is None:
            self._smoothed_pos = np.array([float(x), float(y)])
        else:
            raw = np.array([float(x), float(y)])

            # Weighted average of recent positions for extra smoothness
            if len(self._pos_history) >= 3:
                weights = np.array([0.1, 0.2, 0.7])[-len(self._pos_history):]
                weights = weights / weights.sum()
                history = np.array(list(self._pos_history))[-len(weights):]
                weighted_pos = np.average(history, axis=0, weights=weights)
                raw = weighted_pos

            # EMA smoothing
            alpha = self.smoothing_factor
            self._smoothed_pos = alpha * raw + (1 - alpha) * self._smoothed_pos

        smoothed = tuple(self._smoothed_pos.astype(int))

        # Draw line segment from previous position
        if self._prev_smoothed_pos is not None:
            prev = self._prev_smoothed_pos

            # Only draw if movement is significant (> 1 pixel)
            dist = np.sqrt((smoothed[0] - prev[0])**2 + (smoothed[1] - prev[1])**2)
            
            # DEBUG: Always print
            # print(f"DEBUG: Stroke dist={dist:.2f} | Pos={smoothed}")

            if dist > 0:  # Relaxed from 1 to 0 for sensitivity
                # Draw on canvas with anti-aliasing
                cv2.line(
                    self.canvas,
                    prev, smoothed,
                    (*self.current_color, 255),  # BGRA
                    self.current_thickness,
                    cv2.LINE_AA
                )
                self._current_stroke.append(smoothed)

        self._prev_smoothed_pos = smoothed

    def set_color(self, color):
        """
        Set the current drawing color.

        Args:
            color: Tuple (B, G, R) color values.
        """
        self.current_color = color

    def set_thickness(self, thickness):
        """
        Set the current line thickness.

        Args:
            thickness: Line thickness in pixels.
        """
        self.current_thickness = max(1, min(thickness, 20))

    def clear_canvas(self):
        """Clear the entire canvas."""
        self.canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self._strokes.clear()
        self._current_stroke.clear()
        self._smoothed_pos = None
        self._prev_smoothed_pos = None

    def undo(self):
        """Undo the last completed stroke."""
        if not self._strokes:
            return

        self._strokes.pop()

        # Redraw all remaining strokes
        self.canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for stroke in self._strokes:
            points = stroke['points']
            color = stroke['color']
            thickness = stroke['thickness']
            for i in range(1, len(points)):
                cv2.line(
                    self.canvas,
                    points[i - 1], points[i],
                    (*color, 255),
                    thickness,
                    cv2.LINE_AA,
                )

    def get_overlay(self):
        """
        Get the canvas as a BGR overlay ready for compositing.

        Returns:
            tuple: (overlay_bgr, alpha_mask) for blending with the video frame.
        """
        bgr = self.canvas[:, :, :3]
        alpha = self.canvas[:, :, 3]
        return bgr, alpha

    def composite(self, frame):
        """
        Composite the drawing canvas onto a video frame.

        Args:
            frame: BGR video frame (numpy array).

        Returns:
            Composited frame with drawings overlaid.
        """
        bgr, alpha = self.get_overlay()

        # Create alpha mask (normalized to 0-1)
        alpha_mask = alpha.astype(float) / 255.0

        # Blend
        for c in range(3):
            frame[:, :, c] = (
                alpha_mask * bgr[:, :, c] +
                (1 - alpha_mask) * frame[:, :, c]
            ).astype(np.uint8)

        return frame

    def get_stroke_count(self):
        """Return the number of completed strokes."""
        return len(self._strokes)
