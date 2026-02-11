"""
Air Drawing — Touchless Gesture-Based Drawing System
Main application entry point.

Draw in the air using hand gestures captured by your webcam.
Uses pinch gestures to draw, with temporal smoothing for smooth strokes.

Controls:
    - Pinch (index + thumb) to draw
    - Pinch over toolbar buttons to select colors, sizes, clear, or undo
    - Press 'q' to quit
    - Press 'c' to clear canvas
    - Press 'u' to undo last stroke
    - Press 'd' to toggle debug overlay
"""

import cv2
import numpy as np
import time

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer, GestureState
from drawing_engine import DrawingEngine
from ui_overlay import UIOverlay


class AirDrawingApp:
    """Main application class for the Air Drawing system."""

    def __init__(self, camera_id=0, width=1280, height=720):
        """
        Initialize the Air Drawing application.

        Args:
            camera_id: Camera device index.
            width: Capture width.
            height: Capture height.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height

        # Initialize components
        self.tracker = HandTracker(
            max_hands=1,
            min_detection_conf=0.7,
            min_tracking_conf=0.7,
        )
        self.gesture = GestureRecognizer(
            pinch_threshold_ratio=0.12,
            debounce_frames=3,
        )
        self.engine = None  # Initialized after camera opens
        self.ui = None      # Initialized after camera opens

        # Application state
        self.show_debug = False
        self.show_landmarks = True
        self.running = False

        # FPS tracking
        self._fps_times = []
        self._fps = 0

        # Keyboard state
        self.spacebar_pressed = False

        # UI interaction cooldown (prevent rapid-fire clicks)
        self._last_ui_action_time = 0
        self._ui_cooldown = 0.5  # seconds

        # Previous gesture state tracking
        self._prev_state = GestureState.IDLE

    def run(self):
        """Run the main application loop."""
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            print("⚠️  Permission denied? If on macOS, check System Settings > Privacy & Security > Camera.")
            print("   You may need to run this from your terminal: python main.py")
            return

        # Get actual resolution
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {actual_w}x{actual_h}")

        # Initialize components with actual resolution
        self.engine = DrawingEngine(actual_w, actual_h, smoothing_factor=0.45)
        self.ui = UIOverlay(actual_w, actual_h)

        self.running = True
        print("\n╔══════════════════════════════════════════╗")
        print("║    Air Drawing — Touchless Drawing App   ║")
        print("╠══════════════════════════════════════════╣")
        print("║  ✋ Pinch index+thumb to draw            ║")
        print("║  🎨 Pinch over colors to switch          ║")
        print("║  🧹 Pinch over CLEAR to reset            ║")
        print("║  ↩️  Pinch over UNDO or press 'u'         ║")
        print("║  📊 Press 'd' to toggle debug info       ║")
        print("║  ❌ Press 'q' to quit                    ║")
        print("╚══════════════════════════════════════════╝\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Mirror the frame for natural interaction
                frame = cv2.flip(frame, 1)

                # Process frame through pipeline
                frame = self._process_frame(frame)

                # Display
                cv2.imshow('Air Drawing', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)

                # Check actual spacebar state (async key check for holding)
                # 32 is spacebar. 0x8000 checks if key is down.
                # Note: cv2.waitKey is discrete. For "holding", we might need a toggle or different approach.
                # Simpler: Toggle 's' to start/stop drawing if holding is hard with cv2.
                # BETTER: Just map 'space' to toggle for now as cv2 key holding is tricky without OS-specifics.
                pass 

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.release()
            print("Application closed.")

    def _process_frame(self, frame):
        """
        Process a single video frame through the complete pipeline.

        Args:
            frame: BGR video frame.

        Returns:
            Processed frame with drawings, UI, and annotations.
        """
        # 1. Hand tracking
        results = self.tracker.process_frame(frame)
        landmarks = self.tracker.get_landmarks(results, frame.shape)
        hand_size = self.tracker.get_hand_size(landmarks)

        # 2. Gesture recognition
        state, pinch_pos = self.gesture.update(landmarks, hand_size)

        # 3. Handle state transitions
        current_time = time.time()

        if state == GestureState.DRAWING:
            # Check if pinch is in the UI region (toolbar area)
            if pinch_pos and pinch_pos[1] < self.ui.get_toolbar_height():
                # UI interaction
                if current_time - self._last_ui_action_time > self._ui_cooldown:
                    action = self.ui.hit_test(pinch_pos)
                    if action:
                        self._handle_ui_action(action)
                        self._last_ui_action_time = current_time
                # Don't draw in toolbar area
                if self.engine.is_drawing:
                    self.engine.end_stroke()
            else:
                # Drawing
                if not self.engine.is_drawing:
                    self.engine.start_stroke()
                if pinch_pos:
                    self.engine.add_point(pinch_pos)

        elif state == GestureState.IDLE:
            if self.engine.is_drawing:
                self.engine.end_stroke()

        if state != self._prev_state:
            print(f"State change: {self._prev_state} -> {state}")
            if state == GestureState.DRAWING:
                print("  --> STARTED DRAWING")
            elif state == GestureState.IDLE:
                print("  --> STOPPED DRAWING")

        # Check for spacebar override (Force Drawing)
        if self.spacebar_pressed:
            state = GestureState.DRAWING
            # Use index tip if not pinching, otherwise pinch mid-point
            if not pinch_pos and landmarks:
                pinch_pos = landmarks['index_tip']

        self._prev_state = state

        # 4. Composite layers
        # Draw hand landmarks (below canvas so they don't interfere)
        if self.show_landmarks:
            self.tracker.draw_landmarks(frame, results)

        # Overlay the drawing canvas
        frame = self.engine.composite(frame)

        # Draw the UI toolbar
        self.ui.render(frame)

        # Draw cursor indicator
        self._draw_cursor(frame, landmarks, state, pinch_pos)

        # Draw FPS counter
        self._draw_fps(frame)

        # Draw debug info
        if self.show_debug:
            self._draw_debug(frame, landmarks, state, pinch_pos)

        # Draw status bar
        self._draw_status_bar(frame)

        return frame

    def _draw_cursor(self, frame, landmarks, state, pinch_pos):
        """Draw cursor feedback at the fingertip / pinch position."""
        if landmarks is None:
            return

        index_tip = landmarks['index_tip']
        thumb_tip = landmarks['thumb_tip']

        if state == GestureState.DRAWING and pinch_pos:
            # Drawing mode: filled circle at pinch point
            cv2.circle(frame, pinch_pos, self.engine.current_thickness + 4,
                        self.engine.current_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pinch_pos, self.engine.current_thickness + 6,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Idle mode: show fingertip indicators
            cv2.circle(frame, index_tip, 8, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.circle(frame, thumb_tip, 8, (0, 200, 255), 2, cv2.LINE_AA)

            # Draw line between fingers showing proximity
            dist = np.sqrt(
                (index_tip[0] - thumb_tip[0]) ** 2 +
                (index_tip[1] - thumb_tip[1]) ** 2
            )
            # Color changes from green to red as fingers get closer
            ratio = min(1.0, dist / 80.0)
            line_color = (
                int(60 * ratio),
                int(255 * ratio),
                int(255 * (1 - ratio)),
            )
            cv2.line(frame, index_tip, thumb_tip, line_color, 1, cv2.LINE_AA)

    def _draw_fps(self, frame):
        """Draw FPS counter on the frame."""
        current_time = time.time()
        self._fps_times.append(current_time)

        # Keep last 30 frames for averaging
        self._fps_times = [t for t in self._fps_times if current_time - t < 1.0]

        if len(self._fps_times) > 1:
            self._fps = len(self._fps_times)

        # Draw FPS
        h = frame.shape[0]
        fps_text = f"FPS: {self._fps}"
        cv2.putText(frame, fps_text, (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 0), 2, cv2.LINE_AA)

    def _draw_status_bar(self, frame):
        """Draw status information at the bottom of the frame."""
        h, w = frame.shape[:2]
        strokes = self.engine.get_stroke_count()

        info = f"Strokes: {strokes} | Color: {self.ui.selected_color_name} | Size: {self.ui.selected_size_name}"
        cv2.putText(frame, info, (w - 450, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_debug(self, frame, landmarks, state, pinch_pos):
        """Draw debug information overlay."""
        info = self.gesture.get_pinch_info()
        h = frame.shape[0]

        debug_lines = [
            f"State: {info['state']}",
            f"Pinch Dist: {info['distance']:.1f}",
            f"Hand Size: {info['hand_size']:.1f}",
            f"Threshold: {info['threshold']:.1f}",
            f"Pinch Count: {info['pinch_counter']}",
            f"Release Count: {info['release_counter']}",
        ]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 180), (250, h - 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(debug_lines):
            y = h - 160 + i * 22
            cv2.putText(frame, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 200), 1, cv2.LINE_AA)

    def _handle_ui_action(self, action):
        """Handle a UI button press action."""
        if action['type'] == 'color':
            self.engine.set_color(action['color'])
            self.ui.set_selected_color(action['name'])
            print(f"  🎨 Color: {action['name']}")

        elif action['type'] == 'size':
            self.engine.set_thickness(action['size'])
            self.ui.set_selected_size(action['name'])
            print(f"  ✏️  Size: {action['name']}")

        elif action['type'] == 'clear':
            self.engine.clear_canvas()
            self.gesture.reset()
            print("  🧹 Canvas cleared!")

        elif action['type'] == 'undo':
            self.engine.undo()
            print("  ↩️  Undo!")

    def _handle_key(self, key):
        """Handle keyboard input."""
        if key == ord('q'):
            self.running = False
        elif key == 32:  # Spacebar
            self.spacebar_pressed = not self.spacebar_pressed
            state_str = "ON" if self.spacebar_pressed else "OFF"
            print(f"  ⌨️  Spacebar Drawing: {state_str}")
        elif key == ord('c'):
            self.engine.clear_canvas()
            self.gesture.reset()
            print("  🧹 Canvas cleared! (keyboard)")
        elif key == ord('u'):
            self.engine.undo()
            print("  ↩️  Undo! (keyboard)")
        elif key == ord('d'):
            self.show_debug = not self.show_debug
            print(f"  📊 Debug: {'ON' if self.show_debug else 'OFF'}")
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
            print(f"  🖐️ Landmarks: {'ON' if self.show_landmarks else 'OFF'}")


def main():
    """Entry point."""
    app = AirDrawingApp(
        camera_id=0,
        width=1280,
        height=720,
    )
    app.run()


if __name__ == '__main__':
    main()
