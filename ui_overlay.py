"""
UI Overlay Module
Renders virtual buttons, color palette, and brush size controls.
Provides hit-testing for touch-free interactions.
"""

import cv2
import numpy as np


class UIOverlay:
    """
    Virtual UI overlay for the Air Drawing application.

    Renders a toolbar at the top of the screen with:
    - Color palette swatches
    - Clear canvas button
    - Brush size controls
    - Active indicator (shows current color/size)
    """

    # Color palette (BGR format)
    COLORS = {
        'White': (255, 255, 255),
        'Red': (60, 60, 255),
        'Green': (80, 220, 80),
        'Blue': (255, 140, 50),
        'Yellow': (50, 230, 255),
        'Cyan': (230, 220, 50),
        'Magenta': (200, 60, 220),
        'Orange': (40, 140, 255),
    }

    # Brush sizes
    BRUSH_SIZES = {
        'S': 2,
        'M': 4,
        'L': 8,
        'XL': 14,
    }

    TOOLBAR_HEIGHT = 80
    BUTTON_SIZE = 50
    BUTTON_MARGIN = 12
    BUTTON_Y_OFFSET = 15

    def __init__(self, frame_width, frame_height):
        """
        Initialize the UI overlay.

        Args:
            frame_width: Width of the video frame.
            frame_height: Height of the video frame.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.selected_color_name = 'White'
        self.selected_size_name = 'M'

        # Build button layout
        self._buttons = []
        self._build_layout()

    def _build_layout(self):
        """Calculate button positions."""
        self._buttons = []
        x = self.BUTTON_MARGIN
        y = self.BUTTON_Y_OFFSET

        # Color buttons
        for name, color in self.COLORS.items():
            self._buttons.append({
                'type': 'color',
                'name': name,
                'color': color,
                'x': x, 'y': y,
                'w': self.BUTTON_SIZE, 'h': self.BUTTON_SIZE,
            })
            x += self.BUTTON_SIZE + self.BUTTON_MARGIN

        # Separator space
        x += self.BUTTON_MARGIN

        # Brush size buttons
        for name, size in self.BRUSH_SIZES.items():
            self._buttons.append({
                'type': 'size',
                'name': name,
                'size': size,
                'color': (200, 200, 200),
                'x': x, 'y': y,
                'w': self.BUTTON_SIZE, 'h': self.BUTTON_SIZE,
            })
            x += self.BUTTON_SIZE + self.BUTTON_MARGIN

        # Separator space
        x += self.BUTTON_MARGIN

        # Clear button
        clear_w = self.BUTTON_SIZE * 2
        self._buttons.append({
            'type': 'clear',
            'name': 'CLEAR',
            'color': (80, 80, 240),
            'x': x, 'y': y,
            'w': clear_w, 'h': self.BUTTON_SIZE,
        })
        x += clear_w + self.BUTTON_MARGIN

        # Undo button
        undo_w = self.BUTTON_SIZE * 2
        self._buttons.append({
            'type': 'undo',
            'name': 'UNDO',
            'color': (180, 140, 60),
            'x': x, 'y': y,
            'w': undo_w, 'h': self.BUTTON_SIZE,
        })

    def render(self, frame):
        """
        Render the UI overlay onto the frame.

        Args:
            frame: BGR video frame to draw onto (modified in-place).
        """
        # Draw semi-transparent toolbar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.TOOLBAR_HEIGHT),
                       (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw thin separator line
        cv2.line(frame, (0, self.TOOLBAR_HEIGHT),
                 (self.frame_width, self.TOOLBAR_HEIGHT), (80, 80, 80), 2)

        for btn in self._buttons:
            x, y, w, h = btn['x'], btn['y'], btn['w'], btn['h']

            if btn['type'] == 'color':
                # Color swatch
                cv2.rectangle(frame, (x, y), (x + w, y + h), btn['color'], -1)

                # Selection indicator
                if btn['name'] == self.selected_color_name:
                    cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3),
                                   (255, 255, 255), 3)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

            elif btn['type'] == 'size':
                # Brush size button
                is_selected = btn['name'] == self.selected_size_name
                bg_color = (100, 100, 100) if is_selected else (50, 50, 50)
                cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)

                # Draw size circle preview
                center = (x + w // 2, y + h // 2)
                radius = btn['size'] + 1
                cv2.circle(frame, center, radius, (255, 255, 255), -1, cv2.LINE_AA)

                # Selection border
                if is_selected:
                    cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2),
                                   (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

                # Size label
                cv2.putText(frame, btn['name'],
                            (x + w // 2 - 6, y + h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (200, 200, 200), 1, cv2.LINE_AA)

            elif btn['type'] == 'clear':
                # Clear button
                cv2.rectangle(frame, (x, y), (x + w, y + h), btn['color'], -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 255), 2)
                cv2.putText(frame, 'CLEAR',
                            (x + 12, y + h // 2 + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)

            elif btn['type'] == 'undo':
                # Undo button
                cv2.rectangle(frame, (x, y), (x + w, y + h), btn['color'], -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 180, 100), 2)
                cv2.putText(frame, 'UNDO',
                            (x + 15, y + h // 2 + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)

    def get_ui_regions(self):
        """
        Get all button regions for hit-testing.

        Returns:
            List of (x, y, w, h, action_dict) tuples.
        """
        regions = []
        for btn in self._buttons:
            action = {
                'type': btn['type'],
                'name': btn['name'],
            }
            if btn['type'] == 'color':
                action['color'] = btn['color']
            elif btn['type'] == 'size':
                action['size'] = btn['size']
            regions.append((btn['x'], btn['y'], btn['w'], btn['h'], action))
        return regions

    def hit_test(self, pos):
        """
        Check if a position hits any UI button.

        Args:
            pos: Tuple (x, y) position to test.

        Returns:
            Action dict if a button is hit, None otherwise.
        """
        if pos is None:
            return None

        px, py = pos
        for btn in self._buttons:
            x, y, w, h = btn['x'], btn['y'], btn['w'], btn['h']
            if x <= px <= x + w and y <= py <= y + h:
                action = {'type': btn['type'], 'name': btn['name']}
                if btn['type'] == 'color':
                    action['color'] = btn['color']
                elif btn['type'] == 'size':
                    action['size'] = btn['size']
                return action
        return None

    def set_selected_color(self, name):
        """Update the selected color indicator."""
        if name in self.COLORS:
            self.selected_color_name = name

    def set_selected_size(self, name):
        """Update the selected size indicator."""
        if name in self.BRUSH_SIZES:
            self.selected_size_name = name

    def get_toolbar_height(self):
        """Return the toolbar height for drawing area calculations."""
        return self.TOOLBAR_HEIGHT
