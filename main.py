"""
Real-Time Hand Finger Counter Application
Entry point for launching the application following MVC pattern.
"""
import argparse
import sys
from typing import List, Optional
from src.controllers.main_controller import MainController
from src.config import AppSettings


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser for runtime settings."""
    parser = argparse.ArgumentParser(description="Real-time hand finger counter")
    parser.add_argument("--camera", type=int, default=AppSettings.camera_index, help="Camera index to open")
    parser.add_argument("--width", type=int, default=AppSettings.window_width, help="Capture width")
    parser.add_argument("--height", type=int, default=AppSettings.window_height, help="Capture height")
    parser.add_argument("--max-hands", type=int, default=AppSettings.max_hands, help="Maximum hands to track")
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=AppSettings.min_detection_confidence,
        help="MediaPipe minimum detection confidence"
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=AppSettings.min_tracking_confidence,
        help="MediaPipe minimum tracking confidence"
    )
    parser.add_argument("--smoothing-window", type=int, default=AppSettings.smoothing_window, help="Frames used for vote smoothing")
    parser.add_argument("--hide-diagnostics", action="store_true", help="Start with diagnostics panel hidden")
    parser.add_argument("--hide-controls", action="store_true", help="Start with keyboard shortcut hint hidden")
    parser.add_argument("--hide-skeleton", action="store_true", help="Start with hand skeleton overlay hidden")
    return parser


def parse_settings(argv: Optional[List[str]] = None) -> AppSettings:
    """Parse runtime settings from CLI arguments."""
    args = build_parser().parse_args(argv)
    return AppSettings(
        camera_index=args.camera,
        window_width=args.width,
        window_height=args.height,
        max_hands=args.max_hands,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        smoothing_window=max(1, args.smoothing_window),
        show_diagnostics=not args.hide_diagnostics,
        show_controls=not args.hide_controls,
        show_skeleton=not args.hide_skeleton,
    )


def main(argv: Optional[List[str]] = None):
    try:
        controller = MainController(parse_settings(argv))
        controller.run()
    except Exception as e:
        print(f"\n[Error] Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
