"""
Real-Time Hand Finger Counter Application
Entry point for launching the application following MVC pattern.
"""
import sys
from src.controllers.main_controller import MainController

def main():
    try:
        controller = MainController()
        controller.run()
    except Exception as e:
        print(f"\n[Error] Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
