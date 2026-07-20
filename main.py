"""
Gesture-Based Master System Volume Controller
Entry point for launching the application following MVC pattern.
"""
from src.controllers.main_controller import MainController

def main():
    controller = MainController(min_dist=20, max_dist=200)
    controller.run()

if __name__ == "__main__":
    main()
