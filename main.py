"""
Real-Time Hand Finger Counter Application
Entry point for launching the application following MVC pattern.
"""
from src.controllers.main_controller import MainController

def main():
    controller = MainController()
    controller.run()

if __name__ == "__main__":
    main()
