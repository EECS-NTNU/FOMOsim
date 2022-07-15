#!/bin/python3
"""
FOMO simulator GUI main program
"""

from GUI.dashboard import GUI_main
import main

if __name__ == "__main__":
    if not GUI_main():
        main.main()
