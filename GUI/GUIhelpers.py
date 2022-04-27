# GUIhelpers.py

import beepy
from GUI.GUI_layout import window

def userError(string):
    window["-FEEDBACK-"].update(string, text_color = "dark orange")
    beepy.beep(sound="error")

def userFeedback_OK(string):
    window["-FEEDBACK-"].update(string, text_color = "LightGreen")

def userFeedbackClear():
    window["-FEEDBACK-"].update("")

def updateField(field, string):
    window[field].update(string) # color not given, will be kept

def updateFieldOK(field, string):
    window[field].update(string, text_color = "LightGreen")    

def updateFieldDone(field, string):
    window[field].update(string, text_color = "Cyan")

def updateFieldOperation(field, string):
    updateFieldDone(field, string)      
