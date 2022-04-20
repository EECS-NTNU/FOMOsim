# dashboard.py
"""
GUI dashboard using PySimpleGUI
"""

import os
import PySimpleGUI as sg
import beepy
import tripStats
from tripStats.helpers import strip
from tripStats.analyze import openVisual1, openVisual2, openVisual3, openVisual4

from GUI.script import Session, doCommand, replayScript, policyMenu

################################ GUI LAYOUT CODE
colWidth = 55
dashboardColumn = [
    [sg.Text("Prep. and set up ", font='Lucida', text_color = 'Yellow'), sg.VSeparator(), 
        sg.Button("Fast-Track", button_color = "forest green"), sg.Button("main.py", button_color="snow4"), sg.Button("Exit")],
    [sg.Text("Set up simulation", font="Helvetica 14", size=(30, 1), text_color = "spring green", key="-FEEDBACK-")],
    [sg.Text('_'*colWidth)],
    [sg.Text("Download Oslo trips, 1 = April 2019 ... 35 = February 2022)")],
    [sg.Button("All Oslo"), sg.Button("Clear"), sg.Input("From: ", key="-INPUTfrom-", size=8), 
        sg.Input("To: ", key="-INPUTto-", size = 6), sg.Button("Download Oslo")],
    [sg.Text('_'*colWidth)],
    [sg.Text("Select city (Oslo is default, only relevant for F & H)")],
    [sg.Radio("Oslo", "RADIO1", key = "-OSLO-"), sg.Radio("Bergen", "RADIO1", key = "-BERGEN-"), 
        sg.Radio("Utopia", "RADIO1", key = "-UTOPIA-"), sg.Button("Find stations and distances")],
    [sg.Text('_'*colWidth)],
    [sg.Text("Set initial state"), sg.Text("", key = "-STATE-MSG-")],
    [sg.Button("Fosen & Haldorsen"), sg.Input("Week no: ", key="-WEEK-", size=12), sg.VSeparator(), 
        sg.Button("Haflan, Haga & Spetalen")],
    [sg.Text('_'*colWidth)],
    [sg.Text("Method for calculating ideal state: "), sg.Text("...", key="-IDEAL-METHOD-")],
    [sg.Button("Calculate"), sg.Text("", key="-CALC-MSG-")], 
    [sg.Text('_'*colWidth)],
    [sg.Button("Test state"), sg.Button("Store state"), sg.Input("State-name: ", key ="-NAME-", size = 30)],    
    [sg.Text('_'*colWidth)],
    [sg.Text("Select policy: "), sg.Listbox( values=policyMenu, enable_events=True, size=(17, 4), key="-POLICIES-"), 
        sg.Text("Hours: 1", size = 11, key="-HOURS-")],
    [sg.Button("Simulate"), sg.Button("Replay script")],
    [sg.Text("", key="-SIM-MSG-")],    
]
statusColumn = [
    [sg.Text("Simulation status", font='Lucida', text_color = 'Yellow')],
    [sg.Text('_'*50)],
    [sg.Text("Simulation progress:"), sg.Text("",key="-START-TIME-"), sg.Text("",key="-END-TIME-")],
    [sg.Button("Timestamp and save session results"), sg.Button("Save session as script")],
    [sg.Text('_'*50)],
    [sg.Text("Visualization of results (not ready)", font='Lucida', text_color = 'Yellow')],
    [sg.Text("Choose folder for fomo results files")],
    [sg.Text("Specify results Folder:")],  
    [sg.Input(size=(42, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse()],
    [sg.Listbox( values=[], enable_events=True, size=(30, 8), key="-FILE LIST-")],
    [sg.Text("The following checkboxes are still not implemented")],
    [sg.Checkbox ('Option 1', key='check_value1'), sg.Checkbox ('Option 2',key='check_value2'), sg.Checkbox ('Grid',key='check_value2')],
    [sg.Button("Test-1"), sg.Button("Test-2"), sg.Button("Test-3"), sg.Button("Trips/week-Oslo")],
    [sg.Text('_'*50)],
]
layout = [ [sg.Column(dashboardColumn), sg.VSeperator(), sg.Column(statusColumn) ] ]
window = sg.Window("FOMO Digital Twin Dashboard 0.2", layout)

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

#########################################

def GUI_main():
    session = Session("GUI_main_session") 
    task = [] # TODO, only one task allowed in queue at the moment
    readyForTask = False # used together with timeout to ensure one iteration in loop for 
                         # updating field -SIM-MSG- or -STATE-MSG- before starting long operation
    while True:
        GUI_event, GUI_values= window.read(timeout = 100) # waits 100 millisecs before looking in taskQueue
#        GUI_event, GUI_values= window.read() 
        if len(task) > 0: ###### handling of lengthy operations is done in a two-stage process to be able to give message
            if not readyForTask:
                readyForTask = True
            else:
                doCommand(session, task) # TODO opprett session passende sted
                readyForTask = False
                task = []   
        
        ###### SPECIAL BUTTONS GUI PART
        if GUI_event == "Fast-Track":
            userError("No code currently placed in FastTrack")
            #-------------
            # bigOsloTest()
            #-------------            
            # session = Session("Fast-Track-session")
            # replayScript(session, "tripStats/scripts/script.txt")
        elif GUI_event == "main.py":
            print("Leaves GUI-dashboard-code, continues in main.py")
            window.close()
            return False    
        elif GUI_event == "Exit" or GUI_event == sg.WIN_CLOSED:
            return True

        ###### DOWNLOAD GUI PART
        elif GUI_event == "All Oslo":
            window["-INPUTfrom-"].update("From: 1")
            window["-INPUTto-"].update("To: 35")  # TODO, Magic number, move to local settings ?? 
        elif GUI_event == "Clear":
            window["-INPUTfrom-"].update("From: ")
            window["-INPUTto-"].update("To: ")   
        elif GUI_event == "Download Oslo": # TODO, should maybe be handled like time-consuming tasks, or give warning
            task = ["Download-Oslo", GUI_values["-INPUTfrom-"], GUI_values["-INPUTto-"]]
            window["-FEEDBACK-"].update("Lengthy operation started (see terminal)", text_color="cyan") 

        ###### SELECT CITY GUI PART     
        elif GUI_event == "Find stations and distances":
            if GUI_values["-OSLO-"]:
                task = ["Find-stations", "Oslo"]
                updateFieldOperation("-FEEDBACK-", "Lengthy operation started ...") 
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented") 
            elif GUI_values["-UTOPIA-"]:
                task = ["Find-stations", "Utopia"]
                updateFieldOperation("-FEEDBACK-", "short operation started ...")
            else:
                print("*** Error: wrong value from Radiobutton")         
        
        ###### INIT STATE GUI PART
        elif GUI_event == "Fosen & Haldorsen":
            if GUI_values["-OSLO-"]:
                weekNo = 0
                if GUI_values["-WEEK-"] == "Week no: ": # TODO, improve code, make function, reuse
                    weekNo = 53
                    window["-WEEK-"].update("Week no: 53") 
                else:
                    if GUI_values["-WEEK-"] == "Week no: -na-":
                        window["-WEEK-"].update("Week no: ")
                        userError("You must select a week no")
                    else:    
                        weekNo = int(strip("Week no: ", GUI_values["-WEEK-"]))
                if weekNo != 0 :
                    task = ["Init-state-FH", "Oslo", str(weekNo)]    
                    updateFieldOperation("-STATE-MSG-", "Lengthy operation started ... (4 - 6 minutes)") 
            elif GUI_values["-UTOPIA-"]: # This is (still) quick
                window["-WEEK-"].update("Week no: 48") # Only week with traffic at the moment for Utopia
                task = ["Init-state-FH", "Utopia", "48"]    
                updateFieldOperation("-STATE-MSG-", "short operation started ...") 
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented")
            else:
                userError("You must select a city")
            window["-IDEAL-METHOD-"].update("Fosen & Haldorsen")
            window["-CALC-MSG-"].update("")
        elif GUI_event == "Haflan, Haga & Spetalen": # handled here since it is relativelu quick
            task = ["Init-state-HHS"] 
            window["-WEEK-"].update("Week no: -na-")
            window["-IDEAL-METHOD-"].update("Haflan, Haga & Spetalen")
            window["-OSLO-"].update(True) # entur data are from Oslo
            window["-UTOPIA-"].update(False)
            window["-CALC-MSG-"].update("")
            window["-STATE-MSG-"].update("Lengthy operation started ... (see progress in terminal)", text_color="cyan")
        elif GUI_event == "Test state":
            task = ["Init-manual", "Small-Circle"]
        ###### IDEAL STATE GUI PART   
        elif GUI_event == "Calculate":
            task = ["Ideal-state"]
            if session.initStateType == "HHS":
                window["-CALC-MSG-"].update("Lengthy operation started ... (see progress in terminal)", text_color="cyan")
        elif GUI_event == "Test state":
            task = ["Init-manual", "Small-Circle"]
        elif GUI_event == "Store state":
            pass
        ###### SIMULATE GUI PART
        elif GUI_event == "Simulate":
            if session.simPolicy == "":
                userError("You must select a policy")
            else:
                task = ["Sim", session.simPolicy]
                # window["-WEEK-"].update("Week no: ") # TODO, usikker på denne, henger igjen
                window["-SIM-MSG-"].update("Simulation started ...  (see progress in terminal)", text_color="cyan")

        elif GUI_event == "Replay script":
            replayScript()

        #####################################################
        elif GUI_event == "-FOLDER-": # Folder name was filled in, make a list of files in the folder
            folder = GUI_values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".txt"))
            ]
            window["-FILE LIST-"].update(fnames)

        elif GUI_event == "Test-1":
            openVisual1()
        elif GUI_event == "Test-2":
            openVisual2()
        elif GUI_event == "Test-3":
            openVisual3()
        elif GUI_event == "Trips/week-Oslo":
            openVisual4()

        if GUI_values["-POLICIES-"] != []:
            session.simPolicy = GUI_values["-POLICIES-"][0]