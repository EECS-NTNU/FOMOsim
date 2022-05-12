# dashboard.py

import os
import socket
import PySimpleGUI as sg
import matplotlib # TODO MARK-C

import settings

# import init_state.cityBike
from init_state.cityBike.helpers import dateAndTimeStr, strip, write, fixComputerName, get_duration
from init_state.cityBike.analyze import openVisual1, openVisual2, openVisual3, openVisual4

from GUI import loggFile
from GUI.script import Session, doCommand, replayScript
from GUI.GUI_layout import window
from GUI.GUIhelpers import *

def GUI_main():
    matplotlib.use("TkAgg") # TODO MARK-C
    session = Session("GUI_main_session")
    betterName = fixComputerName(socket.gethostname())
    write(loggFile, ["Session-start:", session.name, dateAndTimeStr(), "Computer:", betterName]) 
    task = [] # TODO, only one task allowed in queue at the moment
    readyForTask = False # used together with timeout to ensure one iteration in loop for 
    resultFile = ""                     
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
            updateField("-INPUTfrom-", "From: 1")
            updateField("-INPUTto-", "To: 35")  # TODO, Magic number, move to local settings ? 
        elif GUI_event == "Clear":
            updateField("-INPUTfrom-", "From: ")
            updateField("-INPUTto-", "To: ")   
        elif GUI_event == "Download Oslo": # TODO, should maybe be handled like time-consuming tasks, or give warning
            task = ["Download-Oslo", GUI_values["-INPUTfrom-"], GUI_values["-INPUTto-"]]
            updateFieldOperation("-FEEDBACK-", "Lengthy operation started (see terminal)") 

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
                userFeedbackClear()
                weekNo = 0
                if GUI_values["-WEEK-"] == "Week no: ": # TODO, improve code, make function, reuse
                    weekNo = 53
                    updateField("-WEEK-", "Week no: 53") 
                else:
                    if GUI_values["-WEEK-"] == "Week no: -na-":
                        updateField("-WEEK-", "Week no: ")
                        userError("You must select a week no")
                    else:    
                        weekNo = int(strip("Week no: ", GUI_values["-WEEK-"]))
                if weekNo != 0 :
                    task = ["Init-state-FH", "Oslo", str(weekNo)]    
                    updateFieldOperation("-STATE-MSG-", "Lengthy operation started ... (4 - 6 minutes)") 
            elif GUI_values["-UTOPIA-"]: # This is (still) quick
                userFeedbackClear()
                updateField("-WEEK-", "Week no: 48") # Only week with traffic at the moment for Utopia
                task = ["Init-state-FH", "Utopia", "48"]    
                updateFieldOperation("-STATE-MSG-", "short operation started ...") 
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented")
            else:
                userError("You must select a city")
            updateField("-CALC-MSG-", "")
        elif GUI_event == "Haflan, Haga & Spetalen": # handled here since it is relativelu quick
            task = ["Init-state-HHS"] 
            updateField("-WEEK-","Week no: -na-")
            window["-OSLO-"].update(True) # entur data are from Oslo
            window["-UTOPIA-"].update(False)
            updateField("-CALC-MSG-","")
            updateFieldOperation("-STATE-MSG-", "Lengthy operation started ... (see progress in terminal)")
        elif GUI_event == "Test state": 
            task = ["Init-test-state", "Small-Circle"] 
        elif GUI_event == "Save state":
            fileName = strip("Name: ", GUI_values["-INPUTname-"])
            task = ["Save-state", fileName]
        elif GUI_event == "Load state":
            task = ["Load-state"] # TODO not implemented

        ###### IDEAL STATE GUI PART   
        elif GUI_event == "Evenly distributed":
            task = ["Ideal-state-evenly-distributed"]
            if session.initStateType == "HHS":
                updateFieldOperation("-CALC-MSG-", "Lengthy operation started ... (see progress in terminal)")
        elif GUI_event == "Outflow":
            task = ["Ideal-state-outflow"]
            if session.initStateType == "FH":
                updateFieldOperation("-CALC-MSG-", "Lengthy operation started ... (see progress in terminal)")
    
        ###### SIMULATE GUI PART
        elif GUI_event == "Simulate":
            if session.simPolicy == "":
                userError("You must select a policy")
            else:
                startDay = int(strip("Start-day:", GUI_values["-START-D-"]))
                startHour = int(strip("Start-hour:", GUI_values["-START-H-"]))
                numDays = int(strip("#days:", GUI_values["-NUM-DAYS-"]))
                numHours = int(strip("#hours:", GUI_values["-NUM-HOURS-"]))
                period = get_duration(numDays, numHours)
                startTime = (24*startDay + startHour)*60 
                if GUI_values["-LOGG-TRAFFIC-"] == True:
                    settings.TRAFFIC_LOGGING = True
                else:
                    settings.TRAFFIC_LOGGING = False                
                task = ["Sim", session.simPolicy, str(startTime), str(period)]
                updateFieldOperation("-SIM-MSG-", "Simulation started ...  (see progress in terminal)")

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
            window["-RESULT-FILES-"].update(fnames)

        elif GUI_event == "Test-1":
            openVisual1()
        elif GUI_event == "Test-2":
            openVisual2(resultFile)
        elif GUI_event == "Test-3":
            openVisual3()
        elif GUI_event == "Trips/week-Oslo":
            openVisual4()

        if GUI_values["-POLICIES-"] != []:
            session.simPolicy = GUI_values["-POLICIES-"][0]
        if GUI_values["-RESULT-FILES-"] != []:
            resultFile = GUI_values["-FOLDER-"] + "/" + GUI_values["-RESULT-FILES-"][0]
            
