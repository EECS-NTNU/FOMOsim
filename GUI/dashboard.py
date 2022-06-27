# dashboard.py

import socket
import PySimpleGUI as sg
import matplotlib # TODO MARK-C

import settings

from init_state.cityBike.helpers import dateAndTimeStr, strip, write, fixComputerName, get_duration

from GUI import loggFile
from GUI.script import Session, doCommand, replayScript
from GUI.GUI_layout import window
from GUI.GUIhelpers import *

def GUI_main():
    matplotlib.use("TkAgg") # TODO MARK-C
    session = Session("GUI_main_session")
    betterName = fixComputerName(socket.gethostname())
    write(loggFile, ["Session-start:", session.name, dateAndTimeStr(), "Computer:", betterName]) 
    task = [] # TODO, currently only one task allowed in queue 
    readyForTask = False # used together with timeout to ensure one iteration in loop         
    while True:
        GUI_event, GUI_values= window.read(timeout = 50) # waits some millisecs before looking in taskQueue
        if len(task) > 0: ###### handling of lengthy operations is done in a two-stage process to be able to give rapid feedback in GUI
            if not readyForTask:
                readyForTask = True
            else:
                doCommand(session, task) 
                readyForTask = False
                task = []   
        
        ###### SPECIAL BUTTONS GUI PART
        if GUI_event == "Fast-Track":  
            userError("No code currently placed in FastTrack")

        elif GUI_event == "main.py":
            print("Leaves GUI-dashboard-code, continues in main.py")
            window.close()
            return False    
        elif GUI_event == "Exit" or GUI_event == sg.WIN_CLOSED:
            return True

        ###### DOWNLOAD GUI PART
        elif GUI_event == "Oslo": 
            task = ["Download", "https://data.urbansharing.com/oslobysykkel.no/trips/v1/"]
            updateFieldOperation("-FEEDBACK-", "Lengthy operation started (see terminal)") 
        elif GUI_event == "Oslo vinter":
            task = ["Download", "https://data.urbansharing.com/oslovintersykkel.no/trips/v1/"]
            updateFieldOperation("-FEEDBACK-", "Lengthy operation started (see terminal)") 
        elif GUI_event == "Bergen":
            task = ["Download", "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/"]
            updateFieldOperation("-FEEDBACK-", "Lengthy operation started (see terminal)") 
        elif GUI_event == "Edinburgh":
            task = ["Download", "https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/"]
            updateFieldOperation("-FEEDBACK-", "Lengthy operation started (see terminal)") 
        elif GUI_event == "Trondheim":
            task = ["Download", "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/"]
            updateFieldOperation("-FEEDBACK-", "Lengthy operation started (see terminal)") 

        ###### SELECT CITY GUI PART     
        elif GUI_event == "Stations and distances":
            if GUI_values["-OSLO-"]:
                task = ["Find-stations", "Oslo"]
                updateFieldOperation("-FEEDBACK-", "Lengthy operation started ...") 
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented") 
            elif GUI_values["-UTOPIA-"]:
                task = ["Find-stations", "Utopia"]
                updateFieldOperation("-FEEDBACK-", "short operation started ...")
            else:
                userError("You must select a city") 
        
        ###### INIT STATE GUI PART
        elif GUI_event == "CityBike":
            if GUI_values["-OSLO-"]:
                userFeedbackClear()
                weekNo = 0
                if GUI_values["-WEEK-"] == "Week no: ": 
                    weekNo = 53
                    updateField("-WEEK-", "Week no: 53") 
                else:
                    if GUI_values["-WEEK-"] == "Week no: -na-":
                        updateField("-WEEK-", "Week no: ")
                        userError("You must select a week no")
                    else:    
                        weekNo = int(strip("Week no: ", GUI_values["-WEEK-"]))
                if weekNo != 0 :
                    updateFieldOperation("-STATE-MSG-", "Lengthy operation started ... (4 - 6 minutes)") 
                    task = ["Init-state-CityBike", "Oslo", str(weekNo)]    
            elif GUI_values["-UTOPIA-"]: 
                userFeedbackClear()
                updateField("-WEEK-", "Week no: 48") # Only one week with traffic at the moment for Utopia (3 trips)
                task = ["Init-state-CityBike", "Utopia", "48"]    
                updateFieldOperation("-STATE-MSG-", "short operation started ...") 
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented")
            else:
                userError("You must select a city")
            updateField("-CALC-MSG-", "")
        elif GUI_event == "Entur": 
            userFeedbackClear()
            task = ["Init-state-Entur"] 
            updateField("-WEEK-","Week no: -na-")
            window["-OSLO-"].update(True) # entur data are from Oslo, indicate it in GUI
            window["-UTOPIA-"].update(False)
            updateField("-CALC-MSG-","")
            updateFieldOperation("-STATE-MSG-", "Lengthy operation started ... (see progress in terminal)")
        elif GUI_event == "US-init":
            userFeedbackClear()
            updateFieldOperation("-STATE-MSG-", "Lengthy operation started ...")
            task = ["Init-state-US"]         
        elif GUI_event == "Test-state": 
            userFeedbackClear()
            task = ["Init-state-test", "allToAll4"] 

        ###### TARGET STATE GUI PART   
        elif GUI_event == "Evenly":
            task = ["Target-state-evenly-distributed"]
            userFeedbackClear()
            updateFieldOperation("-CALC-MSG-", "Lengthy operation started ...")        
        elif GUI_event == "Outflow":
            task = ["Target-state-outflow"]
            userFeedbackClear()
            updateFieldOperation("-CALC-MSG-", "Lengthy operation started ...")        
        elif GUI_event == "US":
            task = ["Target-state-US"]
            userFeedbackClear()
            updateFieldOperation("-CALC-MSG-", "Lengthy operation started ... (see progress in terminal)")        

        ###### SAVE AND LOAD STATE PART
        elif GUI_event == "Save state":
            userFeedbackClear()
            fileName = strip("Name: ", GUI_values["-INPUTname-"])
            if fileName == "":
                fileName = "SavedState"
            task = ["Save-state", fileName]
        elif GUI_event == "Load state":
            fileName = strip("Name: ", GUI_values["-INPUTname-"])
            userFeedbackClear()
            task = ["Load-state", fileName] 

        ###### SIMULATE GUI PART
        elif GUI_event == "Simulate":
            if session.simPolicy == "":
                userError("You must select a policy")
            else:
                userFeedbackClear()
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

        if GUI_values["-POLICIES-"] != []:
            session.simPolicy = GUI_values["-POLICIES-"][0]
    