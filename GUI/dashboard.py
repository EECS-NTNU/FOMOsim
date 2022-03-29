# dashboard.py
import PySimpleGUI as sg

import copy
import policies
import sim

from tripStats.download import *
from tripStats.parse import calcDistances, get_initial_state

import ideal_state  

policyMenu = ["Do nothing", "bbb", "cccc"]

###### GUI layout
dashboardColumn = [
    [sg.Text("Preparation and set up ", font='Lucida', text_color = 'Yellow')],
    [sg.Text("Download Oslo trips, 1 = April 2019 ... 35 = February 2022)")],
    [sg.Button("All Oslo"), sg.Button("Clear"), sg.Input("From: ", key="-INPUTfrom-", size=8), 
        sg.Input("To: ", key="-INPUTto-", size = 6)],
    [sg.Button("Download Oslo")],
    [sg.Text('_'*40)],
    [sg.Text("Select city (Oslo is default)")],
    [sg.Radio("Oslo", "RADIO1", key = "-OSLO-"), sg.Radio("Bergen", "RADIO1", key = "-BERGEN-"), 
        sg.Radio("Utopia", "RADIO1", key = "-UTOPIA-")], 
    [sg.Button("Find stations and distances"), sg.Text(size=(30, 1), key="-CITY-STATUS-", text_color = "Red")],
    [sg.Text('_'*40)],
    [sg.Button("Set initial state"), sg.Input("Week no: ", key="-WEEK-", size=12)],
    [sg.Text(size=(30, 1), key="-SIMULATE-SETUP-", text_color = "Red")],
    [sg.Text('_'*40)],
    [sg.Text("Select policy: "), sg.Listbox( values=policyMenu, enable_events=True, size=(20, 5), key="-POLICIES-")],
    [sg.Button("Simulate")],
    [sg.Text('_'*40)],     
    [sg.Button("Exit")]
]
statusColumn = [
    [sg.Text("Simulation status", font='Lucida', text_color = 'Yellow', key="-TOUT1-")],
    [sg.Text('_'*40)],
    [sg.Text("Visualize")],
    [sg.Text('_'*40)],
]
layout = [ [sg.Column(dashboardColumn), sg.VSeperator(), sg.Column(statusColumn) ] ]
window = sg.Window("FOMO Digital Twin Dashboard 0.1", layout)

DURATION = 960 # change to input-field with default value

def startSimulation(simPolicy, state):
    print("simulate...", simPolicy)
    if simPolicy == "Do nothing": 
        simulator = sim.Simulator( 
            DURATION,
            policies.DoNothing(),
            copy.deepcopy(state),
            verbose=True,
            label="DoNothing",
        )
        simulator.run()

def GUI_main():
    print("GUI main called")
    state = sim.State()
    while True:
        GUI_event, GUI_values= window.read()
        if GUI_event == "All Oslo":
            print("All-Oslo-button pressed")
            window["-INPUTfrom-"].update("From: 1")
            window["-INPUTto-"].update("To: 35")  # TODO, Magic number, move to local settings ?? 
        elif GUI_event == "Clear":
            window["-INPUTfrom-"].update("From: ")
            window["-INPUTto-"].update("To: ")   
        elif GUI_event == "Download Oslo":
            oslo(GUI_values["-INPUTfrom-"], GUI_values["-INPUTto-"])
        elif GUI_event == "Find stations and distances":
            if GUI_values["-OSLO-"]:
                window["-CITY-STATUS-"].update("City OK", text_color = "LightGreen")
                calcDistances("Oslo")
            elif GUI_values["-BERGEN-"]:
                window["-CITY-STATUS-"].update("City not yet implemented", text_color = "red") 
            elif GUI_values["-UTOPIA-"]:
                window["-CITY-STATUS-"].update("City OK", text_color = "LightGreen") 
                calcDistances("Utopia")
            else:
                print("*** Error: wrong value from Radiobutton")         
        elif GUI_event == "Set initial state":
            if GUI_values["-WEEK-"] == "Week no: ":
                weekNo = 53
                window["-WEEK-"].update("Week no: 53") 
            else:
                weekNo = int(strip("Week no: ", GUI_values["-WEEK-"]))
            pass
            if GUI_values["-OSLO-"]:
                state = get_initial_state("Oslo", week = weekNo)
                new_ideal_state = ideal_state.evenly_distributed_ideal_state(state)
                state.set_ideal_state(new_ideal_state)
            if GUI_values["-UTOPIA-"]:
                window["-WEEK-"].update("Week no: 48") 
                state = get_initial_state("Utopia", week = 48)
                pass
                new_ideal_state = ideal_state.evenly_distributed_ideal_state(state)
                state.set_ideal_state(new_ideal_state)
        elif GUI_event == "Simulate":
            if len(state.stations) == 0:
                print("You must set an initial state")
                window["-SIMULATE-SETUP-"].update("You must set an initial state", text_color = "red") 
            else:
                startSimulation(simPolicy, state)
                state = sim.State()

        if GUI_values["-POLICIES-"] != []:
            simPolicy = GUI_values["-POLICIES-"][0]
        if GUI_event == "Exit" or GUI_event == sg.WIN_CLOSED:
            break
    window.close()