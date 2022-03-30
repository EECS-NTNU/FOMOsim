# dashboard.py

import copy
import policies
import policies.fosen_haldorsen
import sim
import ideal_state  

from tripStats.download import *
from tripStats.parse import calcDistances, get_initial_state
from tripStats.helpers import readTime
import PySimpleGUI as sg
import beepy

policyMenu = ["Do nothing", "Rebalancing", "Fosen & Haldorsen", "F & H Greedy"]

###### GUI layout
dashboardColumn = [
    [sg.Text("Preparation and set up ", font='Lucida', text_color = 'Yellow'), [sg.Button("Exit")]],
    [sg.Text("User feedback: ", font="Helvetica 14"), sg.Text("Set up simulation", font="Helvetica 14", size=(30, 1), text_color = "spring green", key="-FEEDBACK-")],
    [sg.Text('_'*40)],
    [sg.Text("Download Oslo trips, 1 = April 2019 ... 35 = February 2022)")],
    [sg.Button("All Oslo"), sg.Button("Clear"), sg.Input("From: ", key="-INPUTfrom-", size=8), 
        sg.Input("To: ", key="-INPUTto-", size = 6)],
    [sg.Button("Download Oslo")],
    [sg.Text('_'*40)],
    [sg.Text("Select city (Oslo is default)")],
    [sg.Radio("Oslo", "RADIO1", key = "-OSLO-"), sg.Radio("Bergen", "RADIO1", key = "-BERGEN-"), 
        sg.Radio("Utopia", "RADIO1", key = "-UTOPIA-")], 
    [sg.Button("Find stations and distances")],
    [sg.Text('_'*40)],
    [sg.Button("Set initial state"), sg.Input("Week no: ", key="-WEEK-", size=12), 
        sg.Text("", key="-STATE-CITY-"), sg.Text("", key="-STATE-MSG-")],
    [sg.Text('_'*40)],
    [sg.Text("Select ideal state")],
    [sg.Radio("Evenly distributed", "RADIO2", key = "-EVENLY-"), sg.Radio("Haflan, Haga & Spetalen", "RADIO2", key = "-HHS-")], 
    [sg.Text('_'*40)],
    [sg.Text("Select policy: "), sg.Listbox( values=policyMenu, enable_events=True, size=(20, 5), key="-POLICIES-")],
    [sg.Button("Simulate"), sg.Text("", key="-SIM-MSG-")],    
]
statusColumn = [
    [sg.Text("Simulation status", font='Lucida', text_color = 'Yellow')],
    [sg.Text('_'*50)],
    [sg.Text("Simulation progress:"), sg.Text("",key="-START-TIME-"), sg.Text("",key="-END-TIME-")],
    [sg.Text('_'*50)],
    [sg.Text("Visualize")],
    [sg.Text('_'*50)],
]
layout = [ [sg.Column(dashboardColumn), sg.VSeperator(), sg.Column(statusColumn) ] ]
window = sg.Window("FOMO Digital Twin Dashboard 0.1", layout)

def userError(string):
    window["-FEEDBACK-"].update(string, text_color = "dark orange")
    beepy.beep(sound="error")
def userFeedback_OK(string):
    window["-FEEDBACK-"].update(string, text_color = "LightGreen")
def userFeedbackClear():
    window["-FEEDBACK-"].update("")

DURATION = 960 # change to input-field with default value

def startSimulation(simPolicy, state):
    if simPolicy == "Do nothing": 
        simulator = sim.Simulator( 
            DURATION,
            policies.DoNothing(),
            copy.deepcopy(state),
            verbose=True,
            label="DoNothing",
        )
        metrics = simulator.metrics.get_all_metrics()
    elif simPolicy == "Rebalancing":
        simulator = sim.Simulator( 
            DURATION,
            policies.RebalancingPolicy(),
            copy.deepcopy(state),
            verbose=True,
            label="Rebalancing",
        )
    elif simPolicy == "Fosen & Haldorsen":
        simulator = sim.Simulator(
            DURATION,
            policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False),
            copy.deepcopy(state),
            verbose=True,
            label="FosenHaldorsen",
        )
    elif simPolicy == "F & H Greedy":
        simulator = sim.Simulator(
            DURATION,
            policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True),
            copy.deepcopy(state),
            verbose=True,
            label="Greedy",
        )
    window["-START-TIME-"].update("Start: " + readTime())
    simulator.run()
    window["-END-TIME-"].update("End:" + readTime())
    metrics = simulator.metrics.get_all_metrics()
    beepy.beep(sound="ready")

def GUI_main():
    simPolicy = "" 
    state = sim.State()
    task = [] # TODO, only one task allowed in queue at the moment
    readyForTask = False # used together with timeout to ensure one iteration in loop for 
                         # updating field -SIM-MSG- or -STATE-MSG- before starting long operation
    while True:
        GUI_event, GUI_values= window.read(timeout = 200) # waits 200 millisec before looking in taskQueue
        # print("*", end="")
        if len(task) > 0:
            if not readyForTask:
                readyForTask = True
            elif task[0] == "Sim":
                startSimulation(simPolicy, state) # TODO, change to task[1] and task[2]
                state = sim.State() # to ensure state is set before new simulation
                window["-SIM-MSG-"].update("")
                task = []
                readyForTask = False
            elif task[0] == "Init state": # TODO try to make general for several cities
                state = get_initial_state(task[1], week = task[2])
                new_ideal_state = ideal_state.evenly_distributed_ideal_state(state)
                state.set_ideal_state(new_ideal_state)
                window["-STATE-CITY-"].update(task[1])
                window["-STATE-MSG-"].update("")
                userFeedback_OK("Initial state set OK")
                beepy.beep(sound="ping")
                task = []
                readyForTask = False


        # window["-FEEDBACK-"].update(" ") # clear user feedback field # TODO, must be handled differently after timeout in main GUI loop
        if GUI_event == "All Oslo":
            window["-INPUTfrom-"].update("From: 1")
            window["-INPUTto-"].update("To: 35")  # TODO, Magic number, move to local settings ?? 
        elif GUI_event == "Clear":
            window["-INPUTfrom-"].update("From: ")
            window["-INPUTto-"].update("To: ")   
        elif GUI_event == "Download Oslo":
            oslo(GUI_values["-INPUTfrom-"], GUI_values["-INPUTto-"])
        elif GUI_event == "Find stations and distances":
            if GUI_values["-OSLO-"]:
                window["-FEEDBACK-"].update("City OK", text_color = "")
                calcDistances("Oslo")
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented") 
            elif GUI_values["-UTOPIA-"]:
                window["-FEEDBACK-"].update("City OK", text_color = "LightGreen") 
                calcDistances("Utopia")
            else:
                print("*** Error: wrong value from Radiobutton")         
        elif GUI_event == "Set initial state":
            if GUI_values["-OSLO-"]:
                if GUI_values["-WEEK-"] == "Week no: ":
                    weekNo = 53
                    window["-WEEK-"].update("Week no: 53") 
                else:
                    weekNo = int(strip("Week no: ", GUI_values["-WEEK-"]))
                task = ["Init state", "Oslo", weekNo]    
                window["-STATE-MSG-"].update("Lengthy operation started ...", text_color="cyan")
            elif GUI_values["-UTOPIA-"]: # This is (still) quick
                window["-WEEK-"].update("Week no: 48") # Only week with traffic at the moment for Utopia
                task = ["Init state", "Utopia", 48]    
                window["-STATE-MSG-"].update("short operation started ...", text_color="cyan")
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented")
            else:
                userError("You must select a city")        
        elif GUI_event == "Simulate":
            if len(state.stations) == 0:
                userError("You must set an initial state")
            elif simPolicy == "":
                userError("You must select a policy")
            else:
                task =["Sim", simPolicy, state]
                # startSimulation(simPolicy, state)
                # state = sim.State() # to ensure state is set before new simulation
                window["-STATE-CITY-"].update("")
                window["-WEEK-"].update("Week no: ")
                window["-SIM-MSG-"].update("Lengthy operation started ...", text_color="cyan")
        if GUI_values["-POLICIES-"] != []:
            simPolicy = GUI_values["-POLICIES-"][0]
        if GUI_event == "Exit" or GUI_event == sg.WIN_CLOSED:
            break

                
    window.close()