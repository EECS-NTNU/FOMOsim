# dashboard.py

import copy
import policies
import policies.fosen_haldorsen
# import policies.haflan_haga_spetalen # TODO ??? gives error
import sim
import clustering.scripts

import ideal_state.evenly_distributed_ideal_state
import ideal_state.haflan_haga_spetalen
 
from tripStats.download import *
from tripStats.parse import calcDistances, get_initial_state
from tripStats.helpers import printTime, readTime, write
import PySimpleGUI as sg
import beepy

policyMenu = ["Do-nothing", "Rebalancing", "Fosen&Haldorsen", "F&H-Greedy"] # must be single words
scriptFile = open("tripStats/scripts/sessionLog.txt", "w")
resultsFile = open("tripStats/results/results.txt", "w")

################################ GUI LAYOUT CODE
colWidth = 55
dashboardColumn = [
    [sg.Text("Prep. and set up ", font='Lucida', text_color = 'Yellow'), sg.VSeparator(), 
        sg.Button("Fast-Track", button_color = "forest green"), sg.Button("Asbjørn", button_color="snow4"), sg.Button("Exit")],
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
    [sg.Text("Set initial state"), sg.Button("main.py-small-test-case")],
    [sg.Button("Fosen & Haldorsen"), sg.Input("Week no: ", key="-WEEK-", size=12), sg.VSeparator(), 
        sg.Button("Haflan, Haga & Spetalen")],
    [sg.Text("", key = "-STATE-MSG-")],
    [sg.Text('_'*colWidth)],
    [sg.Text("Method for calculating ideal state: "), sg.Text("...", key="-IDEAL-METHOD-")],
    [sg.Button("Calculate"), sg.Text("", key="-CALC-MSG-")], 
    [sg.Text('_'*colWidth)],
    [sg.Text("Select policy: "), sg.Listbox( values=policyMenu, enable_events=True, size=(17, 4), key="-POLICIES-"), sg.Text("Hours: 16", size = 11, key="-HOURS-")],
    [sg.Button("Simulate"), sg.Button("Simulate all"), sg.Button("Replay script")],
    [sg.Text("", key="-SIM-MSG-")],    
]
statusColumn = [
    [sg.Text("Simulation status", font='Lucida', text_color = 'Yellow')],
    [sg.Text('_'*50)],
    [sg.Text("Simulation progress:"), sg.Text("",key="-START-TIME-"), sg.Text("",key="-END-TIME-")],
    [sg.Button("Timestamp and save session results"), sg.Button("Save session as script")],
    [sg.Text('_'*50)],
    [sg.Text("Visualize")],
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

DURATION = 960 # change to input-field with default value

def runAllPolicies(state, duration): # TODO bare alle som kan starte i samme state på samme data !!!???
    # må ta kopi av state, starte på samme gir riktigere sammenlikning og raskere oppstart 
    # - rapporter tid brukt i simlator, i GUI!, og på fil
    # - flere simuleringer på en fil, dvs. for en sesjon
    pass

def startSimulation(simPolicy, state):
    simulator = sim.Simulator(0,  policies.DoNothing(), sim.State()) # make empty Simulator for scope
    if simPolicy == "Do-nothing": 
        simulator = sim.Simulator( 
            DURATION,
            policies.DoNothing(),
            copy.deepcopy(state),
            verbose=True,
            label="DoNothing",
        )
    elif simPolicy == "Rebalancing":
        simulator = sim.Simulator( 
            DURATION,
            policies.RebalancingPolicy(),
            copy.deepcopy(state),
            verbose=True,
            label="Rebalancing",
        )
    elif simPolicy == "Fosen&Haldorsen":
        simulator = sim.Simulator(
            DURATION,
            policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False),
            copy.deepcopy(state),
            verbose=True,
            label="FosenHaldorsen",
        )
    elif simPolicy == "F&H-Greedy":
        simulator = sim.Simulator(
            DURATION,
            policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True),
            copy.deepcopy(state),
            verbose=True,
            label="Greedy",
        )
    print("Sim time: ")
    printTime()
    window["-START-TIME-"].update("Start: " + readTime())
    simulator.run()
    window["-END-TIME-"].update("End:" + readTime())
    printTime()
    metrics = simulator.metrics.get_all_metrics()
    beepy.beep(sound="ready")

def doCommand(task, savedInitialState, savedIdealState):
    if task[0] == "Init-state-FH": # TODO try to make general for several cities and methods
        write(scriptFile, ["Init-state-FH", task[1], str(task[2])])
        print("FH init-state: ") # TODO change these to session-log
        printTime()
        state = get_initial_state(task[1], week = task[2])
        window["-STATE-MSG-"].update("FH ==> OK")
        userFeedback_OK("Initial state set OK")
        savedInitialState = state
        if len(savedIdealState.stations) > 0 : # if it exists, it must be cleared
            savedIdealState = sim.State()
        print(" ", end="")
        printTime()
    elif task[0] == "Init-state-HHS": # TODO try to make general for several cities and methods
        write(scriptFile, ["Init-state-HHS"])
        state = clustering.scripts.get_initial_state(
            "test_data",
            "0900-entur-snapshot.csv",
            "Scooter",
            number_of_scooters = 2500,
            number_of_clusters = 10,
            number_of_vans = 2,
            random_seed = 1,
        )
        window["-STATE-MSG-"].update("HHS ==> OK")
        userFeedback_OK("Initial state set OK")
        savedInitialState = state
        if len(savedIdealState.stations) > 0 : # it exists, and must be cleared
            savedIdealState = sim.State()
        print("initial state saved")
    elif task[0] == "Ideal-state":
        if task[2] == "HHS":
            write(scriptFile, ["Ideal-state", "HHS"])
            print("HHS ideal-state:")
            printTime()
            ideal_state.haflan_haga_spetalen_ideal_state(task[1])
            savedIdealState = task[1]
            printTime()
        elif task[2] == "FH":
            write(scriptFile, ["Ideal-state", "FH"])
            newIdeal_state = ideal_state.evenly_distributed_ideal_state(task[1])
            task[1].set_ideal_state(newIdeal_state) # TODO, had to (try again?) go via newIdeal_state variable due to import-trouble !???
            savedIdealState = task[1]
        else:
            print("*** Error in task: Ideal")
        window["-CALC-MSG-"].update(task[2] + " ==> OK", text_color="cyan")
        userFeedback_OK("Ideal state calculated OK")

    elif task[0] == "Sim":
        fromState = ""
        if len(savedIdealState.stations) > 0 :
            state = savedIdealState
            fromState = "IDEAL"
            print("simulates from IDEAL state")
        elif len(savedInitialState.stations) > 0:
            fromState = "INIT"
            state = savedInitialState
            print("simulates from INIT state")
        else:
            print("*** ERROR: state not available")
        write(scriptFile, ["Sim", fromState, task[1]] )    
        startSimulation(task[2], state) # TODO, change to task[1] and task[2]
        state = sim.State() # to ensure state is reset before new simulation
        window["-SIM-MSG-"].update("")

    elif task[0] == "Sim-all":
        scriptFile.write("Sim" + " " + " not ready ---")    
        printTime()
        stateCopy = task[1]
        for pol in range(len(policyMenu)):
            state = stateCopy
            print("Start simulation with policy: ", end="")
            print(policyMenu[pol])
            startSimulation(policyMenu[pol], state)   
        printTime()
    return savedInitialState, savedIdealState

def replayScript(fileName):
    state = sim.State() # all three set initially empty
    savedInitialState = sim.State()
    savedIdealState = sim.State()
    script = open(fileName, "r")
    lines = script.readlines()
    command = []
    for i in range(len(lines)):
        for word in lines[i].split():
             command.append(word)
        savedInitialState, savedIdealState = doCommand(command, savedInitialState, savedIdealState)
        command = []
  
def GUI_main():
    simPolicy = "" 
    state = sim.State() # all three set initially empty
    savedInitialState = sim.State()
    savedIdealState = sim.State()

    task = [] # TODO, only one task allowed in queue at the moment
    readyForTask = False # used together with timeout to ensure one iteration in loop for 
                         # updating field -SIM-MSG- or -STATE-MSG- before starting long operation
    idealStateMethod = ""
    while True:
        GUI_event, GUI_values= window.read(timeout = 200) # waits 200 millisecs before looking in taskQueue
        if len(task) > 0: ###### handling of lengthy operations is done in a two-stage process to be able to give message
            if not readyForTask:
                readyForTask = True
            else:
                savedInitialState, savedIdealState = doCommand(task, savedInitialState, savedIdealState)
                readyForTask = False
                task = []   
                beepy.beep(sound="ping")
        
        ###### DOWNLOAD GUI PART
        if GUI_event == "All Oslo":
            window["-INPUTfrom-"].update("From: 1")
            window["-INPUTto-"].update("To: 35")  # TODO, Magic number, move to local settings ?? 
        elif GUI_event == "Clear":
            window["-INPUTfrom-"].update("From: ")
            window["-INPUTto-"].update("To: ")   
        elif GUI_event == "Download Oslo":
            oslo(GUI_values["-INPUTfrom-"], GUI_values["-INPUTto-"])

        ###### SELECT CITY GUI PART     
        elif GUI_event == "Find stations and distances":
            if GUI_values["-OSLO-"]:
                calcDistances("Oslo") # TODO, should ideally be handled like time-consuming tasks
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented") 
            elif GUI_values["-UTOPIA-"]:
                calcDistances("Utopia")
            else:
                print("*** Error: wrong value from Radiobutton")         
            window["-FEEDBACK-"].update("City OK", text_color = "LightGreen") 
        
        ###### INIT STATE GUI PART
        elif GUI_event == "Fosen & Haldorsen":
            if GUI_values["-OSLO-"]:
                if GUI_values["-WEEK-"] == "Week no: ":
                    weekNo = 53
                    window["-WEEK-"].update("Week no: 53") 
                else:
                    if GUI_values["-WEEK-"] == "Week no: -na-":
                        window["-WEEK-"].update("Week no: ")
                        userError("You must select a week no")
                    else:    
                        weekNo = int(strip("Week no: ", GUI_values["-WEEK-"]))
                        task = ["Init-state-FH", "Oslo", weekNo]    
                        window["-STATE-MSG-"].update("Lengthy operation started ... (4 - 6 minutes)", text_color="cyan")
            elif GUI_values["-UTOPIA-"]: # This is (still) quick
                window["-WEEK-"].update("Week no: 48") # Only week with traffic at the moment for Utopia
                task = ["Init-state-FH", "Utopia", 48]    
                window["-STATE-MSG-"].update("short operation started ...", text_color="cyan")
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented")
            else:
                userError("You must select a city")
            idealStateMethod = "FH"
            window["-IDEAL-METHOD-"].update("Fosen & Haldorsen")
            window["-CALC-MSG-"].update("")

        if GUI_event == "Haflan, Haga & Spetalen": # handled here since it is relativelu quick
            task = ["Init-state-HHS"] 
            window["-WEEK-"].update("Week no: -na-")
            window["-IDEAL-METHOD-"].update("Haflan, Haga & Spetalen")
            window["-OSLO-"].update(True) # entur data are from Oslo
            window["-UTOPIA-"].update(False)
            window["-CALC-MSG-"].update("")
            window["-STATE-MSG-"].update("Lengthy operation started ... (see progress in terminal)", text_color="cyan")
            idealStateMethod = "HHS"

        ###### IDEAL STATE GUI PART   
        elif GUI_event == "Calculate":
            if len(savedInitialState.stations) > 0 :
                # task = ["Ideal-state", savedInitialState, idealStateMethod]
                task = ["Ideal-state", "FH"]
                if idealStateMethod == "HHS":
                    window["-CALC-MSG-"].update("Lengthy operation started ... (see progress in terminal)", text_color="cyan")
                    task = ["Ideal-state", "HHS"]
                else:
                    task = ["Ideal-state", "FH"]
            else:
                userError("You must set an initial state")
        ###### SIMULATE GUI PART
        elif GUI_event == "Simulate":
            if len(savedIdealState.stations) > 0 :
                state = savedIdealState # it exists
            elif len(savedInitialState.stations) > 0 :
                state = savedInitialState
            if len(state.stations) == 0:
                userError("You must set an initial (or ideal) state")
            elif simPolicy == "":
                userError("You must select a policy")
            else:
                task = ["Sim", simPolicy, state]
                window["-WEEK-"].update("Week no: ") # TODO, usikker på denne, henger igjen
                window["-SIM-MSG-"].update("Lengthy operation started ...  (see progress in terminal)", text_color="cyan")
        elif GUI_event == "Simulate all":
            # TODO use same error checking as above, refactor 
            task = ["Sim-all", state]

        elif GUI_event == "Replay script":
            replayScript("tripStats/scripts/script.txt")

        ###### TODO review under here
        if GUI_values["-POLICIES-"] != []:
            simPolicy = GUI_values["-POLICIES-"][0]
        if GUI_event == "Asbjørn":
            window.close()
            return False    
        if GUI_event == "Exit" or GUI_event == sg.WIN_CLOSED:
            return True
            # was break