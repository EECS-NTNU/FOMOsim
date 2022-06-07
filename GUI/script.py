# script.py
"""
Commands called from GUI dashboard and script handling, operates via 
GUI dashboard functions
"""
import copy
import os
import zipfile
from datetime import datetime
import jsonpickle
import beepy
import PySimpleGUI as sg
from numpy import DataSource

import sim
import init_state.entur.scripts
import target_state.outflow_target_state
import policies
import policies.fosen_haldorsen

from init_state.cityBike.download import oslo
from init_state.cityBike.helpers import write, dateAndTimeStr, readTime, trafficLogg, saveTrafficLogg
from init_state.cityBike.parse import calcDistances, get_initial_state, get_initial_stateOLD

from GUI import loggFile, scriptFile
from GUI.GUIhelpers import *
 
class Session:
    def __init__(self, sessionName):
        self.name = sessionName
        self.startTime = ""
        self.endTime = ""
        self.state = sim.State()  # TODO see if these three can be replaced by = []
        self.initState = sim.State()
        self.initStateType = ""  # is "", "FH", "HHS", "manual", "loaded"
        self.targetState = sim.State()
        self.targetStateType = ""  # is "", "evenly" or "outflow"
        self.simPolicy = ""

    def saveState(self, filename):
        print("SaveState called for " + self.name + "at" + self.startTime) #### TODO not difference simulated time and measured simulator exec time

policyMenu = ["Do-nothing", "Rebalancing", "Fosen&Haldorsen", "F&H-Greedy"] # must be single words

def doCommand(session, task):
    if len(task) == 0:
        return

    if task[0] == "Find-stations":
        write(scriptFile, ["Find-stations", task[1]])
        calcDistances(task[1])
        updateFieldOK("-FEEDBACK-", "City OK")
        if task[1] == "Oslo":
            beepy.beep(sound="ping")
        write(loggFile, [task[0], "finished:", dateAndTimeStr(), "city:", task[1]])
    elif task[0] == "Download-Oslo":
        write(scriptFile, ["Download-Oslo", task[1], task[2]])
        oslo(task[1], task[2])
        write(loggFile, [task[0], "finished:", dateAndTimeStr(), "fromWeek:", task[1], "toWeek:", task[2]])
        userFeedback_OK("Tripdata downloaded")
        beepy.beep(sound="ping")
   
    #### INIT STATE handling
    elif task[0] == "Init-state-FH" or task[0] == "Init-state-HHS" or task[0] == "Init-test-state":
        if task[0] == "Init-state-FH":
            write(scriptFile, ["Init-state-FH", task[1], task[2]])
            session.initState = get_initial_state(task[1], week = int(task[2]), bike_class="Bike", number_of_vans=1, random_seed=1) # TODO, hardwired, not good, fix 
            session.initStateType = "FH"
        elif task[0] == "Init-state-HHS": 
            write(scriptFile, ["Init-state-HHS"])
            session.initState = init_state.entur.scripts.get_initial_state(
                "test_data",
                "0900-entur-snapshot.csv",
                "Scooter",
                number_of_scooters = 150,
                number_of_clusters = 5,
                number_of_vans = 1,
                random_seed = 1,
            )
            session.initStateType = "HHS"
        elif task[0] == "Init-test-state":
            manualInitState(session, task[1]) # second param opens for several different
            session.initStateType = "manual"
        updateField("-STATE-MSG-", session.initStateType + " ==> OK")
        userFeedback_OK("Initial state set OK")
        session.targetState = sim.State() # targetState must be cleared, if it exist or not
        session.targetStateType = ""
        write(loggFile, [task[0], "finished:", dateAndTimeStr()])
        beepy.beep(sound="ping")
   
    #### Target STATE handling
    elif task[0] == "Target-state-outflow": # TODO, refactor code in this and next elif
        if session.initStateType == "": # an initial state does NOT exist 
            userError("Can't calc target state without init state")
        else:
            if session.initStateType == "HHS" or session.initStateType =="FH" or session.initStateType == "manual": # TODO 3x similar code, REFACTOR
                state = session.initState # via local variable to ensure initState is not destroyed
                session.targetState = target_state.outflow_target_state(state)
                session.targetStateType = "outflow"
            else:
                print("*** Error: initStateType invalid") 
            write(scriptFile, ["Target-state-outflow"])
            write(loggFile, [task[0], "finished:", dateAndTimeStr()]) 
            updateFieldDone("-CALC-MSG-", session.initStateType + " -> outflow -> OK") 
            userFeedback_OK("Target state outflow calculated OK")
            beepy.beep(sound="ping")

    elif task[0] == "Target-state-evenly-distributed":        
        if session.initStateType == "": # an initial state does NOT exist  
                userError("Can't calc target state without init state")
        else:
            if session.initStateType == "HHS" or session.initStateType =="FH" or session.initStateType == "manual": 
                state = session.initState
                newTarget_state = target_state.evenly_distributed_target_state(state)
                session.targetState = newTarget_state # TODO, probably clumsy, had to (try again?) go via newTarget_state variable due to import-trouble !???
                session.targetStateType = "evenly"
            else:
                print("*** Error: initStateType invalid") 
            write(scriptFile, ["Target-state-evenly-distributed"])
            write(loggFile, [task[0], "finished:", dateAndTimeStr()]) 
            updateFieldDone("-CALC-MSG-", session.initStateType + " -> evenly -> OK") 
            userFeedback_OK("Target state evenly calculated OK")
            beepy.beep(sound="ping")

    elif task[0] == "Target-state-US":
        if session.initStateType == "": # an initial state does NOT exist  
                userError("Can't calc target state without init state")
        else:
            if session.initStateType == "HHS" or session.initStateType =="FH" or session.initStateType == "manual": 
                state = session.initState
                newTarget_state = target_state.us_target_state(state)
                session.targetState = newTarget_state # TODO, probably clumsy, had to (try again?) go via newTarget_state variable due to import-trouble !???
                session.targetStateType = "UrbanSharing"
            else:
                print("*** Error: initStateType invalid") 
            write(scriptFile, ["Target-state-UrbanSharing"])
            write(loggFile, [task[0], "finished:", dateAndTimeStr()]) 
            updateFieldDone("-CALC-MSG-", session.initStateType + " -> Urban -> OK") 
            userFeedback_OK("Target state US calculated OK")
            beepy.beep(sound="ping")

    elif task[0] == "Save-state": # saves target state if it exists, otherwise from init state
        savedStateFile = open(task[1] + ".json", "w")
        if session.targetStateType == "outflow" or session.targetStateType == "evenly":
            savedStateFile.write(jsonpickle.encode(session.targetState))  
            savedStateFile.close()
            write(loggFile, ["Save-state-from-target-state:", session.targetStateType])
        else: # saves init state
            savedStateFile.write(jsonpickle.encode(session.initState)) 
            savedStateFile.close()
            write(loggFile, ["Save-state-from-init-state:", session.initStateType])
        write(scriptFile, ["Save-state"])

    elif task[0] == "Load-state":
        print(" -- load state only from Oslo17.json (preliminary) --")
        print("before" + dateAndTimeStr())
        # loggText = [] # not used in this case
        loadStateFile = open("Oslo17out.json", "r") # from JSON alternative
        string = loadStateFile.read()
        # with zipfile.ZipFile("Oslo17out.zip", mode = "r") as archive:
        #     string = archive.read("Oslo17out.json")
        session.initState = jsonpickle.decode(string)
        session.initStateType = "loaded"  
        print("after" + dateAndTimeStr())

        write(scriptFile, ["Load-state"])   
        write(loggFile, ["Loaded-state"])
        beepy.beep(sound="ping")

    elif task[0] == "Sim":
        fromState = ""  
        if not session.targetStateType == "": 
            session.state = session.targetState 
            fromState = "IDEAL"
        elif not session.initStateType == "":
            session.state = session.initState 
            fromState = "INIT"
        else:
            userError("You must set an initial (or target) state")
        if not fromState == "":
            session.startTime = dateAndTimeStr()
            policy = task[1]
            simDuration=int(task[3])
            startTime=int(task[2])
            write(scriptFile, ["Sim", policy, str(startTime), str(simDuration)])
            # TODO Debug-plan: tap out used state to check that several simulations in a row start from same state"])
            if session.targetStateType == "":
                startState = session.initStateType
            else:
                startState = session.targetStateType

            simulationDescr =  ["Simulation-start:", dateAndTimeStr(), "startState:", startState, "simPolicy:", policy, 
                "simDuration:", str(simDuration), "startTime:", str(startTime) ] 
            write(loggFile, simulationDescr)
            write(trafficLogg, simulationDescr)
            startSimulation(session.startTime, policy, session.state, startTime, simDuration)
            write(loggFile, ["Sim", policy, "finished:", dateAndTimeStr()])
        updateField("-SIM-MSG-", "")

def dumpMetrics(metric):
    # print("dumpMetrics called")
    metricsCopy = metric
    pass

def startSimulation(timeStamp, simPolicy, state, startTime, simDuration):
    # from GUI.dashboard import updateField
    simulator = sim.Simulator(0,  policies.DoNothing(), sim.State()) # TODO (needed?), make empty Simulator for scope
    if simPolicy == "Do-nothing":
        policy = policies.DoNothing()
    elif simPolicy == "Random":
        policy = policies.RandomActionPolicy()
    elif simPolicy == "Rebalancing":
        policy = policies.RebalancingPolicy()
    elif simPolicy == "Fosen&Haldorsen":
        policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False)
    elif simPolicy == "F&H-Greedy":
        policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
        
    simulator = sim.Simulator( 
        duration = simDuration,
        policy = policies.DoNothing(),
        initial_state = state,
        verbose=True,
        start_time = startTime,
    )

    updateField("-START-TIME-", "Start: " + readTime())
    start = datetime.now()
    simulator.run()
    end = datetime.now()
    duration = end - start
    seconds = str(duration.total_seconds())
    updateField("-END-TIME-", "End:" + readTime())
    write(loggFile, ["Simulation-end:", dateAndTimeStr(), "usedTime(s):", seconds ])
    write(trafficLogg, ["usedTime(s):", seconds ])
    saveTrafficLogg(timeStamp) # save and reset traffic-file with timeStamp of start in filename
    trafficLogg.seek(0)
    trafficLogg.truncate()
    metrics = simulator.metrics.get_all_metrics()
    dumpMetrics(metrics)
    beepy.beep(sound="ready")

def allToAll4(session): # all to all topology with 4 stations
    arrive_intensities = [] # 3D matrix indexed as [station][day][hour]
    leave_intensities = []  # 3D matrix indexed as [station][day][hour]
    move_probabilities = [] # 4D matrix indexed as [from-station][day][hour][to-station]
    for station in range(4): 
        arrive_intensities.append([])
        leave_intensities.append([])
        move_probabilities.append([])
        for day in range(7):
            arrive_intensities[station].append([])
            leave_intensities[station].append([])
            move_probabilities[station].append([])
            for hour in range(24):
                arrive_intensities[station][day].append(1) # at this station it arrives 1 bike per hour
                leave_intensities[station][day].append(1)  # from this station it leaves 1 bike every hour 
                move_probabilities[station][day].append([1/3, 1/3, 1/3, 1/3]) # probabilities for moving to other stations. One 1/3 is set to 0 in next codeline 
                move_probabilities[station][day][hour][station] = 0 # zero probability for traveling from and to same station

    state = sim.State.get_initial_state(
                bike_class = "Scooter", # TODO logging code will crash if Bike is used TODO test again
                distance_matrix = [ # km
                    [0, 2, 2, 2],
                    [2, 0, 2, 2],
                    [2, 2, 0, 2],
                    [2, 2, 2, 0],
                ],
                speed_matrix = [ # km/h
                    [10, 10, 10, 10],
                    [10, 10, 10, 10],
                    [10, 10, 10, 10],
                    [10, 10, 10, 10]
                ],
                main_depot = None,
                secondary_depots = [],
                number_of_scooters = [1, 1, 1, 1],
                number_of_vans = 1,
                random_seed = 1,
                arrive_intensities = arrive_intensities,
                leave_intensities = leave_intensities,
                move_probabilities = move_probabilities,
            )
    session.initState = state

def manualInitState(session, testName):
    if testName == "allToAll4":
        allToAll4(session)

def bigOsloTest():
    session = Session("bigOsloTest")
    for week in range(6):
        command = ["Init-state-FH", "Oslo", str(week+1)]
        doCommand(session, command)
        command = ["Target-state"]
        doCommand(session, command)
        command = ["Sim", "Rebalancing"]
        doCommand(session, command)

def doScript(session, fileName):
    write(loggFile, ["Script-started:", fileName, dateAndTimeStr()]) 
    script = open(fileName, "r")
    lines = script.readlines()
    command = []
    for i in range(len(lines)):
        for word in lines[i].split():
            command.append(word)
        doCommand(session, command) 
        command = []

def replayScript(): # subwindow for selecting 
    session = Session("Replay")
    layout = [[sg.Text("Select script:", key="new")],
        [sg.Button("Load scripts"), sg.Button("Confirm")],
        [sg.Listbox( values=[], enable_events=True, size=(40, 10), key="-FILE LIST-")]
    ]
    window = sg.Window("FOMO - select script", layout, modal=True)
    file_list = os.listdir("GUI/scripts")

    while True:
        event, values = window.read()
        if event == "Load scripts":
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join("GUI/scripts", f))
                and f.lower().endswith((".txt"))
            ]
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            if len(values["-FILE LIST-"]) > 0:
                filepath = os.path.join("GUI/scripts", values["-FILE LIST-"][0])
        elif event == "Confirm":
            # print(filepath)
            doScript(session,filepath)
            break 
        elif event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()
