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
from init_state.cityBike.parse import calcDistances, get_initial_state

from GUI import loggFile, scriptFile
from GUI.GUIhelpers import *
 
class Session:
    def __init__(self, sessionName):
        self.name = sessionName
        self.startTime = ""
        self.endTime = ""
        self.initState = sim.State()
        self.initStateType = ""  # is "", "cityBike", "Entur", "manual or test?? ", "loaded"
        self.targetStateMethod = ""  # is "", "evenly", "outflow" or "US"
        self.simPolicy = ""

    def saveState(self, filename):
        print("SaveState called for " + self.name + "at" + self.startTime) #### TODO not difference simulated time and measured simulator exec time

def doCommand(session, task):
    if len(task) == 0:
        return
    if task[0] == "Find-stations":
        write(scriptFile, ["Find-stations", task[1]])
        calcDistances(task[1])
        updateFieldOK("-FEEDBACK-", "City OK")
        beepy.beep(sound="ping")
        write(loggFile, [task[0], "finished:", dateAndTimeStr(), "city:", task[1]])
    elif task[0] == "Download-Oslo":
        write(scriptFile, ["Download-Oslo", task[1], task[2]])
        oslo(task[1], task[2])
        write(loggFile, [task[0], "finished:", dateAndTimeStr(), "fromWeek:", task[1], "toWeek:", task[2]])
        userFeedback_OK("Tripdata downloaded")
        beepy.beep(sound="ping")
   
    #### INIT STATE handling
    elif task[0] == "Init-state-CityBike" or task[0] == "Init-state-Entur" or task[0] == "Init-state-US" or task[0] == "Init-state-test":
        if task[0] == "Init-state-CityBike":
            write(scriptFile, ["Init-state-CityBike", task[1], task[2]])
            session.initState = get_initial_state(task[1], week = int(task[2]), bike_class="Bike", number_of_vans=3, random_seed=1) # TODO, hardwired, not good, fix 
            session.initStateType = "CityBike"
        elif task[0] == "Init-state-Entur": 
            write(scriptFile, ["Init-state-Entur"])
            session.initState = init_state.entur.scripts.get_initial_state(
                "test_data",
                "0900-entur-snapshot.csv",
                "Scooter",
                number_of_scooters = 150, # TODO, check with AD,cannot use more default params
                number_of_clusters = 5, 
                number_of_vans = 3,
                random_seed = 1,
            )
            session.initStateType = "Entur"
        elif task[0] == "Init-state-US":
            write(scriptFile, ["Init-state-US"])
            session.initState =  init_state.fosen_haldorsen.get_initial_state(init_hour=7, number_of_stations=50, number_of_vans=3, random_seed=1)        
            session.initStateType = "US"
        elif task[0] == "Init-state-test":
            write(scriptFile, ["Init-state-test"])
            manualInitState(session, task[1]) # second param opens for several different
            session.initStateType = "test"
        updateFieldDone("-STATE-MSG-", session.initStateType + " ==> OK")
        userFeedback_OK("Initial state set OK")
        session.targetState = sim.State() # targetState must be cleared, if it exist or not
        session.targetStateType = ""
        write(loggFile, [task[0], "finished:", dateAndTimeStr()])
        beepy.beep(sound="ping")
   
    #### TARGET state METHOD handling
    elif task[0] == "Target-state-evenly-distributed":
        if session.initStateType == "": # an initial state does NOT exist  
                userError("Can't calc target state without init state")
        else:
            userFeedbackClear()
            if session.initStateType == "Entur" or session.initStateType =="CityBike" or session.initStateType =="US" or session.initStateType == "test": # TODO 3x similar code, REFACTOR
                tState = target_state.evenly_distributed_target_state(session.initState)
                session.initState.set_target_state(tState) 
                session.targetStateType = "evenly"
            else:
                print("*** Error: initStateType invalid") 
            write(scriptFile, ["Target-state-method-evenly"])
            write(loggFile, [task[0], "finished:", dateAndTimeStr()]) 
            updateFieldDone("-CALC-MSG-", session.initStateType + " -> evenly -> OK") 
            userFeedback_OK("Target state evenly calculated OK")
            beepy.beep(sound="ping")

    elif task[0] == "Target-state-outflow": # TODO, refactor code in this and next elif
        if session.initStateType == "": # an initial state does NOT exist 
            userError("Can't calc target state without init state")
        else:
            userFeedbackClear()
            if session.initStateType == "Entur" or session.initStateType =="CityBike" or session.initStateType =="US" or session.initStateType == "test": # TODO 3x similar code, REFACTOR
                tState = target_state.outflow_target_state(session.initState)
                session.initState.set_target_state(tState) 
                session.targetStateMethod = "outflow"
            else:
                print("*** Error: initStateType invalid") 
            write(scriptFile, ["Target-state-method-outflow"])
            write(loggFile, [task[0], "finished:", dateAndTimeStr()]) 
            updateFieldDone("-CALC-MSG-", session.initStateType + " -> outflow -> OK") 
            userFeedback_OK("Target state outflow calculated OK")
            beepy.beep(sound="ping")

    elif task[0] == "Target-state-US":
        if session.initStateType == "": # an initial state does NOT exist  
                userError("Can't calc target state without init state")
        else:
            if session.initStateType == "Entur" or session.initStateType =="CityBike" or session.initStateType =="US" or session.initStateType == "test": # TODO 3x similar code, REFACTOR
                if session.initStateType != "US":
                    print("*** Warning: you are calculating target state with US method on an initial state different from US-init !?")
                tState = target_state.equal_prob_target_state(session.initState)
                session.initState.set_target_state(tState) 
                session.targetStateType = "UrbanSharing"
            else:
                print("*** Error: initStateType invalid") 
            write(scriptFile, ["Target-state-method-UrbanSharing"])
            write(loggFile, [task[0], "finished:", dateAndTimeStr()]) 
            updateFieldDone("-CALC-MSG-", session.initStateType + " -> Urban -> OK") 
            userFeedback_OK("Target state US calculated OK")
            beepy.beep(sound="ping")

    elif task[0] == "Save-state": # saves the initial state 
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
        if task[1] == "":
            userError("Give a valid filename (see terminal)")
            print("*** Error: Give a valid filename for saved state (without the .json extension")
        else:
            loadFileName = task[1] + ".json"
            # print("before" + dateAndTimeStr()) # code for measuring time reduction when using ZIP
            loadStateFile = open(loadFileName, "r") # from JSON alternative
            string = loadStateFile.read()
            # with zipfile.ZipFile(loadFileName, mode = "r") as archive:
            #     string = archive.read(loadFileName) # NOTE, compression can be really efficient here
            session.initState = jsonpickle.decode(string)
            session.initStateType = "loaded"  
            # print("after" + dateAndTimeStr())

            write(scriptFile, ["Load-state"])   
            write(loggFile, ["Loaded-state", "from", loadFileName])
            userFeedback_OK("Initial state loaded OK")
            beepy.beep(sound="ping")

    elif task[0] == "Sim":
        session.startTime = dateAndTimeStr()
        policy = task[1]
        simDuration=int(task[3])
        startTime=int(task[2])
        write(scriptFile, ["Sim", policy, str(startTime), str(simDuration)])
        if session.initStateType != "":
            simulationDescr =  ["Simulation-start:", dateAndTimeStr(), "startState:", session.initStateType, "simPolicy:", policy, 
                "simDuration:", str(simDuration), "startTime:", str(startTime) ] 
            write(loggFile, simulationDescr)
            write(trafficLogg, simulationDescr)
            # state = session.initState # TODO Ask AD, see NOTE 14 Juni testing
            state = copy.deepcopy(session.initState) # NOTE 20 Juni
            startSimulation(session.startTime, policy, state, startTime, simDuration) # # NOTE 20 Juni
            # startSimulation(session.startTime, policy, session.initState, startTime, simDuration)
            write(loggFile, ["Sim", policy, "finished:", dateAndTimeStr()])
            updateField("-SIM-MSG-", "")
        else:
            userError("Set an initial state")

def dumpMetrics(metric):
    # print("dumpMetrics called")
    metricsCopy = metric
    pass

def startSimulation(timeStamp, simPolicy, state, startTime, simDuration):
    # from GUI.dashboard import updateField
    # simulator = sim.Simulator(0,  policies.DoNothing(), sim.State()) # TODO (needed?), make empty Simulator for scope
    if simPolicy == "Do-nothing":
        policy = policies.DoNothing()
    elif simPolicy == "Random":
        policy = policies.RandomActionPolicy()
    elif simPolicy == "HHS-Greedy":
        policy = policies.GreedyPolicy()
    elif simPolicy == "Fosen&Haldorsen":
        policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=False, scenarios=2, branching=7, time_horizon=25)
    elif simPolicy == "F&H-Greedy":
        policy = policies.fosen_haldorsen.FosenHaldorsenPolicy(greedy=True)
        
    simulator = sim.Simulator( 
        initial_state = state,
        policy = policy,
        start_time = startTime,
        duration = simDuration,
        verbose=True,
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
                traveltime_matrix = [ # in minutes
                    [10, 10, 10, 10],
                    [10, 10, 10, 10],
                    [10, 10, 10, 10],
                    [10, 10, 10, 10]
                ],
                traveltime_van_matrix = [ # minutes
                    [5, 5, 5, 5],
                    [5, 5, 5, 5],
                    [5, 5, 5, 5],
                    [5, 5, 5, 5],
                ],
                main_depot = None,
                secondary_depots = 0,
                number_of_scooters = [1, 1, 1, 1],
                capacities = [4, 4, 4, 4],
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
        [sg.Button("Load script names"), sg.Button("Confirm")],
        [sg.Listbox( values=[], enable_events=True, size=(40, 10), key="-FILE LIST-")]
    ]
    window = sg.Window("FOMO - select script", layout, modal=True)
    file_list = os.listdir("GUI/scripts")

    while True:
        event, values = window.read()
        if event == "Load script names":
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
