# script.py
"""
Commands called from GUI dashboard and script handling, operates via 
GUI dashboard functions
"""
import copy
import os
from datetime import datetime
import beepy
import PySimpleGUI as sg

import sim
import clustering.scripts
import ideal_state.haflan_haga_spetalen
import policies
import policies.fosen_haldorsen

# import GUI.dashboard

from tripStats.download import oslo
from tripStats.helpers import write, dateAndTimeStr, readTime
from tripStats.parse import calcDistances, get_initial_state

from GUI import loggFile, scriptFile, trafficLogg
# from GUI.dashboard import  updateFieldOK, userError, userFeedback_OK


class Session:
    def __init__(self, sessionName):
        self.name = sessionName
        self.startTime = ""
        self.endTime = ""
        self.state = sim.State()  # TODO see if these three can be replaced by = []
        self.initState = sim.State()
        self.initStateType = ""  # is "", "FH" or "HHS"
        self.idealState = sim.State()
        self.idealStateType = ""  # is "", "FH" or "HHS" # TODO, redundant, since ideal state must be of same kind as init ?
        self.simPolicy = ""

    def saveState(self, filename):
        print("SaveState called for " + self.name + "at" + self.startTime)


# DURATION = 43200 # 43200 = 24 * 60 * 30, ca en måned. TODO change to input-field with default value
# DURATION = 960 # 16 hours

DURATION = 180  # 3 hours  # TODO move this to GUI and log in session.log

policyMenu = ["Do-nothing", "Rebalancing", "Fosen&Haldorsen", "F&H-Greedy"] # must be single words


def doCommand(session, task):
    from GUI.dashboard import updateField, updateFieldOK, userFeedback_OK, updateFieldDone, userError

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

    elif task[0] == "Init-state-FH" or task[0] == "Init-state-HHS" or task[0] == "Init-manual":
        if task[0] == "Init-state-FH":
            write(scriptFile, ["Init-state-FH", task[1], task[2]])
            session.initState, loggText = get_initial_state(task[1], week = int(task[2]))
            # TODO retur  av loggText var fordi jeg ikke fikk til å bruke dashboard.loggFile fra denne fila
            session.initStateType = "FH"
        elif task[0] == "Init-state-HHS": 
            write(scriptFile, ["Init-state-HHS"])
            session.initState = clustering.scripts.get_initial_state(
                "test_data",
                "0900-entur-snapshot.csv",
                "Scooter",
                number_of_scooters = 2500,
                number_of_clusters = 10,
                number_of_vans = 2,
                random_seed = 1,
            )
            loggText = [] # not used in this case
            session.initStateType = "HHS"
        elif task[0] == "Init-manual":
            loggText = [] # not used in this case
            manualInitState(session, task[1])
            session.initStateType = "manual"

        updateField("-STATE-MSG-", session.initStateType + " ==> OK")
        userFeedback_OK("Initial state set OK")
        session.idealState = sim.State() # ideaLstate must be cleared, if it exist or not
        session.idealStateType = ""
        write(loggFile, [task[0], "finished:", dateAndTimeStr()] + loggText)
        beepy.beep(sound="ping")


    elif task[0] == "Ideal-state":
        if session.initStateType == "": # an initial state does NOT exist
            userError("Cannot calculate ideal state without an initial state")
        else:
            if session.initStateType == "HHS": # OK to make ideal-HHS from init-HS
                state = session.initState # via local variable to be sure initState is not destroyed (?)
                ideal_state.haflan_haga_spetalen_ideal_state(state)
                session.idealState = state
            if session.initStateType =="FH":
                state = session.initState
                newIdeal_state = ideal_state.evenly_distributed_ideal_state(state)
                session.idealState = newIdeal_state # TODO, probably clumsy, had to (try again?) go via newIdeal_state variable due to import-trouble !???
            write(scriptFile, ["Ideal-state"])
            write(loggFile, [task[0], "finished:", dateAndTimeStr()]) 
            updateFieldDone("-CALC-MSG-", session.initStateType + " ==> OK") 
            userFeedback_OK("Ideal state calculated OK")
            beepy.beep(sound="ping")

    elif task[0] == "Sim":
        fromState = ""  
        if not session.idealStateType == "": 
            session.state = session.idealState 
            fromState = "IDEAL"
        elif not session.initStateType == "":
            session.state = session.initState 
            fromState = "INIT"
        else:
            userError("You must set an initial (or ideal) state")
        if not fromState == "":
            policy = task[1]
            write(scriptFile, ["Sim", policy] )
            # TODO Debug-plan: tap out used state to check that several simulations in a row start from same state"])    
            startSimulation(policy, session.state)
            write(loggFile, ["Sim", policy, "finished:", dateAndTimeStr()]) 
        updateField("-SIM-MSG-", "")

def dumpMetrics(metric):
    # print("dumpMetrics called")
    metricsCopy = metric
    pass

def startSimulation(simPolicy, state):
    from GUI.dashboard import updateField
    simulator = sim.Simulator(0,  policies.DoNothing(), sim.State()) # TODO (needed?), make empty Simulator for scope
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
    simulationDescr =  ["Simulation-start:", simPolicy, dateAndTimeStr()] 
    write(loggFile, simulationDescr)
    write(trafficLogg, simulationDescr)
    updateField("-START-TIME-", "Start: " + readTime())
    start = datetime.now()
    simulator.run()
    end = datetime.now()
    duration = end - start
    seconds = str(duration.total_seconds())
    updateField("-END-TIME-", "End:" + readTime())
    write(loggFile, ["Simulation-end:", dateAndTimeStr(), "usedTime(s):", seconds ])
    write(trafficLogg, ["usedTime(s):", seconds ])
    metrics = simulator.metrics.get_all_metrics()
    dumpMetrics(metrics)
    beepy.beep(sound="ready")

def smallCircle(session):
    arrive_intensities = [] # 3D matrise som indekseres [station][day][hour]
    leave_intensities = []  # 3D matrise som indekseres [station][day][hour]
    move_probabilities = [] # 4D matrise som indekseres [from-station][day][hour][to-station]
    for station in range(4): # eksempelet har 4 stasjoner
        arrive_intensities.append([])
        leave_intensities.append([])
        move_probabilities.append([])
        for day in range(7):
            arrive_intensities[station].append([])
            leave_intensities[station].append([])
            move_probabilities[station].append([])
            for hour in range(24):
                arrive_intensities[station][day].append(1) # fra denne stasjonen på gitt tidspunkt drar det 1 sykkel i timen 
                leave_intensities[station][day].append(1)  # fra denne stasjonen på gitt tidspunkt kommer det 1 sykkel i timen
                move_probabilities[station][day].append([1/3, 1/3, 1/3, 1/3]) # sannsynlighetsfordeling for å dra til de forskjellige stasjonene
                move_probabilities[station][day][hour][station] = 0 # null i sannsynlighet for å bli på samme plass

    state = sim.State.get_initial_state(
                bike_class = "Scooter",
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
#                number_of_scooters = [2, 2, 2, 2],
                number_of_scooters = [1, 1, 1, 1],
                number_of_vans = 2,
                random_seed = 1,
                arrive_intensities = arrive_intensities,
                leave_intensities = leave_intensities,
                move_probabilities = move_probabilities,
            )
    session.initState = state
    pass        

def manualInitState(session, testName):
    if testName == "Small-Circle":
        smallCircle(session)

def bigOsloTest():
    session = Session("bigOsloTest")
    for week in range(6):
        command = ["Init-state-FH", "Oslo", str(week+1)]
        doCommand(session, command)
        command = ["Ideal-state"]
        doCommand(session, command)
        command = ["Sim", "Rebalancing"]
        doCommand(session, command)

def doScript(session, fileName):
    session.name += "-FROM:"
    session.name += fileName
    script = open(fileName, "r")
    lines = script.readlines()
    command = []
    for i in range(len(lines)):
        for word in lines[i].split():
            command.append(word)
        doCommand(session, command) 
        command = []

def replayScript(): 
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
