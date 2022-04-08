# dashboard.py

import copy
import numpy as np
import os

import policies
import policies.fosen_haldorsen
import sim
import clustering.scripts

import ideal_state.evenly_distributed_ideal_state
import ideal_state.haflan_haga_spetalen

import GUI

from GUI import * 

import tripStats 

from tripStats.helpers import *

import PySimpleGUI as sg

import matplotlib.pyplot as plt
import beepy

class Session:
    def __init__(self, sessionName):
        self.name = sessionName
        self.startTime = ""
        self.endTime = ""
        self.state = sim.State() # TODO see if these three can be replaced by = []
        self.initState = sim.State()
        self.initStateType = "" # is "", "FH" or "HHS"
        self.idealState = sim.State()
        self.idealStateType = "" # is "", "FH" or "HHS" # TODO, redundant, since ideal state must be of same kind as init ?
        self.simPolicy = ""
    def saveState(self, filename):
        print("SaveState called for " + self.name + "at" + self.startTime)

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
    [sg.Button("Test state")],    
    [sg.Text('_'*colWidth)],
    [sg.Text("Method for calculating ideal state: "), sg.Text("...", key="-IDEAL-METHOD-")],
    [sg.Button("Calculate"), sg.Text("", key="-CALC-MSG-")], 
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
    [sg.Listbox( values=[], enable_events=True, size=(40, 10), key="-FILE LIST-")],
    [sg.Checkbox ('Option 1', key='check_value1'), sg.Checkbox ('Option 2',key='check_value2'), sg.Checkbox ('Grid',key='check_value2')],
    [sg.Button("Test-1"), sg.Button("Test-2"), sg.Button("Test-3"), sg.Button("Test-4")],
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

policyMenu = ["Do-nothing", "Rebalancing", "Fosen&Haldorsen", "F&H-Greedy"] # must be single words
DURATION = 180 # TODO change to input-field with default value

def dumpMetrics(metric):
    # print("dumpMetrics called")
    metricsCopy = metric
    pass

def startSimulation(simPolicy, state):
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
    write(loggFile, ["Simulation-start:", simPolicy, dateAndTimeStr()])
    window["-START-TIME-"].update("Start: " + readTime())
    start = datetime.now()
    simulator.run()
    end = datetime.now()
    duration = end - start
    window["-END-TIME-"].update("End:" + readTime())
    write(loggFile, ["Simulation-end:", dateAndTimeStr(), "usedTime(s):", str(duration.total_seconds())])
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
        pass


# def manualInitState(testName):
#     arrive_intensities = [] # 3D matrise som indekseres [station][day][hour]
#     leave_intensities = []  # 3D matrise som indekseres [station][day][hour]
#     move_probabilities = [] # 4D matrise som indekseres [from-station][day][hour][to-station]
#     for station in range(4): # eksempelet har 4 stasjoner
#         arrive_intensities.append([])
#         leave_intensities.append([])
#         move_probabilities.append([])
#         for day in range(7):
#             arrive_intensities[station].append([])
#             leave_intensities[station].append([])
#             move_probabilities[station].append([])
#             for hour in range(24):
#                 arrive_intensities[station][day].append(2) # fra denne stasjonen på gitt tidspunkt drar det 2 sykler i timer
#                 leave_intensities[station][day].append(2)  # fra denne stasjonen på gitt tidspunkt kommer det 2 sykler i timer
#                 move_probabilities[station][day].append([1/3, 1/3, 1/3, 1/3]) # sannsynlighetsfordeling for å dra til de forskjellige stasjonene
#                 move_probabilities[station][day][hour][station] = 0 # null i sannsynlighet for å bli på samme plass

#     print("Init-manual" + " " + testName)
#     state = sim.State.get_initial_state(
#                 bike_class = "Scooter",
#                 distance_matrix = [ # km
#                     [0, 4, 2, 3],
#                     [4, 0, 5, 1],
#                     [2, 5, 0, 4],
#                     [3, 1, 4, 0],
#                 ],
#                 speed_matrix = [ # km/h
#                     [15, 15, 15, 15],
#                     [15, 15, 15, 15],
#                     [15, 15, 15, 15],
#                     [15, 15, 15, 15],
#                 ],
#                 main_depot = None,
#                 secondary_depots = [],
#                 number_of_scooters = [2, 1, 2, 3],
#                 number_of_vans = 2,
#                 random_seed = 1,
#                 arrive_intensities = arrive_intensities,
#                 leave_intensities = leave_intensities,
#                 move_probabilities = move_probabilities,
#             )
#     return state

def doCommand(session, task):

    if task[0] == "Find-stations":
        write(scriptFile, ["Find-stations", task[1]])
        calcDistances(task[1])
        window["-FEEDBACK-"].update("City OK", text_color = "LightGreen") 
        if task[1] == "Oslo":
            beepy.beep(sound="ping")
        write(loggFile, [task[0], "finished:", dateAndTimeStr(), "city:", task[1]])

    elif task[0] == "Download-Oslo":
        write(scriptFile, ["Download-Oslo", task[1], task[2]])
        oslo(task[1], task[2])
        write(loggFile, [task[0], "finished:", dateAndTimeStr(), "fromWeek:", task[1], "toWeek:", task[2]])
        window["-FEEDBACK-"].update("Tripdata downloaded", text_color = "LightGreen")
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

        window["-STATE-MSG-"].update(session.initStateType + " ==> OK")
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
            window["-CALC-MSG-"].update(session.initStateType + " ==> OK", text_color="cyan")
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
        window["-SIM-MSG-"].update("")

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


def bigOsloTest():
    session = Session("bigOsloTest")
    for week in range(6):
        command = ["Init-state-FH", "Oslo", str(week+1)]
        doCommand(session, command)
        command = ["Ideal-state"]
        doCommand(session, command)
        command = ["Sim", "Rebalancing"]
        doCommand(session, command)

#########################################

def openVisual1():
    def draw_plot():
        plt.plot([0.1, 0.2, 0.5, 0.7])
        # plt.show(block=False)
        plt.show() 

    draw_plot()
  
def openVisual2():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 3])
    plt.show()
    
def openVisual3():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 3])
    plt.show()
def openVisual4():
    print("Visual Test 4, accumulated traffic during 53 weeks in Oslo")
    osloTripsFile = open("GUI/results/sessionLog-Oslo-Init-1-53.txt", "r")
    textLines = osloTripsFile.readlines()
    data = []
    for i in range(len(textLines)):
        line = []
        for word in textLines[i].split():
             line.append(word)
        data.append(int(line[6]))  
    plt.plot(data)
    plt.show()
    pass

######################################

def GUI_main():
    session = Session("GUI_main_session") 
    task = [] # TODO, only one task allowed in queue at the moment
    readyForTask = False # used together with timeout to ensure one iteration in loop for 
                         # updating field -SIM-MSG- or -STATE-MSG- before starting long operation
    while True:
        GUI_event, GUI_values= window.read(timeout = 200) # waits 200 millisecs before looking in taskQueue
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
            pass
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
                window["-FEEDBACK-"].update("Lengthy operation started ...", text_color="cyan")
            elif GUI_values["-BERGEN-"]:
                userError("Bergen not yet implemented") 
            elif GUI_values["-UTOPIA-"]:
                task = ["Find-stations", "Utopia"]
                window["-FEEDBACK-"].update("short operation started ...", text_color="cyan")
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
                    window["-STATE-MSG-"].update("Lengthy operation started ... (4 - 6 minutes)", text_color="cyan")
            elif GUI_values["-UTOPIA-"]: # This is (still) quick
                window["-WEEK-"].update("Week no: 48") # Only week with traffic at the moment for Utopia
                task = ["Init-state-FH", "Utopia", "48"]    
                window["-STATE-MSG-"].update("short operation started ...", text_color="cyan")
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
        elif GUI_event == "Test-4":
            openVisual4()

        if GUI_values["-POLICIES-"] != []:
            session.simPolicy = GUI_values["-POLICIES-"][0]