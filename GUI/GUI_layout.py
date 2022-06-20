
import PySimpleGUI as sg

colWidth = 55
policyMenu = ["Do-nothing", "Random", "HHS-Greedy", "Fosen&Haldorsen", "F&H-Greedy"] # must be single words
simOptions = [
    [sg.Checkbox('Logg traffic', key='-LOGG-TRAFFIC-')], 
    # [sg.Checkbox('Option-1 (na)', key='-SIM-OPT-1-')],
    # [sg.Checkbox('Option-2 (na)', key='-SIM-OPT-2-')],
]
dashboardColumn = [
    [sg.Text("Prep. and set up ", font='Lucida', text_color = 'Yellow'), sg.VSeparator(), 
        sg.Button("Fast-Track", button_color = "forest green"), sg.Button("main.py", button_color="snow4"), sg.Button("Exit")],
    [sg.Text("Set up simulation", font="Helvetica 14", size=(30, 1), text_color = "spring green", key="-FEEDBACK-")],
    [sg.Text('_'*colWidth)],

    [sg.Text("Download Oslo trips, 1 = April 2019 ... 35 = February 2022)")],
    [sg.Button("All Oslo"), sg.Button("Clear"), sg.Input("From: ", key="-INPUTfrom-", size=8), 
        sg.Input("To: ", key="-INPUTto-", size = 6), sg.Button("Download Oslo")],
    [sg.Text('_'*colWidth)],

    [sg.Text("Set initial state"), sg.Text("", key = "-STATE-MSG-")],
    [sg.Text("Select city (Oslo is default, only relevant for CityBike)")],
    [sg.Radio("Oslo", "RADIO1", key = "-OSLO-"), sg.Radio("Bergen", "RADIO1", key = "-BERGEN-"), 
        sg.Radio("Utopia", "RADIO1", key = "-UTOPIA-"), sg.Button("Stations and distances")],
    [sg.Button("CityBike"), sg.Input("Week no: ", key="-WEEK-", size=12), sg.VSeparator(), 
        sg.Button("Entur"), sg.Button("US-init"), sg.Button("Test-state")],   
    [sg.Text('_'*colWidth)],

    [sg.Text("Calculate target state"), sg.Text("", key="-TARGET-METHOD-"), sg.Button("Evenly"), 
        sg.Button("Outflow"), sg.Button("US")], 
    [sg.Text("", key="-CALC-MSG-")], 
    [sg.Text('_'*colWidth)],
    [sg.Button("Save state"), sg.Input("Name: ", key ="-INPUTname-", size = 25), sg.Button("Load state")],  

]
statusColumn = [
    [sg.Text("Simulation", font='Lucida', text_color = 'Yellow'), sg.Text("", key="-SIM-MSG-")],
    [sg.Text("Simulation parameters")],
    [sg.Input("Start-day: 1", key="-START-D-", size = 11), sg.Input("Start-hour: 12", key="-START-H-", size = 12), 
        sg.Input("#days: 0", key="-NUM-DAYS-", size = 9), sg.Input("#hours: 2", key="-NUM-HOURS-", size = 10)],
    [sg.Text("Select policy: "), sg.Listbox( values=policyMenu, enable_events=True, size=(17, 5), key="-POLICIES-"),
        sg.VSeperator(), sg.Column(simOptions)],
    [sg.Button("Simulate")],
    [sg.Text("Simulation progress:"), sg.Text("",key="-START-TIME-"), sg.Text("",key="-END-TIME-")],
    [sg.Button("Timestamp and save session results"), sg.Button("Save session as script")],
    [sg.Button("Replay script")],
    [sg.Text('_'*50)],
    [sg.Text("Analysis of results", font='Lucida', text_color = 'Yellow')],
    [sg.Text("Choose folder for FOMO results files")],
    [sg.Input(size=(42, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse()],
    [sg.Listbox( values=[], enable_events=True, size=(30, 6), key="-RESULT-FILES-")],
    [sg.Text("The following checkboxes are still not implemented")],
    [sg.Checkbox ('Option 1', key='check_value1'), sg.Checkbox ('Option 2',key='check_value2'), sg.Checkbox ('Grid',key='check_value2')],
    [sg.Button("Test-1"), sg.Button("Test-2"), sg.Button("Test-3"), sg.Button("Trips/week-Oslo")],
    [sg.Text('_'*50)],
]
layout = [ [sg.Column(dashboardColumn), sg.VSeperator(), sg.Column(statusColumn) ] ]
window = sg.Window("FOMO Digital Twin Dashboard 0.2", layout)

