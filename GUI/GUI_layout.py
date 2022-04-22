
import PySimpleGUI as sg

colWidth = 55
policyMenu = ["Do-nothing", "Rebalancing", "Fosen&Haldorsen", "F&H-Greedy"] # must be single words
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
    [ sg.Button("Test state"), sg.Button("Save state"), sg.Input("Name: ", key ="-NAME-", size = 25), sg.Button("Load test state")],    
    [sg.Text('_'*colWidth)],
    [sg.Text("Calculate ideal state"), sg.Text("", key="-IDEAL-METHOD-")],
    [sg.Button("Evenly distributed"), sg.Button("Outflow"), sg.Text("", key="-CALC-MSG-")], 
    [sg.Text('_'*colWidth)],
    [sg.Input("Start-day: 2", key="-START-D-", size = 12), sg.Input("Start-hour: 8", key="-START-H-", size = 12), 
        sg.Input("#days: 0", key="-NUM-DAYS-", size = 10), sg.Input("#hours: 16", key="-NUM-HOURS-", size = 10)],
    [sg.Text("Select policy: "), sg.Listbox( values=policyMenu, enable_events=True, size=(17, 4), key="-POLICIES-"), 
        sg.Button("Simulate"), sg.Button("Replay script")],
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
    [sg.Listbox( values=[], enable_events=True, size=(30, 8), key="-FILE LIST-")],
    [sg.Text("The following checkboxes are still not implemented")],
    [sg.Checkbox ('Option 1', key='check_value1'), sg.Checkbox ('Option 2',key='check_value2'), sg.Checkbox ('Grid',key='check_value2')],
    [sg.Button("Test-1"), sg.Button("Test-2"), sg.Button("Test-3"), sg.Button("Trips/week-Oslo")],
    [sg.Text('_'*50)],
]
layout = [ [sg.Column(dashboardColumn), sg.VSeperator(), sg.Column(statusColumn) ] ]
window = sg.Window("FOMO Digital Twin Dashboard 0.2", layout)

