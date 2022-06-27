
import PySimpleGUI as sg

colWidth = 55
policyMenu = ["Do-nothing", "Random", "FH-Greedy", "HHS-Greedy", "FH"] # must be single words. TODO HHS (tensorflow-based) not included
simOptions = [
    [sg.Checkbox('Logg traffic', key='-LOGG-TRAFFIC-')]
]

dashboardColumn = [
    [sg.Text("Prep. and set up ", font='Lucida', text_color = 'Yellow'), sg.VSeparator(), sg.Button("Fast-Track", 
        button_color = "forest green"), sg.Button("main.py", button_color="snow4"), sg.Button("Exit")],
    [sg.Text("Set up simulation", font="Helvetica 14", size=(30, 1), text_color = "spring green", key="-FEEDBACK-")],
    [sg.Text('_'*colWidth)],
    [sg.Text("Dowload trip data")],
    [sg.Button("Oslo"), sg.Button("Oslo vinter"), sg.Button("Bergen"), sg.Button("Edinburgh"), sg.Button("Trondheim")], 
    [sg.Text('_'*colWidth)],
    [sg.Text("Set initial state"), sg.Text("", key = "-STATE-MSG-")],
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
    [sg.Text('_'*colWidth)],
    [sg.Button("Replay script")],
]
layout = [ [sg.Column(dashboardColumn), sg.VSeperator(), sg.Column(statusColumn) ] ]
window = sg.Window("FOMO Digital Twin Dashboard 0.2", layout)

