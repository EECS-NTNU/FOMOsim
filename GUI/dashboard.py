# dashboard.py
import PySimpleGUI as sg
import os.path
from tripStats.parse import *
from tripStats.download import *

###### GUI layout
dashboardColumn = [
    [sg.Text("Preparation and set up ", font='Lucida', text_color = 'Yellow')],
    [sg.Text("Download Oslo trips, 1 = April 2019 ... 35 = February 2022)")],
    [sg.Button("All Oslo"), sg.Button("Clear"), sg.Input("From: ", key="-INPUTfrom-", size=8), 
        sg.Input("To: ", key="-INPUTto-", size = 6)],
    [sg.Button("Download Oslo")],
    [sg.Text('_'*40)],
    [sg.Text("Select city (Oslo is default)")],
    [sg.Radio("Oslo", "RADIO1", key = "-OSLO-", default=True), 
        sg.Radio("Bergen", "RADIO1", key = "-BERGEN-"), sg.Radio("Utopia", "RADIO1", key = "-UTOPIA-")],
    [sg.Button("Find stations and distances")], 
    [sg.Text(size=(40, 1), key="-TOUT2-", text_color = "Red")],
    [sg.Text('_'*40)],
    [sg.Text("Simulate")],
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

def GUI_main():
    while True:
        event, values = window.read()
        if event == "All Oslo":
            print("All-Oslo-button pressed")
            window["-INPUTfrom-"].update("From: 1")
            window["-INPUTto-"].update("To: 35")  # TODO, Magic number, move to local settings 
        elif event == "Clear":
            window["-INPUTfrom-"].update("From: ")
            window["-INPUTto-"].update("To: ")   
        elif event == "Download Oslo":
            oslo(values["-INPUTfrom-"], values["-INPUTto-"])
        elif event == "Find stations and distances":
            if values["-OSLO-"]:
                window["-TOUT2-"].update("City OK", text_color = "LightGreen") 
                calcDistances("Oslo")
            elif values["-BERGEN-"]:
                window["-TOUT2-"].update("City not yet implemented", text_color = "red") 
            elif values["-UTOPIA-"]:
                window["-TOUT2-"].update("City OK", text_color = "LightGreen") 
                calcDistances("Utopia")
            else:
                print("*** Error: wrong value from Radiobutton")         
        elif event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()