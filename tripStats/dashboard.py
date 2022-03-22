# dashboard.py
import PySimpleGUI as sg
import os.path
from parse import *
from download import *

###### GUI layout
json_data_column = [
    [sg.Text("Choose folder for data-files in JSON-format", font='Lucida', text_color = 'Yellow')],
    [ sg.Text("Data Folder"),  sg.Input(size=(30, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse(), ],
    [ sg.Listbox( values=[], enable_events=True, size=(40, 30), key="-FILE LIST-")
    ],
]
process_column = [
    [sg.Text("Select processing", font='Lucida', text_color = 'Yellow', key="-TOUT1-")],
    [sg.Text("Download Oslo trips, 1 = April 2019 ... 35 = February 2022)")],
    [sg.Button("All Oslo"), sg.Button("Clear"), sg.Input("From: ", key="-INPUTfrom-"), sg.Input("To: ", key="-INPUTto-")],
    [sg.Button("Download Oslo")],
    [sg.Text('_'*40)],
    [sg.Text("Not implemented")],
    [sg.Button("Bergen"), sg.Button("Utopia")],
    [sg.Text('_'*40)],
    [sg.Text("Compress --- not implemented")],
    [sg.Text('_'*40)],
    [sg.Text("Analyze", size=(20, 1))],
    [sg.Button("Distance matrix"), sg.Button("Leave and arrive intensity"), sg.Button("Check-Test"),sg.Button("CheckAll")],
    [sg.Text('_'*40)],
    [sg.Text(size=(40, 1), key="-TOUT2-")],
    [sg.Button("Exit")]
]
layout = [ [ sg.Column(json_data_column), sg.VSeperator(), sg.Column(process_column), ]
]
window = sg.Window("FOMO Digital Twin Dashboard 0.1", layout)


# checkTest() # placed here during development  CHANGE NAME

folder = "" #starts empty
while True:
    event, values = window.read()
    window["-TOUT1-"].update("Select processing:", font='Lucida', text_color = 'Yellow')
    if event == "All Oslo":
        print("All-Oslo-button pressed")
        window["-INPUTfrom-"].update("From: 1")
        window["-INPUTto-"].update("To: 35")  # TODO, Magic number, move to loacl settings 
    elif event == "Clear":
        window["-INPUTfrom-"].update("From: ")
        window["-INPUTto-"].update("To: ")   
    elif event == "Download Oslo":
        # print("values:", values)  # debug 
        oslo(values["-INPUTfrom-"], values["-INPUTto-"])
    elif event == "Distance matrix":
        dm = calcDistances("Oslo", "all")    
    elif event == "Leave and arrive intensity":
        leaveIntensity, arriveIntensity = calcIntensity("Oslo", "all", 60) #  last param is length of period
        pass
    elif event == "CheckAll":
        checkAll(folder)
    elif event == "Check-Test":
        checkTest()    
    elif event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "-FOLDER-": # Folder name was filled in, make a list of files in the folder
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".json"))
        ]
        window["-FILE LIST-"].update(fnames)
        window["-TOUT1-"].update("Choose a data file from the list on left:")
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filepath = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            window["-TOUT2-"].update(filepath)
            print(filepath)
            summary(filepath)
        except:
            print("Exception prcessing file")
window.close()