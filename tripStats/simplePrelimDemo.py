# main.py
import PySimpleGUI as sg
import os.path
from readTripData import summary

###### GUI layout
json_data_column = [ [ sg.Text("Data Folder"), 
        sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"), 
        sg.FolderBrowse(), ],
    [ sg.Listbox( values=[], enable_events=True, size=(60, 20), key="-FILE LIST-")
    ],
]
result_column = [
    [sg.Text(size=(40, 1), key="-TOUT1-")],
    [sg.Text(size=(60, 1), key="-TOUT2-")],
    [sg.Text(size=(40, 1), key="-TOUT3-")],
]
layout = [[sg.Text("Choose folder for data-files in JSON-format")], 
    [ sg.Column(json_data_column), sg.VSeperator(), sg.Column(result_column), ]
]
window = sg.Window("JSON-parse test code", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "-FOLDER-": # Folder name was filled in, make a list of files in the folder
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
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            window["-TOUT1-"].update("Starts processing of:", font='Lucida', text_color = 'Yellow')
            window["-TOUT2-"].update(filename)
            window["-TOUT3-"].update("... ",font='Lucida', text_color = 'Red')
            summary(filename)

        except:
            pass
window.close()