import os

guiLoggDir = "GUI/loggFiles"

os.makedirs(guiLoggDir, exist_ok=True)
loggFile = open(guiLoggDir + "/sessionLog.txt", "w")
scriptFile = open("GUI/scripts/sessioncript.txt", "w")
