# helpers.py

from datetime import datetime, date

def strip(removeStr, wholeStr):
    start = wholeStr.find(removeStr)
    if start == 0:
        return wholeStr[len(removeStr):len(wholeStr)]
    else:
        print("*** ERROR could not remove ", removeStr)

def yearWeekNoAndDay(dateString):
    year, month, day = map(int, dateString.split('-'))
    date1 = date(year, month, day)
    return year, int(date1.isocalendar()[1]), date1.weekday()


def timeInHoursAndMinutes(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return str(h) + ":" + str(m)

def printTime():
    print("Time =", datetime.now().strftime("%H:%M:%S"))