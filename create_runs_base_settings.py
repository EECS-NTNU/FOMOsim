from helpers import *

# Duration of each simulation run
NUM_DAYS = 7
DURATION = timeInMinutes(hours=24*NUM_DAYS)

# Enter instance definition here.  

CITIES = [  "Oslo",
            "Bergen",
            "Trondheim",
            "Edinburgh",
            ]
CITY_RANKING = {'EH':0,'TD':1,'BG':2,'OS':3}

ABBRVS = {"Oslo": 'OS',
          "Bergen": 'BG',
          "Trondheim":'TD' ,
          "Edinburgh":'EH'
          }

ABBRVS2 = {v: k for k, v in ABBRVS.items()}

WEEKS = {"Oslo": [10,22,31,50],
          "Bergen": [8,25,35,45],
          "Trondheim":[17,21,34,44] ,
          "Edinburgh":[10,22,31,50]
          }

NUM_SEEDS = {
    "EH_W10":10,"EH_W22":10,    "EH_W31":10,	"EH_W50":10,	
    "TD_W17":10,"TD_W21":15,    "TD_W34":15,	"TD_W44":10,	
    "BG_W8":10, "BG_W25":10,    "BG_W35":10,	"BG_W45":15,    	
    "OS_W10":15,"OS_W22":70,    "OS_W31":70,    "OS_W50":30,
}

# NUM_SEEDS = {
#     "EH_W10":10,"EH_W22":10,    "EH_W31":10,	"EH_W50":10,	
#     "TD_W17":10,"TD_W21":15,    "TD_W34":15,	"TD_W44":10,	
#     "BG_W8":10, "BG_W25":10,    "BG_W35":10,	"BG_W45":15,    	
#     "OS_W10":15,"OS_W22":30,    "OS_W31":30,    "OS_W50":30,
# }

