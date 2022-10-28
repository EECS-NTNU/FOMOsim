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
    "TD_W17":30,"TD_W21":30,    "TD_W34":45,	"TD_W44":35,	
    "BG_W8":35, "BG_W25":35,    "BG_W35":30,	"BG_W45":45,    	
    "OS_W10":15,"OS_W22":25,    "OS_W31":25,    "OS_W50":15,
}

