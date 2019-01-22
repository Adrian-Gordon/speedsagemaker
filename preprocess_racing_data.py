import numpy as np
import pandas as pd

def preprocess(racing_data):
  #filter out invalid speeds and date differences
  racing_data=racing_data[(racing_data["speed1"] <= 20.0)&(racing_data["speed1"] >= 10.0)&(racing_data["speed2"] <= 20.0)&(racing_data["speed2"] >= 10.0)&(racing_data["datediff"] > 0)]

  #filter out invalid weights

  racing_data=racing_data[(racing_data["weight1"] >=100.0)&(racing_data["weight1"] <=170.0)&(racing_data["weight2"] >=100.0)&(racing_data["weight2"] <=170.0)]

  #take log of datediff data, to make the disribution more normal
  racing_data['datediff']=np.log(racing_data['datediff'])


  #generate distancediff

  racing_data["distancediff"] = racing_data["distance2"] - racing_data["distance1"]

  #generate goingdiff

  racing_data["goingdiff"] = racing_data["going2"] - racing_data["going1"]

  #generate weightdiff

  racing_data["weightdiff"] = racing_data["weight2"] - racing_data["weight1"]

  #generate speeddiff

  racing_data["speeddiff"] = racing_data["speed2"] - racing_data["speed1"]

  return(racing_data)

