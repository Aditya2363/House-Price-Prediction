import json
import pickle
import numpy as np

_locations = None
_data_columns = None
_model = None

def get_estimated_price(location, sqft, bhk, bath):
  try:
    loc_index = _data_columns.index(location.lower())
  except:
    loc_index = -1

  x = np.zeros(len(_data_columns))
  x[0] = sqft
  x[1] = bath
  x[2] = bhk
  if loc_index >= 0:
    x[loc_index] = 1

  return round(_model.predict([x])[0], 2)

def get_location_names():
    return _locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global _data_columns
    global _locations

    with open("/content/drive/MyDrive/dataset/Server/artifacts/columns.json", 'r') as f:
        _data_columns = json.load(f)['data_columns']
        _locations = _data_columns[3:]

    global _model
    with open("/content/drive/MyDrive/dataset/Server/artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        _model = pickle.load(f)

    print("loading saved artifacts...done")

if __name__ == '__main__':
  load_saved_artifacts()
  print(get_location_names())
  print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
  print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
  print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
  print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
