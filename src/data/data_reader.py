import os 
import pickle

# for now I am only going to use a function that reads in the training data
# will eventually want to add all of the data pipeline to here as well

def load_training_data(file_path):
    print('Loading the training data...')
    if not os.path.isfile(file_path):
        print(f'File {file_path} doesn\'t exist')
    with open(file_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()

    print('Done loading training data')
    return complete_xy
