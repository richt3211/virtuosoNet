import os 
import pickle
from pyScoreParser.data_generation import load_limited_subfolder, convert_features_to_vector

# for now I am only going to use a function that reads in the training data
# will eventually want to add all of the data pipeline to here as well


def load_development_training_data(file_path):
    train_list = ['Mozart/Piano_Sonatas/8-1/']
    valid_list = ['Bach/Fugue/bwv_874/']
    test_list = ['Liszt/Transcendental_Etudes/5/']

    print(os.path.abspath(file_path))
    chopin_pairs, num_train_pairs, num_valid_pairs, num_test_pairs = load_limited_subfolder(file_path, train_list, valid_list, test_list, 2)

    features = convert_features_to_vector(chopin_pairs, num_train_pairs, num_valid_pairs)

    return features

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
