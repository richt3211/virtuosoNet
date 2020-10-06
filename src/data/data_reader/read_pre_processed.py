import os 
import traceback
from pyScoreParser import dataset_split as split
from pyScoreParser.data_generation import get_piece_pairs

VALID_LIST = split.VALID_LIST
TEST_LIST = split.TEST_LIST

def get_train_val_test_lists(path):
    # path = "../../../chopin_cleaned"
    midi_list = []
    if path[len(path)-1] != "/":
        path = f'{path}/'
    for dp,_,filenames in os.walk(path):
        # print(dp)
        for f in filenames:
            if f == "midi_cleaned.mid":
                midi_file = dp.replace(path, "")
                midi_file = midi_file + "/"
                midi_list.append(midi_file)

    train_list = [os.path.join(path,f) for f in midi_list if f not in VALID_LIST and f not in TEST_LIST]
    valid_list = [os.path.join(path, f) for f in VALID_LIST]
    test_list = [os.path.join(path,f) for f in TEST_LIST]
    return (train_list, valid_list, test_list)

def load_limited_subfolder(path, data_split, minimum_perform_limit):
    train_list_folder = [os.path.join(path,f) for f in data_split['train']]
    valid_list_folder = [os.path.join(path, f) for f in data_split['valid']]
    test_list_folder = [os.path.join(path,f) for f in data_split['test']]

    return load_lists(train_list_folder, valid_list_folder, test_list_folder, minimum_perform_limit)

def load_entire_subfolder(path, minimum_perform_limit=0):
    (train_list, valid_list, test_list) = get_train_val_test_lists(path)
    print(f'Length of training list: {len(train_list)}')
    return load_lists(train_list, valid_list, test_list, minimum_perform_limit)

def load_lists(train_list, valid_list, test_list, minimum_perform_limit):
    print(f'Length of training list: {len(train_list)}')
    entire_pairs = []
    try: 
        print('getting the training list')
        (entire_pairs, num_train_pairs) = get_piece_pairs(train_list, minimum_perform_limit, entire_pairs)
        print('getting the validation list')
        (entire_pairs, num_valid_pairs) = get_piece_pairs(valid_list, minimum_perform_limit, entire_pairs)
        print('getting the test list')
        (entire_pairs, num_test_pairs) = get_piece_pairs(test_list, minimum_perform_limit, entire_pairs)
    except: 
            print()
            print("Error reading the training data")
            traceback.print_exc()
            print()

    print('Number of train pairs: ', num_train_pairs, 'valid pairs: ', num_valid_pairs, 'test pairs: ', num_test_pairs)
    # print('Number of total score notes, performance notes, non matched notes, excluded notes: ', NUM_SCORE_NOTES, NUM_PERF_NOTES, NUM_NON_MATCHED_NOTES, NUM_EXCLUDED_NOTES)
    return entire_pairs, num_train_pairs, num_valid_pairs, num_test_pairs

def read_pre_processed(folder_path, data_split=None):
    if data_split:
        return load_limited_subfolder(folder_path, data_split, 2)
    else:
        return load_entire_subfolder(folder_path, 2)