import pickle
import os 

REGRESSION = True
# ROOT_DIR = os.environ['ROOT_DIR']


def write_featurized_form_to_cache(data, stats, file_path):
    print(os.path.abspath(file_path))
    with open(file_path + ".pickle", "wb") as f:
        pickle.dump({'train': data['train'], 'valid': data['valid']}, f, protocol=2)
    with open(file_path + "_test.pickle", "wb") as f:
        pickle.dump(data['test'], f, protocol=2)

    if REGRESSION:
        with open(file_path + "_stat.pickle", "wb") as f:
            pickle.dump([stats['means'], stats['stds']], f, protocol=2)
    else:
        with open(file_path + "_stat.pickle", "wb") as f:
            pickle.dump([stats['means'], stats['stds'], stats['bins']], f, protocol=2)