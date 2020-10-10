import pickle


def read_featurized(file_path):
    print('Loading the training data')
    with open(file_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()

    train_xy = complete_xy['train']
    test_xy = complete_xy['valid']
    print('number of train performances: ', len(train_xy), 'number of valid perf: ', len(test_xy))
    print('training sample example', train_xy[0][0][0])

    return complete_xy
