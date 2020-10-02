from pyScoreParser.data_generation import convert_features_to_vector

def featurize_pre_processed(chopin_pairs, num_train_pairs, num_valid_pairs):
    features,stats = convert_features_to_vector(chopin_pairs, num_train_pairs, num_valid_pairs)
    return features,stats