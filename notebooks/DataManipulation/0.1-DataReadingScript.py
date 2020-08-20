from code.data.data_reader import load_training_data
from pyScoreParser.data_generation import load_entire_subfolder


print("test")

chopin_pairs, num_train_pairs, num_valid_pairs, num_test_pairs = load_entire_subfolder('../../../chopin_cleaned', 4)
# save_features_as_vector(chopin_pairs, num_train_pairs, num_valid_pairs, 'perform_style_set_5')



