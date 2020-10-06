import src 
from src.data.data_reader.read_pre_processed import read_pre_processed
from src.data.pre_processing import featurize_pre_processed
from src.data.data_writer.data_writer import write_featurized_form_to_cache


print('\n'.join(sys.path))

read_pre_processed()
