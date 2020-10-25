from src.data.data_reader.read_pre_processed import read_single_score
from src.data.data_reader.read_featurized_cache import read_featurized_stats
from src.data.data_writer.data_writer import write_midi_to_raw
from src.data.post_processing import feature_output_to_midi
from src.models.model_run_job import ModelJob
from src.constants import CACHE_DATA_DIR
from src.models.model_writer_reader import read_checkpoint

import src.old.data_process as dp
import torch


class QualitativeEvaluator(ModelJob):
    
    def __init__(self, params, model):
        super().__init__(params)
        self.model = model

    def generate_performance_for_file(self, 
        xml_file_path, 
        midi_file_path, 
        plot_path,
        composer_name, 
        model_path, 
        tempo=None, 
        mean_vel=None,
        pedal=True,
        disklavier=False
    ):
        means,stds = read_featurized_stats(f'{CACHE_DATA_DIR}/train/training_data_stat.pickle')
        test_x, xml_notes, xml_doc, edges, note_locations = \
            read_single_score(xml_file_path, composer_name, means, stds, mean_vel, tempo )
        
        batch_x = torch.Tensor(test_x)
        num_notes = len(test_x)
        input_y = torch.zeros(1, num_notes, self.params.num_output).to(self.params.device)
        
        model = read_checkpoint(self.params.model, model_path, self.params.device)
        total_output = []
        with torch.no_grad():  
            model.eval()
            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=self.params.time_steps, overlap=False)
            for slice_idx in slice_indexes:
                batch_start, batch_end = slice_idx
                batch_input = input[:, batch_start:batch_end, :].view(1,-1,self.model.input_size)
                batch_input_y = input_y[:, batch_start:batch_end, :].view(1,-1,self.model.output_size)
                temp_outputs, = model(batch_input)
                total_output.append(temp_outputs)

        prediction = torch.cat(total_output, 1)
        output_midi, midi_pedals, output_features = feature_output_to_midi(prediction, note_locations, xml_doc, xml_notes, means, stds)
        write_midi_to_raw(midi_file_path, plot_path, output_features, output_midi, midi_pedals, pedal, disklavier )