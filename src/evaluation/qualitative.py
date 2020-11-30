from dataclasses import asdict, dataclass
import logging
import shutil
from src.logger import init_logger
from src.models.params import Params
from src.neptune import get_experiment_by_id, log_neptune_timeline
from neptune.experiments import Experiment
from src.data.data_reader.read_pre_processed import read_single_score
from src.data.data_reader.read_featurized_cache import read_featurized_stats
from src.data.data_writer.data_writer import write_midi_to_raw
from src.data.post_processing import feature_output_to_midi
from src.keys import NEPTUNE_TOKEN
from src.models.model_run_job import ModelJob, ModelJobParams
from src.constants import CACHE_DATA_DIR, PRODUCTION_DATA_DIR, ROOT_DIR
from src.models.model_writer_reader import read_checkpoint
from music21 import midi

import src.old.data_process as dp
import torch
import torch.nn as nn
import os
import zipfile


@dataclass
class QualitativeEvaluatorParams(Params):
    performances = [
        {'song_name': 'chopin_fantasie_impromptu', 'composer': 'Chopin'}, 
        {'song_name': 'bwv_855_prelude', 'composer': 'Bach'},
        {'song_name': 'mozart_sonata_11_1', 'composer': 'Mozart'}
    ]

class QualitativeEvaluator():
    
    def __init__(self, params:QualitativeEvaluatorParams, model:nn.Module, exp:Experiment):
        self.params = params
        self.model = model.to(self.params.device_num)
        self.exp = exp

    def generate_performances(self, model_path):
        for perf in self.params.performances:
            message = f'Generating performance for {perf["song_name"]}'
            self.exp.log_text('performance generation', message)
            logging.info(message)
            self.generate_performance_for_file(
                perf_name=perf['song_name'],
                composer_name=perf['composer'],
                model_path=model_path,
            )

    def generate_performance_for_file(self, 
        perf_name, 
        composer_name, 
        model_path, 
        tempo=0, 
        mean_vel=None,
        pedal=True,
        disklavier=False
    ):
        xml_file_path = f'{PRODUCTION_DATA_DIR}/input/{perf_name}/' 
        means,stds = read_featurized_stats(f'{CACHE_DATA_DIR}/train/training_data_stat.pickle')
        test_x, xml_notes, xml_doc, edges, note_locations = \
            read_single_score(xml_file_path, composer_name, means, stds, mean_vel, tempo )
        
        read_checkpoint(self.model, model_path, self.params.device_num)
        self.model.eval()
        batch_x = torch.Tensor(test_x)
        input_x = batch_x.to(self.params.device_num).view(1, -1, self.params.input_size)
        num_notes = len(test_x)
        input_y = torch.zeros(1, num_notes, self.params.output_size).to(self.params.device)
        
        total_output = self.model_inference(input_x, input_y, note_locations, num_notes)

        prediction = torch.cat(total_output, 1)
        output_midi, midi_pedals, output_features = feature_output_to_midi(prediction, note_locations, xml_doc, xml_notes, means, stds)
        write_midi_to_raw(perf_name, self.exp, output_features, output_midi, midi_pedals, pedal, disklavier )

    def model_inference(self, input_x, input_y, note_locations, num_notes):
        pass

class HANBLQualitativeEvaluator(QualitativeEvaluator):
    def __init__(self, params, model, exp):
        super().__init__(params, model, exp)

    def model_inference(self, input_x, input_y, note_locations, num_notes):
        total_output = []
        with torch.no_grad():  
            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=self.params.time_steps, overlap=False)
            for slice_idx in slice_indexes:
                batch_start, batch_end = slice_idx
                batch_input = input_x[:, batch_start:batch_end, :].view(1,-1,self.params.input_size)
                batch_input_y = input_y[:, batch_start:batch_end, :].view(1,-1,self.params.output_size)
                temp_outputs,_,_,_ = self.model(batch_input, batch_input_y, note_locations, batch_start, initial_z='zero')
                total_output.append(temp_outputs)
        return total_output

class TransformerEncoderQualitativeEvaluator(QualitativeEvaluator):
    def __init__(self, params, model, exp):
        super().__init__(params, model, exp)

    def model_inference(self, input_x, input_y, note_locations, num_notes):
        total_output = []
        with torch.no_grad():  
            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=self.params.time_steps, overlap=False)
            for slice_idx in slice_indexes:
                batch_start, batch_end = slice_idx
                batch_input = input_x[:, batch_start:batch_end, :].view(1,-1,self.params.input_size)
                temp_outputs = self.model(batch_input)
                total_output.append(temp_outputs)
        return total_output

def playMidi(filename):
    mf = midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    s.show('midi')

def init_performance_generation(experiment_id: str, is_dev:bool, is_legacy:bool) -> Experiment:
    '''Initalizes and creates a neptune experiment.'''  

    cache_dir = './artifacts'
    for path in [cache_dir, 'source', 'trill_source']:
        if os.path.exists(path):
            shutil.rmtree(path)


    os.mkdir(cache_dir)
    # load the trill model and params
    exp:Experiment = get_experiment_by_id('THESIS-75')
    exp.download_artifact('trill_params.pickle', cache_dir)
    exp.download_artifact('trill_best.pth', cache_dir)
    exp.download_sources()

    with zipfile.ZipFile('source.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
    os.rename('./source', './trill_source')

    init_logger()
    exp:Experiment = get_experiment_by_id(experiment_id)
        
    # using bad names for now. Will update with correct experiment
    if is_legacy:
        model_path = 'prime_model_dev_best.pth' if is_dev else "prime_model_best.pth"
    else:
        model_path = 'model_dev_best.pth' if is_dev else "_best.pth"

    # download model and hyper params
    # if is_legacy:
    #     model_path = 'prime_model_dev_best.pth' if is_dev else "prime_model_best.pth"
    # else:
    #     model_path = 'model_dev_best.pth' if is_dev else "model_best.pth"
    exp.download_artifact(model_path, f'{cache_dir}')
    exp.download_artifact('params.pickle', f'{cache_dir}')
    exp.download_sources()

    with zipfile.ZipFile('source.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

    return exp 

def legacy_test_run_str(model_code:str, exp_id:str, is_dev:str, bool_pedal:bool=True):
  data_path = f'{CACHE_DATA_DIR}/train/training_data_development' if is_dev == 'true' else f'{CACHE_DATA_DIR}/train/training_data'
  model_run_script_path = f'{ROOT_DIR}/virtuosoNet/src/old/model_run.py'
  pedal = "true" if bool_pedal else "false"
  run_str = f'{model_run_script_path} -data={data_path} -mode=test_some -code={model_code} -is_dev={is_dev} -exp_id={exp_id} -bp={pedal}'
  return run_str