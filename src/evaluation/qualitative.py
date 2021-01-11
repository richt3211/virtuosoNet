from src.logger import init_logger
from src.models.params import Params
from src.neptune import get_experiment_by_id 
from neptune.experiments import Experiment
from src.data.data_reader.read_pre_processed import read_single_score
from src.data.data_reader.read_featurized_cache import read_featurized_stats
from src.data.data_writer.data_writer import write_midi_to_raw
from src.data.post_processing import feature_output_to_midi
from src.constants import CACHE_DATA_DIR, DEVELOPMENT_DATA_DIR, PRODUCTION_DATA_DIR, ROOT_DIR
from src.models.model_writer_reader import read_checkpoint, read_params
from music21 import midi
from dataclasses import asdict, dataclass

import logging
import shutil
import src.old.data_process as dp
import torch
import torch.nn as nn
import os
import zipfile


@dataclass
class QualitativeEvaluatorParams(Params):
    performances = [
        {
            'song_path': f'{PRODUCTION_DATA_DIR}/input/chopin_fantasie_impromptu/', 
            'perf_name': 'chopin_fantasie_impromptu',
            'composer': 'Chopin',
            'is_dev': False
        }, 
        {
            'song_path': f'{PRODUCTION_DATA_DIR}/input/bwv_855_prelude/', 
            'perf_name': 'bach_bwv_855_prelude',
            'composer': 'Bach',
            'is_dev': False

        },
        {
            'song_path': f'{PRODUCTION_DATA_DIR}/input/mozart_sonata_11_1/', 
            'perf_name': 'mozart_sonata_11_1',
            'composer': 'Mozart',
            'is_dev': False

        },
        {
            'song_path': f'{DEVELOPMENT_DATA_DIR}/Beethoven/Piano_Sonatas/17-1/', 
            'perf_name': 'beethoven_sonata_17_1',
            'composer': 'Beethoven',
            'is_dev': True
        },
        {
            'song_path': f'{DEVELOPMENT_DATA_DIR}/Chopin/Etudes_op_10/12/', 
            'perf_name': 'chopin_etudes_op_10_12',
            'composer': 'Chopin',
            'is_dev': True
        },
        {
            'song_path': f'{DEVELOPMENT_DATA_DIR}/Bach/Prelude/bwv_858/', 
            'perf_name': 'bach_bwv_858_prelude',
            'composer': 'Bach',
            'is_dev': True
        },
    ]

class QualitativeEvaluator():
    
    # def __init__(self, params:QualitativeEvaluatorParams, model:nn.Module, exp:Experiment):
    #     self.params = params
    #     self.model = model.to(self.params.device_num)
    #     self.exp = exp

    def __init__(self, exp:Experiment, model:nn.Module):
        self.exp = exp
        self.params = QualitativeEvaluatorParams()
        self.model = model

    def generate_performances(self, model_path:str='./artifacts/model_best.pth'):
        for perf in self.params.performances:
            message = f'Generating performance for {perf["perf_name"]}'
            self.exp.log_text('performance generation', message)
            logging.info(message)
            self.generate_performance_for_file(
                song_path=perf['song_path'],
                perf_name=perf['perf_name'],
                composer_name=perf['composer'],
                model_path=model_path,
            )

    def generate_performance_for_file(self, 
        song_path, 
        perf_name,
        composer_name, 
        model_path, 
        tempo=0, 
        mean_vel=None,
        pedal=True,
        disklavier=False
    ):
        # xml_file_path = f'{PRODUCTION_DATA_DIR}/input/{perf_path}/' 
        means,stds = read_featurized_stats(f'{CACHE_DATA_DIR}/train/training_data_stat.pickle')
        test_x, xml_notes, xml_doc, edges, note_locations = \
            read_single_score(song_path, composer_name, means, stds, mean_vel, tempo )
        
        read_checkpoint(self.model, model_path, self.params.device)
        self.model.eval()
        batch_x = torch.Tensor(test_x)
        input_x = batch_x.to(self.params.device).view(1, -1, self.params.input_size)
        num_notes = len(test_x)
        input_y = torch.zeros(1, num_notes, self.params.output_size).to(self.params.device)
        
        total_output = self.model_inference(input_x, input_y, note_locations, num_notes)

        prediction = torch.cat(total_output, 1)
        output_midi, midi_pedals, output_features = feature_output_to_midi(prediction, note_locations, xml_doc, xml_notes, means, stds)
        write_midi_to_raw(perf_name, self.exp, output_features, output_midi, midi_pedals, pedal, disklavier )

    def model_inference(self, input_x, input_y, note_locations, num_notes):
        total_output = []
        with torch.no_grad():  
            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=self.params.time_steps, overlap=False)
            for slice_idx in slice_indexes:
                batch_start, batch_end = slice_idx
                batch_input = input_x[:, batch_start:batch_end, :].view(1,-1,self.params.input_size)
                temp_outputs = self.model(batch_input) # type: ignore
                total_output.append(temp_outputs)
        return total_output

class LSTMBaselineQualitativeEvaluator(QualitativeEvaluator):
    def __init__(self, exp:Experiment, hyper_params_path:str='./artifacts/params.pickle', is_dynamic_source:bool=True):
        if is_dynamic_source:
            from source.lstm_bl import LSTMBaseline # type: ignore
        else:
            from src.models.lstm_bl import LSTMBaseline
        
        hyper_params = read_params(hyper_params_path)
        model = LSTMBaseline(hyper_params)
        super().__init__(exp, model)

class TransformerEncoderQualitativeEvaluator(QualitativeEvaluator):
    def __init__(self, exp, hyper_params_path:str='./artifacts/params.pickle', is_dynamic_source:bool=True):
        if is_dynamic_source:
            from source.transformer import TransformerEncoder # type: ignore
        else:
            from src.models.transformer import TransformerEncoder
        
        hyper_params = read_params(hyper_params_path)
        model = TransformerEncoder(hyper_params)
        super().__init__(exp, model)


def playMidi(filename):
    mf = midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    s.show('midi')

# TODO: Leaving this here for legacy purposes, but any future perf generation should use the 
# init_evaluation in the utils.py file. 
def init_performance_generation(experiment_id: str, artifacts:list) -> Experiment:
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
    
    for art in artifacts:
        exp.download_artifact(art, f'{cache_dir}')

    exp.download_sources()

    with zipfile.ZipFile('source.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

    return exp 

def legacy_test_run_str(model_code:str, exp_id:str, is_dev:str, pre_train:bool=False, bool_pedal:bool=True, hier_code:str=None):
  data_path = f'{CACHE_DATA_DIR}/train/training_data_development' if is_dev == 'true' else f'{CACHE_DATA_DIR}/train/training_data'
  model_run_script_path = f'{ROOT_DIR}/virtuosoNet/src/old/model_run.py'
  pedal = "true" if bool_pedal else "false"
  p_train = "true" if pre_train else "false"
  run_str = f'{model_run_script_path} -data={data_path} -mode=test_some -code={model_code} {f"-hCode={hier_code}" if hier_code else ""} -is_dev={is_dev} -exp_id={exp_id} -bp={pedal} -pre_train={p_train}'
  return run_str