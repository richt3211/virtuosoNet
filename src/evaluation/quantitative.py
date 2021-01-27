from datetime import datetime
from src.experiments.training.Transformer.transformer_training import TransformerEncoderJob, TransformerEncoderJobParams
from src.logger import init_logger
from src.models.model_run_job import ModelJob, ModelJobParams
from src.models.params import Params
from src.models.transformer import TransformerEncoder
from src.neptune import get_experiment_by_id, log_neptune_timeline 
from neptune.experiments import Experiment
from src.data.data_reader.read_pre_processed import read_single_score
from src.data.data_reader.read_featurized_cache import read_featurized_stats, read_featurized_test
from src.data.data_writer.data_writer import write_midi_to_raw
from src.data.post_processing import feature_output_to_midi
from src.constants import CACHE_DATA_DIR, DEVELOPMENT_DATA_DIR, PRODUCTION_DATA_DIR, ROOT_DIR
from src.models.model_writer_reader import read_checkpoint, read_params, read_training_params
from music21 import midi
from dataclasses import asdict, dataclass

import logging
import shutil
import src.old.data_process as dp
import torch
import torch.nn as nn
import os
import zipfile


# @dataclass
# class QualitativeEvaluatorParams(Params):

def log_evaluation_text(log:str, exp:Experiment):
    exp.log_text('evaluation', f'{datetime.now()} - {log}')
    logging.info(log)

class QuantitativeEvaluator(TransformerEncoderJob):
    
    def __init__(self, params:TransformerEncoderJobParams,  model:TransformerEncoder, exp:Experiment):
        super().__init__(params, model, exp)

    def run_quantitative_evaluation_on_test_set(self, model_path, data):
        log_evaluation_text('Running test set evaluation', self.exp)
        # read in the model
        log_evaluation_text('Reading in model', self.exp)
        read_checkpoint(self.model, model_path, self.params.device)

        # read in test data. 
        file_path = f'{CACHE_DATA_DIR}/train/training_data_test.pickle'
        log_evaluation_text('Reading in test data', self.exp)
        # data = read_featurized_test(file_path, self.exp)

        log_evaluation_text('Running evaluation', self.exp)
        valid_loss, valid_feature_loss = self.evaluate(self.model, data)
        x_axis = 1
        # setting 'x_axis' to 1. In training this value is usually the epoch_num, but for evaluation we only run it once
        self.log_loss(valid_loss, valid_feature_loss, 'test', x_axis)


class LSTMBaselineQuantitativeEvaluator(QuantitativeEvaluator):
    def __init__(self, exp:Experiment, hyper_params_path:str='./artifacts/params.pickle', train_params_path='./artifacts/train_params.pickle', is_dynamic_source:bool=True):
        if is_dynamic_source:
            from source.lstm_bl import LSTMBaseline # type: ignore
        else:
            from src.models.lstm_bl import LSTMBaseline
        
        hyper_params = read_params(hyper_params_path)
        training_params = read_training_params(train_params_path, exp)
        model = LSTMBaseline(hyper_params)
        super().__init__(training_params, model, exp)

class TransformerEncoderQuantitativeEvaluator(QuantitativeEvaluator):
    def __init__(self, exp, hyper_params_path:str='./artifacts/params.pickle', train_params_path='./artifacts/train_params.pickle',is_dynamic_source:bool=True):
        if is_dynamic_source:
            from source.transformer import TransformerEncoder # type: ignore
        else:
            from src.models.transformer import TransformerEncoder
        
        hyper_params = read_params(hyper_params_path)
        training_params = read_training_params(train_params_path, exp)
        model = TransformerEncoder(hyper_params)
        super().__init__(training_params, model, exp)


# def playMidi(filename):
#     mf = midi.MidiFile()
#     mf.open(filename)
#     mf.read()
#     mf.close()
#     s = midi.translate.midiFileToStream(mf)
#     s.show('midi')



# def legacy_test_run_str(model_code:str, exp_id:str, is_dev:str, pre_train:bool=False, bool_pedal:bool=True, hier_code:str=None):
#   data_path = f'{CACHE_DATA_DIR}/train/training_data_development' if is_dev == 'true' else f'{CACHE_DATA_DIR}/train/training_data'
#   model_run_script_path = f'{ROOT_DIR}/virtuosoNet/src/old/model_run.py'
#   pedal = "true" if bool_pedal else "false"
#   p_train = "true" if pre_train else "false"
#   run_str = f'{model_run_script_path} -data={data_path} -mode=test_some -code={model_code} {f"-hCode={hier_code}" if hier_code else ""} -is_dev={is_dev} -exp_id={exp_id} -bp={pedal} -pre_train={p_train}'
#   return run_str