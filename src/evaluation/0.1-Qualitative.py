# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
from src.evaluation.qualitative import (
    QualitativeEvaluatorParams, 
    TransformerEncoderQualitativeEvaluator, 
    playMidi, 
    init_performance_generation
)
from src.models.model_run_job import ModelJobParams
from src.constants import PRODUCTION_DATA_DIR, DEVELOPMENT_DATA_DIR, CACHE_MODEL_DIR
from src.neptune import get_experiment_by_id


# %%
exp = init_performance_generation('THESIS-40', 'transformer.py', is_dev=True)


# %%
from models.transformer import TransformerEncoder
from src.models.model_writer_reader import read_params


# %%
hyper_params = read_params('artifacts/params.pickle')
model = TransformerEncoder(hyper_params)


# %%
params = QualitativeEvaluatorParams(is_dev=True)
qualitative_evaluator = TransformerEncoderQualitativeEvaluator(params, model)


# %%
model_path = './artifacts/model_dev_best.pth'
qualitative_evaluator.generate_performances(model_path)


# %%
qualitative_evaluator.generate_performance_for_file(
    xml_file_path=xml_file_path,
    midi_file_path=midi_file_path,
    plot_path=plot_file_path,
    composer_name='Bach',
    model_path=model_path
)


# %%
performance_midi = midi_file_path
score_midi = f'{DEVELOPMENT_DATA_DIR}/Bach/Fugue/bwv_858/midi_cleaned.mid'


# %%
playMidi(performance_midi)


# %%
playMidi(score_midi)


# %%
xml_file_path = f'{PRODUCTION_DATA_DIR}/input/bwv_858_fugue/'
midi_file_path = f'{PRODUCTION_DATA_DIR}/output/bwv_858_fugue/te_bl.mid'
plot_file_path = f'{PRODUCTION_DATA_DIR}/output/bwv_858_fugue/te_bl.png'
model_path = f'{CACHE_MODEL_DIR}/Transformer/TransformerEncoder/v0.1_best.pth'
qualitative_evaluator.generate_performance_for_file(
    xml_file_path=xml_file_path,
    midi_file_path=midi_file_path,
    plot_path=plot_file_path,
    composer_name='Bach',
    model_path=model_path
)


