# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
from src.logger import init_logger


# %%
from src.constants import CACHE_DATA_DIR
from src.data.data_reader.read_featurized_cache import read_featurized
from src.experiments.training.Transformer.TransformerEncoder_training import TransformerEncoderJob
from src.models.model_run_job import ModelJobParams


# %%
logger_path = 'logs/run_dev_data_0.2.log'
init_logger(logger_path)


# %%
path = f'{CACHE_DATA_DIR}/train/training_data_development.pickle'
dev_data = read_featurized(path)


# %%
num_epochs = 15
run_params = ModelJobParams(is_dev=True)
training_job = TransformerEncoderJob(run_params)
training_job.run_job(dev_data, num_epochs, version=0.2)


# %%
logger_path = 'logs/run_full_data_0.1.log'
init_logger(logger_path)


# %%
path = f'{CACHE_DATA_DIR}/train/training_data.pickle'
data = read_featurized(path)


# %%
num_epochs = 50
run_params = ModelJobParams(is_dev=True)
training_job = TransformerEncoderJob(run_params)
training_job.run_job(data, num_epochs, version=0.2)


# %%



