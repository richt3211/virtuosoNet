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
from src.experiments.training.BL.HANBaseline_training import run_han_bl_job


# %%
logger_path = 'logs/run_dev_data_0.1.log'
init_logger(logger_path)


# %%
path = f'{CACHE_DATA_DIR}/train/training_data_development.pickle'
dev_data = read_featurized(path)


# %%
num_epochs = 5
run_han_bl_job(dev_data, num_epochs, version=0.3, is_dev=True)


# %%



