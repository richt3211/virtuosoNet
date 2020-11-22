from neptune.experiments import Experiment
from pyScoreParser import xml_matching
import pyScoreParser.performanceWorm as perf_worm
import pickle
import os 

REGRESSION = True
# ROOT_DIR = os.environ['ROOT_DIR']


def write_featurized_form_to_cache(data, stats, file_path):
    print(os.path.abspath(file_path))
    with open(file_path + ".pickle", "wb") as f:
        pickle.dump({'train': data['train'], 'valid': data['valid']}, f, protocol=2)
    with open(file_path + "_test.pickle", "wb") as f:
        pickle.dump(data['test'], f, protocol=2)

    if REGRESSION:
        with open(file_path + "_stat.pickle", "wb") as f:
            pickle.dump([stats['means'], stats['stds']], f, protocol=2)
    else:
        with open(file_path + "_stat.pickle", "wb") as f:
            pickle.dump([stats['means'], stats['stds'], stats['bins']], f, protocol=2)

def write_midi_to_raw(perf_name, exp:Experiment, output_features, output_midi, midi_pedals, pedal=True, disklavier=False):
    if not os.path.exists('./artifacts'):
        os.mkdir('./artifacts')
    
    midi_path = f'./artifacts/{perf_name}.mid'
    plot_path = f'./artifacts/{perf_name}.png'
    perf_worm.plot_performance_worm(output_features, plot_path)
    print(f'Saving midi performance to {midi_path}')
    xml_matching.save_midi_notes_as_piano_midi(
        output_midi, 
        midi_pedals, 
        midi_path, 
        bool_pedal=pedal, 
        disklavier=disklavier
    )

    exp.log_artifact(midi_path, f'{perf_name}.mid')
    exp.log_artifact(plot_path, f'{perf_name}.png')


