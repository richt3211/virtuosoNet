import pickle
from src.old import model_constants as cons
# import model_constants as cons

class NetParams:
    class Param:
        def __init__(self):
            self.size = 0
            self.layer = 1
            self.input = 0
            self.margin = 0

    def __init__(self):
        self.note = self.Param()
        self.onset = self.Param()
        self.beat = self.Param()
        self.measure = self.Param()
        self.section = self.Param()
        self.final = self.Param()
        self.voice = self.Param()
        self.sum = self.Param()
        self.encoder = self.Param()
        self.time_reg = self.Param()
        self.margin = self.Param()
        self.input_size = 0
        self.output_size = 0
        self.encoded_vector_size = 16
        self.graph_iteration = 5
        self.sequence_iteration = 5
        self.num_edge_types = 10
        self.num_attention_head = 8
        self.is_graph = False
        self.is_teacher_force = False
        self.is_baseline = True
        self.hierarchy_level = None
        self.is_simplified = False
        self.is_test_version = False
        self.training_args = None


def save_parameters(param, save_name):
    with open(save_name + ".dat", "wb") as f:
        pickle.dump(param, f, protocol=2)


def load_parameters(file_name):
    with open(file_name + ".dat", "rb") as f:
        u = pickle._Unpickler(f)
        net_params = u.load()
        return net_params


def initialize_model_parameters_by_code(model_code):
    net_param = NetParams()
    net_param.input_size = cons.SCORE_INPUT
    net_param.output_size = cons.NUM_PRIME_PARAM

    if 'isgn' in model_code:
        net_param.note.layer = 2
        net_param.note.size = 128
        net_param.measure.layer = 2
        net_param.measure.size = 64
        net_param.final.margin = 32

        net_param.encoded_vector_size = 16
        net_param.encoder.size = 128
        net_param.encoder.layer = 2

        net_param.time_reg.layer = 2
        net_param.time_reg.size = 32
        net_param.graph_iteration = 4
        net_param.sequence_iteration = 3

        net_param.final.input = (net_param.note.size + net_param.measure.size * 2) * 2
        net_param.encoder.input = (net_param.note.size + net_param.measure.size * 2) * 2 \
                                  + cons.NUM_PRIME_PARAM
        if 'sggnn_note' in model_code:
            net_param.final.input += net_param.note.size
            net_param.encoder.input += net_param.note.size

        if 'baseline' in model_code:
            net_param.is_baseline = True

    elif 'han' in model_code:
        net_param.note.layer = 2
        net_param.note.size = 64
        net_param.beat.layer = 2
        net_param.beat.size = 64
        net_param.measure.layer = 1
        net_param.measure.size = 64
        net_param.final.layer = 1
        net_param.final.size = 64
        net_param.voice.layer = 2
        net_param.voice.size = 64

        # net_param.num_attention_head = 1
        net_param.encoded_vector_size = 16
        net_param.encoder.size = 32
        net_param.encoder.layer = 2
        net_param.encoder.input = (net_param.note.size + net_param.beat.size +
                                   net_param.measure.size + net_param.voice.size) * 2 \
                                  + cons.NUM_PRIME_PARAM
        num_tempo_info = 3  # qpm primo, tempo primo
        num_dynamic_info = 0
        net_param.final.input = (net_param.note.size + net_param.voice.size + net_param.beat.size +
                                 net_param.measure.size) * 2 + net_param.encoder.size + \
                                num_tempo_info + num_dynamic_info
        if 'graph' in model_code:
            net_param.is_graph = True
            net_param.graph_iteration = 3
            net_param.encoder.input = (net_param.note.size + net_param.beat.size +
                                       net_param.measure.size) * 2 \
                                      + cons.NUM_PRIME_PARAM
            net_param.final.input = (net_param.note.size +  net_param.beat.size +
                                     net_param.measure.size) * 2 + net_param.encoder.size + \
                                    num_tempo_info + num_dynamic_info
        if 'ar' in model_code:
            net_param.final.input += net_param.output_size

        if 'teacher' in model_code:
            net_param.is_teacher_force = True
        if 'baseline' in model_code:
            net_param.is_baseline = True
            net_param.encoder.input = net_param.note.size * 2 + cons.NUM_PRIME_PARAM
            net_param.final.input = net_param.note.size * 2 + net_param.encoder.size + num_tempo_info + num_dynamic_info + net_param.output_size

    elif 'trill' in model_code:
        net_param.input_size = cons.SCORE_INPUT + cons.NUM_PRIME_PARAM
        net_param.output_size = cons.num_trill_param
        net_param.note.layer = 2
        net_param.note.size = 32

    else:
        print('Unclassified model code')

    if 'measure' in model_code:
        net_param.hierarchy_level = 'measure'
        net_param.output_size = 2
        net_param.encoder.input += 2 - cons.NUM_PRIME_PARAM
    elif 'beat' in model_code:
        net_param.hierarchy_level = 'beat'
        net_param.output_size = 2
        net_param.encoder.input += 2 - cons.NUM_PRIME_PARAM
    elif 'note' in model_code:
        net_param.input_size += 2

    if 'altv' in model_code:
        net_param.is_test_version = True

    return net_param

class ModelRunParams():

    def __init__(self, params):
        print(params)
        self.sessMode = set_param(params, "mode", "test")
        self.testPath = set_param(params, "path", "./data/test_pieces/bps_5_1")
        self.dataName = set_param(params, "data", "self.training_data")
        self.resume = set_param(params, "resume", "_best.pth.tar")
        self.startTempo = set_param(params, "tempo", False)
        self.trainTrill = set_param(params, "trill", False)
        self.slurEdge = set_param(params, "slur", False)
        self.voiceEdge = set_param(params, "voice", True)
        self.velocity = set_param(params, "vel", "50,65")
        self.device = set_param(params, "dev", 1)
        self.modelCode = set_param(params, "code", "isgn")
        self.trillCode = set_param(params, "-tCode", "trill_default")
        self.composer = set_param(params, "comp", "Beethoven")
        self.latent = set_param(params, "latent", 0)
        self.boolPedal = set_param(params, "bp", False)
        self.trainingLoss = set_param(params, "loss", "MSE")
        self.resumeTraining = set_param(params, "reTrain", False)
        self.perfName = set_param(params, "perf", "Anger_sub1")
        self.deltaLoss = set_param(params, "delta", "test")
        self.hierCode = set_param(params, "hCode", "han_ar_measure")
        self.intermediateLoss = set_param(params, "intermd", True)
        self.randomTrain = set_param(params, "randtr", True)
        self.disklavier = set_param(params, "dskl", True)
        self.multi_instruments = set_param(params, "multi", False)       


def set_param(params, param, default):
    if param in params:
        return params[param]
    else:
        return default
