import pickle
import model_constants as cons

class NetParams:
    class Param:
        def __init__(self):
            self.size = 0
            self.layer = 1
            self.input = 0

    def __init__(self):
        self.note = self.Param()
        self.onset = self.Param()
        self.beat = self.Param()
        self.measure = self.Param()
        self.final = self.Param()
        self.voice = self.Param()
        self.sum = self.Param()
        self.encoder = self.Param()
        self.time_reg = self.Param()
        self.input_size = 0
        self.output_size = 0
        self.graph_iteration = 5


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

    if 'ggnn_non_ar' in model_code:

        net_param.note.layer = 2
        net_param.note.size = 64
        net_param.beat.layer = 2
        net_param.beat.size = 32
        net_param.measure.layer = 1
        net_param.measure.size = 16
        net_param.final.layer = 1
        net_param.final.size = 64

        net_param.encoder.size = 64
        net_param.encoder.layer = 2

        net_param.final.input = (net_param.note.size + net_param.beat.size +
                                 net_param.measure.size) * 2 + net_param.encoder.size + \
                                cons.num_tempo_info + cons.num_dynamic_info
        net_param.encoder.input = (net_param.note.size + net_param.beat.size +
                                   net_param.measure.size) * 2 \
                                  + cons.NUM_PRIME_PARAM

    elif 'ggnn_ar' in model_code:

        net_param.note.layer = 2
        net_param.note.size = 64
        net_param.beat.layer = 2
        net_param.beat.size = 32
        net_param.measure.layer = 1
        net_param.measure.size = 16
        net_param.final.layer = 1
        net_param.final.size = 64

        net_param.encoder.size = 64
        net_param.encoder.layer = 2

        net_param.time_reg.size = 64

        net_param.final.input = (net_param.note.size + net_param.beat.size +
                                 net_param.measure.size) * 2
        net_param.encoder.input = (net_param.note.size + net_param.beat.size +
                                   net_param.measure.size) * 2 \
                                  + cons.NUM_PRIME_PARAM
    elif 'ggnn_simple_ar' in model_code:

        net_param.note.layer = 2
        net_param.note.size = 64
        net_param.beat.layer = 1
        net_param.beat.size = 16
        net_param.measure.layer = 1
        net_param.measure.size = 8
        net_param.final.layer = 1
        net_param.final.size = 64

        net_param.encoder.size = 32
        net_param.encoder.layer = 2

        net_param.time_reg.size = 32

        net_param.final.input = (net_param.note.size + net_param.beat.size +
                                 net_param.measure.size) * 2
        net_param.encoder.input = (net_param.note.size + net_param.beat.size +
                                   net_param.measure.size) * 2 \
                                  + cons.NUM_PRIME_PARAM
    elif 'sequential_ggnn' in model_code:

        net_param.note.layer = 2
        net_param.note.size = 32
        net_param.measure.layer = 1
        net_param.measure.size = 8
        net_param.final.layer = 1
        net_param.final.size = 64

        net_param.encoder.size = 16
        net_param.encoder.layer = 2

        net_param.time_reg.size = 16
        net_param.graph_iteration = 5

        net_param.final.input = (net_param.note.size + net_param.measure.size * 2) * 2
        net_param.encoder.input = (net_param.note.size + net_param.measure.size * 2) * 2 \
                                  + cons.NUM_PRIME_PARAM

    elif 'han' in model_code:
        net_param.note.layer = 2
        net_param.note.size = 64
        net_param.beat.layer = 2
        net_param.beat.size = 32
        net_param.measure.layer = 1
        net_param.measure.size = 16
        net_param.final.layer = 1
        net_param.final.size = 64
        net_param.voice.layer = 2
        net_param.voice.size = 64
        net_param.sum.layer = 2
        net_param.sum.size = 64

        net_param.encoder.size = 64
        net_param.encoder.layer = 2
        net_param.encoder.input = (net_param.note.size + net_param.beat.size +
                                   net_param.measure.size + net_param.voice.size) * 2 \
                                  + cons.NUM_PRIME_PARAM
        num_voice_feed_param = 2  # velocity, onset deviation
        num_tempo_info = 3  # qpm primo, tempo primo
        num_dynamic_info = 0
        net_param.final.input = (net_param.note.size + net_param.voice.size + net_param.beat.size +
                                 net_param.measure.size) * 2 + net_param.encoder.size + \
                                num_tempo_info + num_dynamic_info
        if 'ar' in model_code:
            net_param.final.input += net_param.output_size + num_voice_feed_param

    else:
        print('Unclassified model code')

    return net_param