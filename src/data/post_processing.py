from pyScoreParser import xml_matching

import numpy as np

def feature_output_to_midi(prediction, note_locations, xml_doc, xml_notes,means, stds, multi_instruments=False):
    prediction = scale_model_prediction_to_original(prediction, means, stds)

    output_features = xml_matching.model_prediction_to_feature(prediction)
    output_features = xml_matching.add_note_location_to_features(output_features, note_locations)

    output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1,
                                                        predicted=True)
    output_midi, midi_pedals = xml_matching.xml_notes_to_midi(output_xml, multi_instruments)
    return output_midi, midi_pedals, output_features

def scale_model_prediction_to_original(prediction, MEANS, STDS):
    for i in range(len(STDS)):
        for j in range(len(STDS[i])):
            if STDS[i][j] < 1e-4:
                STDS[i][j] = 1
    prediction = np.squeeze(np.asarray(prediction.cpu()))
    num_notes = len(prediction)
    for i in range(11):
        prediction[:, i] *= STDS[1][i]
        prediction[:, i] += MEANS[1][i]
    # for i in range(11, 15):
    #     prediction[:, i] *= STDS[1][i+4]
    #     prediction[:, i] += MEANS[1][i+4]
    return prediction