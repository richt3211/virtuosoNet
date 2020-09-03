def feature_prediction_to_midi(prediction):
    prediction, _ = run_model_in_steps(batch_x, input_y, graph, note_locations, initial_z=initial_z, model=MODEL)
    trill_batch_x = torch.cat((batch_x, prediction), 2)
    trill_prediction, _ = run_model_in_steps(trill_batch_x, torch.zeros(1, num_notes, cons.num_trill_param), graph, note_locations, model=TRILL_MODEL)

    prediction = torch.cat((prediction, trill_prediction), 2)
    prediction = scale_model_prediction_to_original(prediction, MEANS, STDS)

    output_features = xml_matching.model_prediction_to_feature(prediction)
    output_features = xml_matching.add_note_location_to_features(output_features, note_locations)
    if return_features:
        return output_features

    output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1,
                                                           predicted=True)
    output_midi, midi_pedals = xml_matching.xml_notes_to_midi(output_xml, multi_instruments)
    piece_name = path_name.split('/')
    save_name = 'test_result/' + piece_name[-2] + '_by_' + args.modelCode + '_z' + str(z)

    perf_worm.plot_performance_worm(output_features, save_name + '.png')
    print(f'Saving midi performance to {save_name}')
    xml_matching.save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_name + '.mid',
                                               bool_pedal=args.boolPedal, disklavier=args.disklavier)

def run_model_in_steps(input, input_y, edges, note_locations, initial_z=False, model=MODEL):
    num_notes = input.shape[1]
    with torch.no_grad():  # no need to track history in validation
        model_eval = model.eval()
        total_output = []
        total_z = []
        measure_numbers = [x.measure for x in note_locations]
        slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=VALID_STEPS, overlap=False)
        # if edges is not None:
        #     edges = edges.to(DEVICE)

        for slice_idx in slice_indexes:
            batch_start, batch_end = slice_idx
            if edges is not None:
                batch_graph = edges[:, batch_start:batch_end, batch_start:batch_end].to(DEVICE)
            else:
                batch_graph = None
            
            batch_input = input[:, batch_start:batch_end, :].view(1,-1,model.input_size)
            batch_input_y = input_y[:, batch_start:batch_end, :].view(1,-1,model.output_size)
            temp_outputs, perf_mu, perf_var, _ = model_eval(batch_input, batch_input_y, batch_graph,
                                                                           note_locations=note_locations, start_index=batch_start, initial_z=initial_z)
            total_z.append((perf_mu, perf_var))
            total_output.append(temp_outputs)

        outputs = torch.cat(total_output, 1)
        return outputs, total_z