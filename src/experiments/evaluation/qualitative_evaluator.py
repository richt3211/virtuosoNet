from src.models.model_run_job import ModelJobParams
from src.evaluation.qualitative import QualitativeEvaluator

def load_model_and_generate_performance(model, xml_file_path, midi_file_path, plot_file_path):
    params = ModelJobParams(is_dev=False)
    qualitative_evaluator = QualitativeEvaluator(params, model)

    qualitative_evaluator.generate_performance_for_file()