"""Construct one DDI judge request and validate its response; no network calls."""
import json
from inference.evaluation.protocol import REPO_ROOT, prompt_text


def build_judge_request(reference_diagnosis, model_output):
    if not isinstance(reference_diagnosis, str) or not reference_diagnosis.strip():
        raise ValueError('A nonempty reference diagnosis is required for scoring.')
    if not isinstance(model_output, str):
        raise ValueError('The generated response must be a string.')
    config = json.loads((REPO_ROOT / 'configs/ddi_judge.json').read_text())
    tail = model_output[-config['batch_evaluator_defaults']['model_output_tail_characters']:]
    user = prompt_text('ddi_judge_user.txt').format(
        ground_truth=reference_diagnosis, model_output=tail
    )
    return {
        key: config[key] for key in ['model', 'temperature', 'max_tokens', 'response_format', 'extra_body']
    } | {'messages': [
        {'role': 'system', 'content': prompt_text('ddi_judge_system.txt')},
        {'role': 'user', 'content': user},
    ]}


def validate_judge_response(content):
    result = json.loads(content)
    if not isinstance(result, dict) or set(result) != {'predicted_diagnosis', 'correct', 'reason'}:
        raise ValueError('Unexpected judge response fields.')
    if not isinstance(result['predicted_diagnosis'], str) or type(result['correct']) is not bool:
        raise ValueError('Invalid diagnosis or correctness type.')
    if not isinstance(result['reason'], str) or not result['reason'].strip():
        raise ValueError('A nonempty reason is required.')
    if result['correct'] and not result['predicted_diagnosis'].strip():
        raise ValueError('A correct response requires an identified final diagnosis.')
    return result
