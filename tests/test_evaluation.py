import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluation.check_reported_results import percent, wilson
from evaluation.judge_request import build_judge_request, validate_judge_response
from inference.evaluation.protocol import REPO_ROOT, build_messages, generation_kwargs, load_labels
from inference.evaluation.run_inference import check_local_checkpoint, validate_mask


class EvaluationContractTests(unittest.TestCase):
    def test_ddi_system_role_and_no_reference(self):
        messages = build_messages('lesion.jpg')
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        self.assertIn('Do not invent findings', messages[0]['content'])
        self.assertNotIn('ground_truth', json.dumps(messages))
        with self.assertRaises(TypeError):
            build_messages('lesion.jpg', ground_truth='Melanoma')
        with self.assertRaises(ValueError):
            build_messages('lesion.jpg', labels=['Melanoma'])

    def test_classification_preserves_label_order_without_system(self):
        labels = load_labels(REPO_ROOT / 'prompts/labels_160case.json')
        self.assertEqual(len(labels), 23)
        messages = build_messages('lesion.jpg', 'classification', list(reversed(labels)))
        self.assertEqual([m['role'] for m in messages], ['user'])
        text = messages[0]['content'][1]['text']
        self.assertLess(text.index('Vascular Tumors'), text.index('Fungal infection - Tinea'))
        self.assertNotIn('{candidate_labels}', text)

    def test_duplicate_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'labels.json'
            p.write_text('["Eczema", "Eczema"]')
            with self.assertRaises(ValueError):
                load_labels(p)

    def test_greedy_does_not_pass_sampling_arguments(self):
        config = json.loads((REPO_ROOT / 'configs/evaluation.json').read_text())
        self.assertEqual(generation_kwargs(config)['temperature'], 0.7)
        self.assertEqual(generation_kwargs(config)['max_new_tokens'], 4096)
        config['do_sample'] = False
        self.assertNotIn('temperature', generation_kwargs(config))
        self.assertNotIn('top_p', generation_kwargs(config))

    def test_judge_tail_and_separate_reference(self):
        request = build_judge_request('Eczema', 'DROP_THIS' + 'A' * 20000)
        text = request['messages'][1]['content']
        self.assertNotIn('DROP_THIS', text)
        self.assertIn('[DDI reference diagnosis]\nEczema', text)
        self.assertEqual(request['temperature'], 0.0)
        self.assertEqual(request['extra_body'], {'enable_thinking': False})
        self.assertNotIn('seed', request)

    def test_judge_schema_rejects_false_string_and_ambiguous_success(self):
        valid = {'predicted_diagnosis': 'Eczema', 'correct': True, 'reason': 'Equivalent diagnosis.'}
        self.assertEqual(validate_judge_response(json.dumps(valid)), valid)
        for changed in [dict(valid, correct='false'), dict(valid, reason=''),
                        dict(valid, predicted_diagnosis=''), dict(valid, extra=1)]:
            with self.assertRaises(ValueError):
                validate_judge_response(json.dumps(changed))

    def test_mask_requires_binary_nonempty_matching_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'mask.json'
            for value in [[1, 0], [1, 0, 2], [0, 0, 0]]:
                p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    validate_mask(p, 3)
            p.write_text('[1, 0, 1]')
            self.assertEqual(validate_mask(p, 3), [1, 0, 1])

    def test_missing_and_lfs_pointer_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'model.safetensors'}}))
            with self.assertRaises(FileNotFoundError):
                check_local_checkpoint(root)
            (root / 'model.safetensors').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:example\n')
            with self.assertRaises(ValueError):
                check_local_checkpoint(root)

    def test_dry_run_without_model_or_optional_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'lesion.jpg'
            image.write_bytes(b'dry-run fixture; image decoding is not invoked')
            result = subprocess.run([
                sys.executable, '-S', '-m', 'inference.evaluation.run_inference',
                '--image', str(image), '--model-path', str(Path(tmp) / 'no_weights'),
                '--dry-run',
            ], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
            payload = json.loads(result.stdout)
            self.assertTrue(payload['dry_run'])
            self.assertEqual(payload['configuration']['min_pixels'], 3136)
            self.assertEqual(payload['messages'][0][0]['role'], 'system')
            self.assertNotIn('outputs', payload)

    def test_reporting_uses_counts_before_rounding(self):
        self.assertEqual(str(percent(81, 160)), '50.63')
        self.assertEqual(str(percent(90, 160)), '56.25')
        self.assertEqual(str(percent(9, 160)), '5.63')
        low, high = wilson(72, 158)
        self.assertLess(low, 100 * 72 / 158)
        self.assertGreater(high, 100 * 72 / 158)


if __name__ == '__main__':
    unittest.main()
