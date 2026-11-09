"""Fixed vocabulary oracles and behavioral regression coverage."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from image_segmentation_claim_audit import audit as a
from image_segmentation_claim_audit.phrases import IMAGESEGMENTATION, NOIMAGESEGMENTATION, GENERIC_NAMES

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(IMAGESEGMENTATION, ['image segmentation', 'image-segmentation', 'imagesegmentation', 'image segment', 'photo segmentation', 'photo-segmentation', 'picture segmentation', 'segment images', 'segments images', 'segmenting images', 'segment photos', 'segments photos', 'segmenting photos', 'segment pictures', 'segments pictures', 'segmenting pictures', 'segment an image', 'segments an image', 'segmenting an image', 'segment the image', 'segments the image', 'segmenting the image', 'segment image into', 'segment images into', 'image segmenter', 'image segmenters', 'image segmentation tool', 'image segmentation library', 'image segmentation feature', 'image segmentation support', 'supports image segmentation', 'image segmentation capability', 'image segmentation pipeline', 'image segmentation algorithm', 'image segmentation software', 'image segmentation app', 'image segmentation model', 'automatic image segmentation', 'automated image segmentation', 'semantic segmentation', 'semantic-segmentation', 'instance segmentation', 'instance-segmentation', 'panoptic segmentation', 'panoptic-segmentation', 'segment objects', 'segments objects', 'segmenting objects', 'object segmentation', 'segment anything'])
        self.assertEqual(NOIMAGESEGMENTATION, ['does not support image segmentation', 'do not support image segmentation', "doesn't support image segmentation", "don't support image segmentation", 'no image segmentation', 'without image segmentation', 'never performs image segmentation', 'does not perform image segmentation', 'image segmentation not supported', 'image segmentation unsupported', 'no image segmentation support', 'lacks image segmentation', 'missing image segmentation', 'image segmentation disabled', 'no image segmentation capability', 'without image segmentation support', 'does not provide image segmentation', 'no image segmentation model', 'image segmentation not available', 'does not use image segmentation', 'no image segment', 'without image segment', 'does not support image segment', 'no image segmentation feature'])

    def test_generic_names(self):
        self.assertEqual(GENERIC_NAMES, ['segment_image', 'segment_images', 'segment_photo', 'segment_photos', 'segment_picture', 'segment_pictures', 'segment_objects', 'segment_object', 'semantic_segmentation', 'instance_segmentation', 'panoptic_segmentation', 'segment_anything', 'run_segmentation', 'apply_segmentation', 'perform_segmentation', 'create_segmentation', 'generate_segmentation', 'segment_foreground', 'segment_background', 'get_segmentation_mask', 'predict_segmentation', 'segment_regions', 'segment_pixels', 'image_segmentation'])

    def test_route_lists(self):
        self.assertEqual(a.TV_NAMES, ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn'])
        self.assertEqual(a.TV_SOURCES, ['torchvision.models.segmentation', 'torchvision.models.detection'])
        self.assertEqual(a.YOLO_MARKER, 'seg')
        self.assertEqual(a.YOLO_CALLS, ['__call__', 'predict'])
        self.assertEqual(a.PIPELINE_TASK, 'image-segmentation')
        self.assertEqual(a.PIPELINE_SOURCE, 'transformers')
        self.assertEqual(a.CONFIG_MARKERS, ['mask_rcnn', 'panoptic', 'semantic'])
        self.assertEqual(a.DETECTRON_SOURCE, 'detectron2.engine')
        self.assertEqual(a.SAM_NAME, 'SamPredictor')
        self.assertEqual(a.SAM_SOURCE, 'segment_anything')
        self.assertEqual(a.SAM_CALLS, ['predict'])
        self.assertEqual(a.CV_NAMES, ['grabCut', 'watershed'])
        self.assertEqual(a.CV_SOURCE, 'cv2')
        self.assertEqual(a.SK_NAMES, ['slic', 'felzenszwalb', 'watershed', 'quickshift', 'chan_vese', 'morphological_chan_vese', 'find_boundaries'])
        self.assertEqual(a.SK_SOURCE, 'skimage.segmentation')
        self.assertEqual(a.PAIR_CALLS, ['__call__'])

    def test_span_removal(self):
        self.assertEqual(a.claims('does not support image segmentation')[0], [])
        for phrase in NOIMAGESEGMENTATION:
            with self.subTest(phrase=phrase):
                self.assertEqual(a.claims(phrase)[0], [])

    def test_longest_first(self):
        self.assertEqual(a.claims('image segmentation model')[0], ['image segmentation model'])
        self.assertEqual(a.claims('segment images into training and test sets')[0], ['segment images into'])

    def test_collisions(self):
        self.assertTrue(a.claims('we segment images into training and test sets')[0])
        self.assertTrue(a.claims('we segment objects in the scene graph')[0])

    def test_fenced_code_excluded(self):
        self.assertEqual(a.claims('```text\nimage segmentation\n```')[0], [])

    def test_inline_code_counts(self):
        self.assertTrue(a.claims('`image segmentation`')[0])

    def run_cli(self, root, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = a.main(['audit', str(root), *args])
        return result, out.getvalue()

    def test_cli_fixtures(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            readme = root / 'README.md'
            readme.write_text('image segmentation')
            code, out = self.run_cli(root)
            self.assertEqual(code, 2)
            self.assertIn('HIGH documented-image-segmentation-claim-false', out)
            readme.write_text('IMAGE SEGMENTATION')
            self.assertEqual(self.run_cli(root)[0], 2)
            (root / 'app.py').write_text('import cv2\ncv2.grabCut(img)')
            self.assertEqual(self.run_cli(root), (0, 'PASS: no findings\n'))
            readme.write_text('does not support image segmentation')
            code, out = self.run_cli(root, '--json')
            self.assertEqual(code, 2)
            finding = json.loads(out)[0]
            self.assertEqual(finding['rule'], 'documented-no-image-segmentation-claim-false')
            self.assertEqual(finding['severity'], 'HIGH')
            self.assertEqual(finding['verdict'], 'FAIL')
            self.assertTrue(finding['message'])
            (root / 'app.py').unlink()
            self.assertEqual(self.run_cli(root)[0], 0)
            readme.write_text('ordinary documentation')
            self.assertEqual(self.run_cli(root)[0], 0)
            (root / 'tests').mkdir()
            (root / 'tests/fixture.py').write_text('"image segmentation"')
            code, out = self.run_cli(root)
            self.assertEqual(code, 0)
            self.assertIn('MEDIUM', out)
            self.assertEqual(self.run_cli(root, '--fail-on', 'medium')[0], 2)

    def test_cli_ignore_before_positional(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'docs').mkdir()
            (root / 'docs/claim.md').write_text('image segmentation')
            self.assertEqual(self.run_cli(root)[0], 2)
            self.assertEqual(self.run_cli(root, '--ignore', 'docs/')[0], 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(a.main(['audit', '--ignore', 'docs/', str(root)]), 0)

    def test_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exc:
            a.main(['--version'])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), '0.1.0')

    def test_readme_self_audit(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(a.claims((root / 'README.md').read_text()), ([], []))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'README.md').write_text((root / 'README.md').read_text())
            self.assertEqual(self.run_cli(d)[0], 0)
        findings = a.audit(root)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['severity'], 'MEDIUM')

    def test_split_modules(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'README.md').write_text('image segmentation')
            (root / 'one.py').write_text('from transformers import pipeline\np = pipeline("image-segmentation")')
            (root / 'two.py').write_text('p(img)')
            self.assertEqual(a.audit(root)[0]['severity'], 'HIGH')


def add_case(name, source, expected):
    def test(self):
        self.assertEqual(a.evidence(source), expected, source)
    setattr(AuditTests, 'test_' + name, test)

# Each generated method is separately counted and reported by unittest.
for name in ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn']:
    module = 'torchvision.models.' + ('detection' if name.startswith('maskrcnn') else 'segmentation')
    for label, imp, ctor in [('bare', 'from ' + module + ' import ' + name, name), ('alias', 'from ' + module + ' import ' + name + ' as C', 'C'), ('attribute', 'import ' + module + ' as m', 'm.' + name)]:
        add_case('tv_' + name + '_' + label, imp + '\np = ' + ctor + '()\np(img)', True)
    add_case('tv_no_import_' + name, 'p = ' + name + '()\np(img)', False)
    add_case('tv_wrong_base_' + name, 'p = other.' + name + '()\np(img)', False)
    add_case('tv_construct_only_' + name, 'from ' + module + ' import ' + name + '\np = ' + name + '()', False)
for i, source in enumerate([
    'import torchvision.models.segmentation\np = torchvision.models.segmentation.fcn_resnet50()\np(img)',
    'from torchvision.models import segmentation\np = segmentation.fcn_resnet50()\np(img)',
    'from torchvision import models\np = models.segmentation.fcn_resnet50()\np(img)',
    'import torchvision.models.detection\np = torchvision.models.detection.maskrcnn_resnet50_fpn()\np(img)',
]):
    add_case('tv_module_forms_' + str(i), source, True)

pairs = [
    ('yolo', 'ultralytics', 'YOLO', '"yolo11n-seg.pt"', 'p(img)'),
    ('pipeline', 'transformers', 'pipeline', '"image-segmentation"', 'p(img)'),
    ('detectron', 'detectron2.engine', 'DefaultPredictor', 'cfg', 'p(img)'),
    ('sam', 'segment_anything', 'SamPredictor', 'model', 'p.predict(img)'),
]
for label, module, name, args, half in pairs:
    marker = '\nconfig = "mask_rcnn"' if label == 'detectron' else ''
    for form, imp, ctor in [('bare', 'from ' + module + ' import ' + name, name), ('alias', 'from ' + module + ' import ' + name + ' as C', 'C'), ('attribute', 'import ' + module + ' as lib', 'lib.' + name)]:
        add_case(label + '_' + form, imp + marker + '\np = ' + ctor + '(' + args + ')\n' + half, True)
    add_case(label + '_construct_only', 'from ' + module + ' import ' + name + marker + '\np = ' + name + '(' + args + ')', False)
    add_case(label + '_unimported', marker + '\np = ' + name + '(' + args + ')\n' + half, False)
    add_case(label + '_call_only', 'from ' + module + ' import ' + name + marker + '\n' + half, False)
    for kind in ['def', 'class']:
        definition = ('def ' + name + '(*args):\n    pass' if kind == 'def' else 'class ' + name + ':\n    pass')
        add_case(label + '_shadow_' + kind, 'from ' + module + ' import ' + name + marker + '\n' + definition + '\np = ' + name + '(' + args + ')\n' + half, False)

for i, source in enumerate([
    'from ultralytics import YOLO\np=YOLO(model="YOLO11N-SEG.pt")\np.predict(img)',
    'from transformers import pipeline\np=pipeline(task="image-segmentation")\np(img)',
    'from segment_anything.predictor import SamPredictor\np=SamPredictor(model)\np.predict(img)',
    'from detectron2.engine import DefaultPredictor\nc="panoptic"\np=DefaultPredictor(cfg)\np(img)',
    'from detectron2.engine import DefaultPredictor\nc="SEMANTIC"\np=DefaultPredictor(cfg)\np(img)',
]):
    add_case('pair_extra_' + str(i), source, True)
for index, model in enumerate(['"yolov8n.pt"', 'path', 'None']):
    add_case('yolo_bad_model_' + str(index), 'from ultralytics import YOLO\np=YOLO(' + model + ')\np(img)', False)
for method in ['train', 'val', 'export', 'track']:
    add_case('yolo_wrong_half_' + method, 'from ultralytics import YOLO\np=YOLO("a-seg.pt")\np.' + method + '(img)', False)
for i, task in enumerate(['"object-detection"', '"image-classification"', '"IMAGE-SEGMENTATION"', 'task', 'model="some-model"']):
    add_case('pipeline_bad_task_' + str(i), 'from transformers import pipeline\np=pipeline(' + task + ')\np(img)', False)
for i, marker in enumerate(['', '\nconfig="faster_rcnn"']):
    add_case('detectron_bad_config_' + str(i), 'from detectron2.engine import DefaultPredictor' + marker + '\np=DefaultPredictor(cfg)\np(img)', False)
add_case('sam_set_image_only', 'from segment_anything import SamPredictor\np=SamPredictor(model)\np.set_image(img)', False)

for module, names in [('cv2', ['grabCut', 'watershed']), ('skimage.segmentation', ['slic', 'felzenszwalb', 'watershed', 'quickshift', 'chan_vese', 'morphological_chan_vese', 'find_boundaries'])]:
    for name in names:
        label = module.replace('.', '_') + '_' + name
        for form, source in [('attribute', 'import ' + module + '\n' + module + '.' + name + '(img)'), ('alias', 'import ' + module + ' as m\nm.' + name + '(img)'), ('bare', 'from ' + module + ' import ' + name + '\n' + name + '(img)'), ('bare_alias', 'from ' + module + ' import ' + name + ' as f\nf(img)')]:
            add_case(label + '_' + form, source, True)
        add_case(label + '_wrong_base', 'other.' + name + '(img)', False)
        add_case(label + '_no_import', name + '(img)', False)
        for kind in ['def', 'class']:
            definition = 'def ' + name + '(*args):\n    pass' if kind == 'def' else 'class ' + name + ':\n    pass'
            add_case(label + '_shadow_' + kind, 'from ' + module + ' import ' + name + '\n' + definition + '\n' + name + '(img)', False)
add_case('sk_from_module', 'from skimage import segmentation\nsegmentation.slic(img)', True)
for name in GENERIC_NAMES:
    for label, source in [('attribute', 'x.' + name + '(img)'), ('bare', 'from custom import ' + name + '\n' + name + '(img)'), ('alias', 'from custom import ' + name + ' as f\nf(img)')]:
        add_case('generic_' + name + '_' + label, source, True)
    add_case('generic_unbound_' + name, name, False)
    add_case('generic_unimported_' + name, name + '(img)', False)
    add_case('generic_shadow_' + name, 'from custom import ' + name + '\ndef ' + name + '(*args):\n    pass\n' + name + '(img)\nx.' + name + '(img)', False)
for name, module in [('deeplabv3_resnet50', 'torchvision.models.segmentation'), ('maskrcnn_resnet50_fpn', 'torchvision.models.detection')]:
    for kind in ['def', 'class']:
        definition = 'def ' + name + '(*args):\n    pass' if kind == 'def' else 'class ' + name + ':\n    pass'
        add_case('tv_shadow_' + name + kind, 'from ' + module + ' import ' + name + '\n' + definition + '\np=' + name + '()\np(img)', False)

rejections = {
    'bare_imports': 'import torchvision, ultralytics, transformers, detectron2, segment_anything, cv2, skimage',
    'strings': '"segment segmentation"\n# segment_image(img)\ns="image segmentation"',
    'unbound': 'from cv2 import grabCut\ngrabCut',
    'subprocess': 'import subprocess\nsubprocess.run(["segment", "img.png"])',
    'rembg': 'from rembg import remove\nremove(img)',
    'mediapipe': 'import mediapipe as mp\np=mp.solutions.selfie_segmentation.SelfieSegmentation()\np.process(img)',
    'crop': 'img.crop(box)', 'resize': 'img.resize(size)',
    'rotation': 'img.rotate(90)\nimg.transpose(1)',
    'inpaint': 'import cv2\ncv2.inpaint(img,mask,3,0)',
    'super_resolution': 'from cv2 import dnn_superres\np=dnn_superres.DnnSuperResImpl_create()\np.upsample(img)',
    'colorization': 'model.colorize(img)', 'classification': 'model.classify(img)',
    'captioning': 'model.caption(img)', 'generation': 'model.generate(img)',
    'threshold': 'import cv2\ncv2.threshold(img,0,255,0)\ncv2.adaptiveThreshold(img)',
    'contours': 'import cv2\ncv2.findContours(img)\ncv2.HoughCircles(img)',
    'kmeans': 'import cv2\ncv2.kmeans(img)',
    'cascade': 'import cv2\np=cv2.CascadeClassifier()\np.detectMultiScale(img)',
    'darknet': 'import cv2\np=cv2.dnn.readNetFromDarknet(cfg)\np.forward(img)',
}
for name in ['fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'ssd300_vgg16', 'fcos_resnet50_fpn']:
    rejections[name] = 'from torchvision.models.detection import ' + name + '\np=' + name + '()\np(img)'
for name, source in rejections.items():
    add_case('reject_' + name, source, False)

if __name__ == '__main__':
    unittest.main()
