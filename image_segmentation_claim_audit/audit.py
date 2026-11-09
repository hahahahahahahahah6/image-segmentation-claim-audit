"""Offline documentation and Python AST audit."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
from .phrases import IMAGESEGMENTATION, NOIMAGESEGMENTATION, GENERIC_NAMES

TV_NAMES = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn']
TV_SOURCES = ['torchvision.models.segmentation', 'torchvision.models.detection']
YOLO_MARKER = 'seg'
YOLO_CALLS = ['__call__', 'predict']
PIPELINE_TASK = 'image-segmentation'
PIPELINE_SOURCE = 'transformers'
CONFIG_MARKERS = ['mask_rcnn', 'panoptic', 'semantic']
DETECTRON_SOURCE = 'detectron2.engine'
SAM_NAME = 'SamPredictor'
SAM_SOURCE = 'segment_anything'
SAM_CALLS = ['predict']
CV_NAMES = ['grabCut', 'watershed']
CV_SOURCE = 'cv2'
SK_NAMES = ['slic', 'felzenszwalb', 'watershed', 'quickshift', 'chan_vese', 'morphological_chan_vese', 'find_boundaries']
SK_SOURCE = 'skimage.segmentation'
PAIR_CALLS = ['__call__']
EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env'}


def match_phrases(text, phrases):
    """Return longest-first matches and text with matching spans blanked."""
    text = text.lower()
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = text.find(phrase)
        while start >= 0:
            found.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
            start = text.find(phrase)
    return found, text


def prose(text):
    lines = []
    fenced = False
    for line in text.splitlines():
        if re.match(r'^\s*```', line):
            fenced = not fenced
        elif not fenced:
            lines.append(line.replace('`', ''))
    return '\n'.join(lines)


def claims(text):
    denial, remainder = match_phrases(prose(text), NOIMAGESEGMENTATION)
    positive, _ = match_phrases(remainder, IMAGESEGMENTATION)
    return positive, denial


def evidence(source):
    """Resolve imported call targets and module-local construction/call pairs."""
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    shadowed = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                imports[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            for a in n.names:
                imports[a.asname or a.name] = (n.module or '') + '.' + a.name

    def resolve(n):
        if isinstance(n, ast.Name):
            return imports.get(n.id, '') if n.id not in shadowed else ''
        if isinstance(n, ast.Attribute):
            if n.attr in shadowed:
                return ''
            base = resolve(n.value)
            return base + '.' + n.attr if base else ''
        return ''

    def arg(call, keyword):
        if call.args:
            return call.args[0]
        return next((k.value for k in call.keywords if k.arg == keyword), None)

    def string(node):
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else ''

    marker = any(any(m in n.value.lower() for m in CONFIG_MARKERS) for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str))
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    instances = {}
    for n in nodes:
        if not isinstance(n, (ast.Assign, ast.AnnAssign)) or not isinstance(n.value, ast.Call):
            continue
        c = n.value
        target = resolve(c.func)
        halves = None
        if any(target == (TV_SOURCES[1] if name == TV_NAMES[-1] else TV_SOURCES[0]) + '.' + name for name in TV_NAMES):
            halves = PAIR_CALLS
        elif target == 'ultralytics.YOLO' and YOLO_MARKER in string(arg(c, 'model')).lower():
            halves = YOLO_CALLS
        elif target == PIPELINE_SOURCE + '.pipeline' and string(arg(c, 'task')) == PIPELINE_TASK:
            halves = PAIR_CALLS
        elif target == DETECTRON_SOURCE + '.DefaultPredictor' and marker:
            halves = PAIR_CALLS
        elif target in (SAM_SOURCE + '.' + SAM_NAME, SAM_SOURCE + '.predictor.' + SAM_NAME):
            halves = SAM_CALLS
        if halves:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id not in shadowed:
                    instances[t.id] = halves
    for c in calls:
        f = c.func
        resolved = resolve(f)
        if resolved in [CV_SOURCE + '.' + x for x in CV_NAMES] + [SK_SOURCE + '.' + x for x in SK_NAMES]:
            return True
        if isinstance(f, ast.Attribute) and f.attr in GENERIC_NAMES and f.attr not in shadowed:
            return True
        if isinstance(f, ast.Name) and resolved and resolved.rsplit('.', 1)[-1] in GENERIC_NAMES:
            return True
        if isinstance(f, ast.Name) and '__call__' in instances.get(f.id, []):
            return True
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.attr in instances.get(f.value.id, []):
            return True
    return False


def audit(repo, ignore=(), decisions=None):
    repo = Path(repo)
    if not repo.is_dir():
        raise OSError('Repository is not a directory: ' + str(repo))
    positive = denial = test_positive = supported = False
    files = sorted(repo.rglob('*'))
    for path in files:
        rel = path.relative_to(repo)
        if any(part in EXCLUDED for part in rel.parts) or any((repo / Path(*rel.parts[:i]) / 'pyvenv.cfg').is_file() for i in range(1, len(rel.parts))):
            continue
        if not path.is_file():
            continue
        if any(rel.as_posix().startswith(prefix) for prefix in ignore):
            if decisions is not None:
                decisions.append('ignored ' + rel.as_posix())
            continue
        doc = rel.as_posix() == 'README.md' or (rel.parts[0] == 'docs' and path.suffix == '.md')
        test = 'tests' in rel.parts[:-1]
        if not (doc or test or path.suffix == '.py'):
            continue
        text = path.read_text(encoding='utf-8')
        if decisions is not None:
            decisions.append('scanned ' + rel.as_posix())
        if doc or test:
            yes, no = claims(text)
            if doc:
                positive |= bool(yes)
                denial |= bool(no)
            elif test:
                test_positive |= bool(yes)
        if path.suffix == '.py':
            try:
                supported |= evidence(text)
            except SyntaxError as exc:
                if decisions is not None:
                    decisions.append('invalid Python ' + rel.as_posix() + ': ' + str(exc))
    findings = []
    def finding(rule, severity, message):
        findings.append(dict(rule=rule, severity=severity, verdict='FAIL', message=message))
    if positive and not supported:
        finding('documented-image-segmentation-claim-false', 'HIGH', 'Documentation claims the checked capability without Python evidence.')
    elif test_positive and not supported:
        finding('documented-image-segmentation-claim-false', 'MEDIUM', 'Only test fixtures claim the checked capability; no Python evidence.')
    if denial and supported:
        finding('documented-no-image-segmentation-claim-false', 'HIGH', 'Documentation denies the checked capability but Python evidence exists.')
    return findings


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(prog='image-segmentation-claim-audit')
    parser.add_argument('--version', action='version', version='0.1.0')
    sub = parser.add_subparsers(dest='command', required=True)
    cmd = sub.add_parser('audit')
    cmd.add_argument('repo', nargs='?', default='.')
    cmd.add_argument('--json', action='store_true')
    cmd.add_argument('--verbose', action='store_true')
    cmd.add_argument('--ignore', action='append', default=[], metavar='PREFIX')
    cmd.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    args = parser.parse_args(argv)
    decisions = []
    try:
        findings = audit(args.repo, args.ignore, decisions)
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.verbose:
        print('Scope: README.md, docs/**/*.md, tests/, and Python sources in ' + str(Path(args.repo).resolve()), file=sys.stderr)
        print('\n'.join(decisions), file=sys.stderr)
    if args.json:
        print(json.dumps(findings))
    elif findings:
        for f in findings:
            print('{severity} {rule}: {message}'.format(**f))
    else:
        print('PASS: no findings')
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0
