#!/usr/bin/env python3
"""Explicitly download/export official YOLOv8n COCO weights on a dev computer.

Run this in a dedicated Python virtual environment with:
  torch==2.5.1 torchvision==0.20.1 (CPU wheels)
  ultralytics==8.3.40 onnx==1.17.0 numpy==1.26.4

PyTorch and Ultralytics are export-time dependencies only. Jetson inference
uses the resulting ONNX file and ONNX Runtime. Ultralytics weights/software
are under AGPL-3.0 or an applicable commercial license; review your usage.
"""

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


WEIGHTS_URL = ('https://github.com/ultralytics/assets/releases/download/'
               'v8.3.0/yolov8n.pt')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='Destination .onnx file to copy to the Jetson')
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    manifest = output.with_suffix('.json')
    if output.suffix.lower() != '.onnx':
        parser.error('--output must end in .onnx')
    if output.exists() or manifest.exists():
        parser.error('Output or manifest already exists; choose a new path '
                     'or remove it explicitly')
    try:
        import onnx
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils.downloads import safe_download
    except ImportError as error:
        parser.exit(1, 'Missing export dependency: {}. Install the pinned packages '
                    'listed in this script in an isolated environment.\n'.format(error))
    if ultralytics.__version__ != '8.3.40':
        parser.error('Use ultralytics==8.3.40 for this fixed YOLOv8 export')
    print('Downloading official Ultralytics YOLOv8n COCO weights and exporting '
          'static 640px ONNX (bottle class 39).')
    with tempfile.TemporaryDirectory(prefix='snail-bottle-export-') as temporary:
        with working_directory(temporary):
            # Pin the official asset URL rather than resolve a latest release
            # or silently reuse a user's unrelated model cache.
            safe_download(url=WEIGHTS_URL, file='yolov8n.pt', min_bytes=1e5)
            model = YOLO('yolov8n.pt')
            weights_sha256 = sha256_file('yolov8n.pt')
            exported = Path(model.export(format='onnx', imgsz=640, batch=1,
                                         opset=12, simplify=False, dynamic=False,
                                         nms=False, half=False, device='cpu'))
            graph = onnx.load(str(exported))
            onnx.checker.check_model(graph)
            input_shape = [dim.dim_value for dim in
                           graph.graph.input[0].type.tensor_type.shape.dim]
            output_shape = [dim.dim_value for dim in
                            graph.graph.output[0].type.tensor_type.shape.dim]
            if input_shape != [1, 3, 640, 640] or output_shape != [1, 84, 8400]:
                raise RuntimeError('Unexpected exported shapes: {} -> {}'.format(
                    input_shape, output_shape))
            metadata = {entry.key: entry.value for entry in graph.metadata_props}
            names = ast.literal_eval(metadata['names'])
            if len(names) != 80 or names[39] != 'bottle':
                raise RuntimeError('Export does not contain COCO bottle class 39')
            output.parent.mkdir(parents=True, exist_ok=True)
            # Do not silently replace an existing prepared model.
            with output.open('xb') as destination, exported.open('rb') as source:
                shutil.copyfileobj(source, destination)
            provenance = {
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_weights_url': WEIGHTS_URL,
                'source_weights_sha256': weights_sha256,
                'model_filename': output.name,
                'model_sha256': sha256_file(output),
                'ultralytics_version': ultralytics.__version__,
                'onnx_version': onnx.__version__,
                'license': 'AGPL-3.0 or applicable Ultralytics commercial license',
                'license_url': 'https://www.ultralytics.com/license',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'export': {'imgsz': 640, 'batch': 1, 'opset': 12,
                           'simplify': False, 'dynamic': False, 'nms': False},
                'bottle_class': 39,
                'class_names': names,
                'limitation': 'Detects generic bottles, not their contents.',
            }
            with manifest.open('x', encoding='utf-8') as destination:
                json.dump(provenance, destination, indent=2)
                destination.write('\n')
    print('Prepared {} ({} bytes).'.format(output, output.stat().st_size))
    print('Provenance, license and SHA-256: {}'.format(manifest))
    print('This detects generic bottles; it does not distinguish water bottles.')


if __name__ == '__main__':
    main()
