import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import Detect
from utils.datasets import LoadImages
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_version, colorstr,
                           file_size, print_args, url2file)
from utils.torch_utils import select_device


def export_formats():
    """
    Returns a DataFrame containing information about different export formats for YOLOv5 models.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Format', 'Argument', 'Suffix', 'GPU'].
            - 'Format': The name of the export format.
            - 'Argument': The argument to be passed when exporting the model.
            - 'Suffix': The file suffix for the exported model.
            - 'GPU': Indicates whether the export format supports GPU acceleration (True/False).
    """
    x = [
        ['PyTorch', '-', '.pt', True],
        ['TorchScript', 'torchscript', '.torchscript', True],
        ['ONNX', 'onnx', '.onnx', True],
        ['OpenVINO', 'openvino', '_openvino_model', False],
        ['TensorRT', 'engine', '.engine', True],
        ['CoreML', 'coreml', '.mlmodel', False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True],
        ['TensorFlow GraphDef', 'pb', '.pb', True],
        ['TensorFlow Lite', 'tflite', '.tflite', False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False],
        ['TensorFlow.js', 'tfjs', '_web_model', False],
    ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'GPU'])


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    """
    Export the YOLOv5 TorchScript model.

    Args:
        model (torch.nn.Module): The YOLOv5 model to be exported.
        im (torch.Tensor): The input image tensor.
        file (Path): The file path to save the exported TorchScript model.
        optimize (bool): Whether to optimize the exported model for mobile interpreter.
        prefix (str): The prefix string for logging.

    Returns:
        Path: The file path where the exported TorchScript model is saved.

    Raises:
        Exception: If there is an error during the export process.
    """
    try:
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript')

        ts = torch.jit.trace(model, im, strict=False)
        d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    """
    Export the model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): The input image tensor.
        file (Path): The file path to save the exported ONNX model.
        opset (int): The ONNX opset version.
        train (bool): Whether to export the model in training mode or evaluation mode.
        dynamic (bool): Whether to use dynamic axes for the exported ONNX model.
        simplify (bool): Whether to simplify the exported ONNX model using onnx-simplifier.
        prefix (str): The prefix string for logging messages.

    Returns:
        Path: The file path of the exported ONNX model.

    Raises:
        Exception: If there is an error during the export process.
    """
    try:
        check_requirements(('onnx',))
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(
            model,
            im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_openvino(model, im, file, prefix=colorstr('OpenVINO:')):
    """
    Export the YOLOv5 model to OpenVINO format.

    Args:
        model: The YOLOv5 model.
        im: The input image.
        file: The file to save the exported model.
        prefix: The prefix to use for logging messages (default: 'OpenVINO:').

    Returns:
        The path to the exported model.

    Raises:
        Exception: If the export fails.
    """
    try:
        check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie

        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        f = str(file).replace('.pt', '_openvino_model' + os.sep)

        cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f}"
        subprocess.check_output(cmd, shell=True)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_coreml(model, im, file, int8, half, prefix=colorstr('CoreML:')):
    """
    Export the given model to CoreML format using coremltools.

    Args:
        model: The model to be exported.
        im: The input image.
        file: The file path to save the exported CoreML model.
        int8: Flag indicating whether to use 8-bit quantization.
        half: Flag indicating whether to use 16-bit quantization.
        prefix: The prefix string for logging messages.

    Returns:
        ct_model: The exported CoreML model.
        f: The file path where the CoreML model is saved.
    """
    try:
        check_requirements(('coremltools',))
        import coremltools as ct

        LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')

        ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape, scale=1 / 255, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear') if half else (32, None)
        if bits < 32:
            if platform.system() == 'Darwin':  # quantization only supported on macOS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            else:
                print(f'{prefix} quantization only supported on macOS, skipping...')
        ct_model.save(f)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return ct_model, f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        return None, None


def export_engine(model, im, file, train, half, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    """
    Export the model to a TensorRT engine file.
    Refer to: https://developer.nvidia.com/tensorrt

    Args:
        model: The model to be exported.
        im: The input image.
        file: The output file path for the exported engine file.
        train: Whether the model is trained or not.
        half: Whether to use half-precision floating-point format.
        simplify: Whether to simplify the model.
        workspace: The maximum workspace size for TensorRT.
        verbose: Whether to enable verbose logging.
        prefix: The prefix for log messages.

    Returns:
        The path to the exported TensorRT engine file.

    Raises:
        AssertionError: If the export is running on CPU instead of GPU.
        AssertionError: If the ONNX file fails to export.
        RuntimeError: If the ONNX file fails to load.
        Exception: If there is an export failure.
    """
    try:
        import tensorrt as trt  # pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com

        if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, train, False, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
            export_onnx(model, im, file, 13, train, False, simplify)  # opset 13
        onnx = file.with_suffix('.onnx')

        LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        LOGGER.info(f'{prefix} Network Description:')
        for inp in inputs:
            LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 else 32} engine in {f}')
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_saved_model(model,
                       im,
                       file,
                       dynamic,
                       tf_nms=False,
                       agnostic_nms=False,
                       topk_per_class=100,
                       topk_all=100,
                       iou_thres=0.45,
                       conf_thres=0.25,
                       keras=False,
                       prefix=colorstr('TensorFlow SavedModel:')):
    """
    Export the YOLOv5 model as a TensorFlow SavedModel.

    Args:
        model (Model): YOLOv5 model.
        im (Tensor): Input image tensor.
        file (str): File path to save the exported model.
        dynamic (bool): Whether the model is dynamic or not.
        tf_nms (bool, optional): Whether to use TensorFlow NMS. Defaults to False.
        agnostic_nms (bool, optional): Whether to use agnostic NMS. Defaults to False.
        topk_per_class (int, optional): Number of top detections per class. Defaults to 100.
        topk_all (int, optional): Number of top detections across all classes. Defaults to 100.
        iou_thres (float, optional): IoU threshold for NMS. Defaults to 0.45.
        conf_thres (float, optional): Confidence threshold for NMS. Defaults to 0.25.
        keras (bool, optional): Whether to save the model in Keras format. Defaults to False.
        prefix (str, optional): Prefix for log messages. Defaults to 'TensorFlow SavedModel:'.

    Returns:
        Tuple[Model, str]: Tuple containing the exported Keras model and the file path where it is saved.
    """
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        from models.tf import TFDetect, TFModel

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = str(file).replace('.pt', '_saved_model')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW

        tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
        im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
        _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
        outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.trainable = False
        keras_model.summary()
        if keras:
            keras_model.save(f, save_format='tf')
        else:
            spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
            m = tf.function(lambda x: keras_model(x))  # full model
            m = m.get_concrete_function(spec)
            frozen_func = convert_variables_to_constants_v2(m)
            tfm = tf.Module()
            tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x)[0], [spec])
            tfm.__call__(im)
            tf.saved_model.save(tfm,
                                f,
                                options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
                                if check_version(tf.__version__, '2.6') else tf.saved_model.SaveOptions())
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return keras_model, f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        return None, None


def export_pb(keras_model, im, file, prefix=colorstr('TensorFlow GraphDef:')):
    """
    Export the Keras model as a TensorFlow GraphDef protobuf file.
    Refer to: https://github.com/leimao/Frozen_Graph_TensorFlow

    Args:
        keras_model (tf.keras.Model): The Keras model to be exported.
        im: The input image.
        file (str): The file path where the exported model will be saved.
        prefix (str): The prefix for log messages.

    Returns:
        str: The file path of the exported model.

    Raises:
        Exception: If there is an error during the export process.
    """
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
    """
    Export a Keras model to TensorFlow Lite format.

    Args:
        keras_model (tf.keras.Model): The Keras model to be exported.
        im (numpy.ndarray): The input image.
        file (str): The file path to save the exported TensorFlow Lite model.
        int8 (bool): Whether to use INT8 quantization.
        data (str): The path to the dataset.
        nms (bool): Whether to enable non-maximum suppression.
        agnostic_nms (bool): Whether to enable agnostic non-maximum suppression.
        prefix (str): The prefix for log messages.

    Returns:
        str: The file path of the exported TensorFlow Lite model.

    Raises:
        Exception: If there is an error during the export process.
    """
    try:
        import tensorflow as tf

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW
        f = str(file).replace('.pt', '-fp16.tflite')

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:
            from models.tf import representative_dataset_gen
            dataset = LoadImages(check_dataset(data)['train'], img_size=imgsz, auto=False)  # representative data
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True
            f = str(file).replace('.pt', '-int8.tflite')
        if nms or agnostic_nms:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

        tflite_model = converter.convert()
        open(f, "wb").write(tflite_model)
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_edgetpu(keras_model, im, file, prefix=colorstr('Edge TPU:')):
    """
    Export the Keras model to Edge TPU format using the Edge TPU compiler.

    Args:
        keras_model: The Keras model to be exported.
        im: The input image.
        file: The file path to save the exported model.
        prefix: The prefix for log messages (default: 'Edge TPU:').

    Returns:
        The file path of the exported Edge TPU model.

    Raises:
        AssertionError: If the export is not supported on the current platform.
        Exception: If the export fails.

    Note:
        This function requires the Edge TPU compiler to be installed on a Linux system.
        For more information on the Edge TPU compiler, see: https://coral.ai/docs/edgetpu/compiler/
    """
    try:
        cmd = 'edgetpu_compiler --version'
        help_url = 'https://coral.ai/docs/edgetpu/compiler/'
        assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
        if subprocess.run(cmd + ' >/dev/null', shell=True).returncode != 0:
            LOGGER.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
            sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
            for c in (
                    'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                    'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                    'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
                subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
        f = str(file).replace('.pt', '-int8_edgetpu.tflite')  # Edge TPU model
        f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model

        cmd = f"edgetpu_compiler -s -o {file.parent} {f_tfl}"
        subprocess.run(cmd, shell=True, check=True)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_tfjs(keras_model, im, file, prefix=colorstr('TensorFlow.js:')):
    """
    Export the Keras model to TensorFlow.js format.

    Args:
        keras_model (object): The Keras model to be exported.
        im (object): The input image.
        file (str): The file path to save the exported model.
        prefix (str, optional): The prefix for log messages. Defaults to colorstr('TensorFlow.js:').

    Returns:
        str: The path to the exported TensorFlow.js model.

    Raises:
        Exception: If there is an error during the export process.
    """
    try:
        check_requirements(('tensorflowjs',))
        import re

        import tensorflowjs as tfjs

        LOGGER.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
        f = str(file).replace('.pt', '_web_model')  # js dir
        f_pb = file.with_suffix('.pb')  # *.pb path
        f_json = f + '/model.json'  # *.json path

        cmd = f'tensorflowjs_converter --input_format=tf_frozen_model ' \
              f'--output_node_names="Identity,Identity_1,Identity_2,Identity_3" {f_pb} {f}'
        subprocess.run(cmd, shell=True)

        with open(f_json) as j:
            json = j.read()
        with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
            subst = re.sub(
                r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}}}', r'{"outputs": {"Identity": {"name": "Identity"}, '
                r'"Identity_1": {"name": "Identity_1"}, '
                r'"Identity_2": {"name": "Identity_2"}, '
                r'"Identity_3": {"name": "Identity_3"}}}', json)
            j.write(subst)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


@torch.no_grad()
def run(
        data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        int8=False,  # CoreML/TF INT8 quantization
        dynamic=False,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
        nms=False,  # TF: add NMS to model
        agnostic_nms=False,  # TF: add agnostic NMS to model
        topk_per_class=100,  # TF.js NMS: topk per class to keep
        topk_all=100,  # TF.js NMS: topk for all classes to keep
        iou_thres=0.45,  # TF.js NMS: IoU threshold
        conf_thres=0.25,  # TF.js NMS: confidence threshold
):
    if isinstance(weights, str):
        weights = weights.split()

    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    formats = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in formats]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {formats}'
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    shape = tuple(y[0].shape)  # model output shape
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * 10  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if jit:
        f[0] = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required before ONNX
        f[1] = export_engine(model, im, file, train, half, simplify, workspace, verbose)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2] = export_onnx(model, im, file, opset, train, dynamic, simplify)
    if xml:  # OpenVINO
        f[3] = export_openvino(model, im, file)
    if coreml:
        _, f[4] = export_coreml(model, im, file, int8, half)

    # TensorFlow Exports
    if any((saved_model, pb, tflite, edgetpu, tfjs)):
        if int8 or edgetpu:  # TFLite --int8 bug https://github.com/ultralytics/yolov5/issues/5707
            check_requirements(('flatbuffers==1.12',))  # required before `import tensorflow`
        assert not (tflite and tfjs), 'TFLite and TF.js models must be exported separately, please pass only one type.'
        model, f[5] = export_saved_model(model.cpu(),
                                         im,
                                         file,
                                         dynamic,
                                         tf_nms=nms or agnostic_nms or tfjs,
                                         agnostic_nms=agnostic_nms or tfjs,
                                         topk_per_class=topk_per_class,
                                         topk_all=topk_all,
                                         conf_thres=conf_thres,
                                         iou_thres=iou_thres)  # keras model
        if pb or tfjs:  # pb prerequisite to tfjs
            f[6] = export_pb(model, im, file)
        if tflite or edgetpu:
            f[7] = export_tflite(model, im, file, int8=int8 or edgetpu, data=data, nms=nms, agnostic_nms=agnostic_nms)
        if edgetpu:
            f[8] = export_edgetpu(model, im, file)
        if tfjs:
            f[9] = export_tfjs(model, im, file)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        LOGGER.info(f'\nExport complete ({time.time() - t:.2f}s)'
                    f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                    f"\nDetect:          python detect.py --weights {f[-1]}"
                    f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
                    f"\nValidate:        python val.py --weights {f[-1]}"
                    f"\nVisualize:       https://netron.app")
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    parser.add_argument('--include',
                        nargs='+',
                        default=['torchscript', 'onnx'],
                        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


def run_export(
        data, 
        weights, 
        height,  # Separated height
        width,   # Separated width
        batch_size, 
        device, 
        half, 
        inplace, 
        train, 
        optimize, 
        int8, 
        dynamic, 
        simplify, 
        opset, 
        verbose, 
        workspace, 
        nms, 
        agnostic_nms, 
        topk_per_class, 
        topk_all, 
        iou_thres, 
        conf_thres, 
        include
    ):
    # Combine height and width into a tuple as imgsz
    imgsz = (int(height), int(width))

    # Call the actual detection function
    return run(
        data=data, 
        weights=weights, 
        imgsz=imgsz, 
        batch_size=batch_size, 
        device=device, 
        half=half, 
        inplace=inplace, 
        train=train, 
        optimize=optimize, 
        int8=int8, 
        dynamic=dynamic, 
        simplify=simplify, 
        opset=opset, 
        verbose=verbose, 
        workspace=workspace, 
        nms=nms, 
        agnostic_nms=agnostic_nms, 
        topk_per_class=topk_per_class, 
        topk_all=topk_all, 
        iou_thres=iou_thres, 
        conf_thres=conf_thres, 
        include=include
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
