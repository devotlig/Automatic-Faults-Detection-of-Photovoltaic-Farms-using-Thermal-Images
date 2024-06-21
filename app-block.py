import gradio as gr
import os
import sys
from pathlib import Path
from detection import run_detection
from train import run_training
from val import run_validation
from difference import run_diff_detection
from export import run_export


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


with gr.Blocks() as iface:
    gr.Markdown("# YOLOv5 Interface")
    with gr.Tabs():
        with gr.Tab("Image Detection"):
            gr.Markdown("## Image Detection \nThis tab is dedicated to configuring and running image detection processes. Adjust the settings below to fit your specific needs.")

            with gr.Column():
                # Grouping file inputs and source details
                gr.Markdown("### File Inputs")
                with gr.Row():
                    weights = gr.Textbox(value=os.path.join(ROOT, 'best-solar.pt'), label='Weights', placeholder='Enter model paths separated by space', info='model path(s)')
                    source = gr.Textbox(label='Source', value=os.path.join(ROOT, 'test_folder'), info='file/dir/URL/glob, 0 for webcam', placeholder='Enter source path')
                    data = gr.Textbox(label='Data', value=os.path.join(ROOT, 'data.yaml'), info='(optional) dataset.yaml path', placeholder='Enter dataset.yaml path')

                # Grouping image specifications
                gr.Markdown("### Image Specifications")
                with gr.Row():
                    image_height = gr.Number(label='Image Height', value=640, info='Inference image size height')
                    image_width = gr.Number(label='Image Width', value=640, info='Inference image size width')
                    device = gr.Radio(choices=['cpu', '0', '1', '2', '3'], label='Device', value='cpu', info='CUDA device, i.e. 0 or 0,1,2,3 or CPU')

                # Grouping detection settings
                gr.Markdown("### Detection Settings")
                with gr.Row():
                    confidence_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.25, label='Confidence Threshold', info='Confidence threshold (0-1)')
                    nms_iou_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.45, label='NMS IOU Threshold', info='NMS IOU threshold (0-1)')
                    max_detections = gr.Slider(minimum=1, maximum=1000, step=1, value=1000, label='Max Detections per Image', info='Maximum detections per image')
                    classes = gr.Textbox(value=None, label='Classes', placeholder='Enter class numbers separated by space, example: 0 1 2', info='Filter by classes')
                    class_agnostic_nms = gr.Checkbox(value=False, label='Class-agnostic NMS', info='Class-agnostic NMS')

                # Grouping visual and operational settings
                gr.Markdown("### Visual and Operational Settings")
                with gr.Row():
                    view_image = gr.Checkbox(value=False, label='View Image', info='Show results')
                    save_text = gr.Checkbox(value=False, label='Save Text', info='Save results to *.txt')
                    save_confidence = gr.Checkbox(value=False, label='Save Confidence', info='Save confidences in --save-txt labels')
                    save_crop = gr.Checkbox(value=False, label='Save Crop', info='Save cropped prediction boxes')
                    no_save = gr.Checkbox(value=False, label='No Save', info='Do not save images/videos')
                    augment = gr.Checkbox(value=False, label='Augment', info='Augmented inference')
                    visualize = gr.Checkbox(value=False, label='Visualize', info='Visualize features')
                    update = gr.Checkbox(value=False, label='Update', info='Update all models')

                # Grouping project and additional settings
                gr.Markdown("### Project and Additional Settings")
                with gr.Row():
                    project = gr.Textbox(value=os.path.join(ROOT, 'detect_results'), label='Project', info='Save results to project/name', placeholder='Enter project directory')
                    name = gr.Textbox(value='exp', label='Name', info='Save results to project/name', placeholder='Enter experiment name')
                    exist_ok = gr.Checkbox(value=False, label='Exist OK', info='Existing project/name ok, do not increment')
                    line_thickness = gr.Slider(minimum=1, maximum=10, step=1, value=3, label='Line Thickness', info='Bounding box thickness (pixels)')
                    hide_labels = gr.Checkbox(value=False, label='Hide Labels', info='Hide labels on images')
                    hide_confidence = gr.Checkbox(value=False, label='Hide Confidence', info='Hide confidence on images')
                    use_fp16 = gr.Checkbox(value=False, label='Use FP16 Half-Precision Inference', info='Use FP16 half-precision inference')
                    use_opencv = gr.Checkbox(value=False, label='Use OpenCV DNN for ONNX Inference', info='Use OpenCV DNN for ONNX inference')

            with gr.Column():
                # Submit Button and Outputs
                gr.Markdown("### Detection Output")
                with gr.Row():
                    saved_images = gr.File(label='Saved Images')
                    saved_text_files = gr.File(label='Saved Text Files')

            with gr.Column():
                # Submit Button
                submit_button_detection = gr.Button("Run Detection")
                submit_button_detection.click(
                    run_detection,
                    inputs=[weights, source, data, image_height, image_width, confidence_threshold, nms_iou_threshold, max_detections, device, view_image, save_text,
                            save_confidence, save_crop, no_save, classes, class_agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness,
                            hide_labels, hide_confidence, use_fp16, use_opencv],
                    outputs=[saved_images, saved_text_files]
                )

        with gr.Tab("Image Training"):
            gr.Markdown("## Image Training \nConfigure and initiate the training of your model. Set parameters such as epochs, batch size, and more below.")

            with gr.Column():
                # Grouping model setup configurations
                gr.Markdown("### Model Setup Configurations")
                with gr.Row():
                    weights = gr.Textbox(label='Weights', value=os.path.join(ROOT, 'yolov5s.pt'), info='Initial weights path', placeholder='Enter single model path')
                    model_config = gr.Textbox(label='Model Config', value='', info='model.yaml path', placeholder='Enter model.yaml path')
                    data_config = gr.Textbox(label='Data Config', value=os.path.join(ROOT, 'data', 'coco128.yaml'), info='dataset.yaml path', placeholder='Enter dataset.yaml path')
                    hyperparameters = gr.Textbox(label='Hyperparameters', value=os.path.join(ROOT, 'data', 'hyps', 'hyp.scratch-low.yaml'), info='Hyperparameters path', placeholder='Enter hyperparameters path')

                # Grouping training execution parameters
                gr.Markdown("### Training Execution Parameters")
                with gr.Row():
                    epochs = gr.Number(label='Epochs', value=300, info='Number of epochs to train')
                    batch_size = gr.Number(label='Batch Size', value=16, info='Total batch size for all GPUs, -1 for autobatch')
                    image_size = gr.Number(label='Image Size', value=640, info='train, val image size (pixels)')
                    image_weights = gr.Checkbox(label='Image Weights', value=False, info='use weighted image selection for training')
                    device = gr.Radio(choices=['cpu', '0', '1', '2', '3'], label='Device', value='cpu', info='CUDA device, i.e. 0, 1, 2, 3 or CPU')

                # Grouping training modifiers and features
                gr.Markdown("### Training Modifiers and Features")
                with gr.Row():
                    optimizer = gr.Radio(label='Optimizer', choices=['SGD', 'Adam', 'AdamW'], value='SGD', info='Optimizer selection')
                    sync_batchnorm = gr.Checkbox(label='Sync BatchNorm', value=False, info='Use SyncBatchNorm, only available in DDP mode')
                    patience = gr.Number(label='Patience', value=100, info='EarlyStopping patience (epochs without improvement)')
                    workers = gr.Number(label='Workers', value=8, info='Max dataloader workers (per RANK in DDP mode)')
                    multi_scale = gr.Checkbox(label='Multi-Scale', value=False, info='Vary img-size +/- 50%')
                    single_class = gr.Checkbox(label='Single Class', value=False, info='Train multi-class data as single-class')
                    label_smoothing = gr.Number(label='Label Smoothing', value=0.0, info='Label smoothing epsilon')
                    cosine_lr = gr.Checkbox(label='Cosine LR Scheduler', value=False, info='Cosine LR scheduler')

                # Grouping save and resume options
                gr.Markdown("### Save and Resume Options")
                with gr.Row():
                    project = gr.Textbox(label='Project', value=os.path.join(ROOT, 'runs', 'train'), info='Save to project/name', placeholder='Enter project directory')
                    name = gr.Textbox(label='Name', value='exp', info='Save to project/name', placeholder='Enter experiment name')
                    save_period = gr.Number(label='Save Period', value=-1, info='Save checkpoint every x epochs (disabled if < 1)')
                    resume_training = gr.Checkbox(label='Resume Training', value=False, info='Resume most recent training')
                    no_save = gr.Checkbox(label='No Save', value=False, info='Only save final checkpoint')
                    exist_ok = gr.Checkbox(label='Exist OK', value=False, info='Existing project/name ok, do not increment')

                # Grouping additional settings and experimental features
                gr.Markdown("### Additional Settings and Experimental Features")
                with gr.Row():
                    rectangular_training = gr.Checkbox(label='Rectangular Training', value=False, info='Rectangular training')
                    no_validation = gr.Checkbox(label='No Validation', value=False, info='Only validate final epoch')
                    no_autoanchor = gr.Checkbox(label='No AutoAnchor', value=False, info='Disable AutoAnchor')
                    evolve = gr.Number(label='Evolve', value=300, info='Evolve hyperparameters for x generations')
                    quad_dataloader = gr.Checkbox(label='Quad DataLoader', value=False, info='Quad dataloader')
                    freeze_layers = gr.Textbox(value='0', label='Freeze Layers', placeholder='Enter layer numbers separated by space', info='Freeze layers: backbone=10, first3=0 1 2')

                # Grouping cloud and data management settings
                gr.Markdown("### Cloud and Data Management Settings")
                with gr.Row():
                    bucket = gr.Textbox(label='Bucket', value='', info='gsutil bucket', placeholder='gs://bucket')
                    cache_images = gr.Radio(label='Cache Images', choices=['ram', 'disk'], value='ram', info='--cache images in "ram" (default) or "disk"')
                    upload_dataset = gr.Checkbox(label='Upload Dataset', value=False, info='W&B: Upload data, "val" option')
                    bbox_interval = gr.Number(label='BBox Interval', value=-1, info='W&B: Set bounding-box image logging interval')
                    artifact_alias = gr.Textbox(label='Artifact Alias', value='latest', info='W&B: Version of dataset artifact to use', placeholder='Enter W&B artifact alias')
                    entity = gr.Textbox(label='Entity', value='', info='W&B: Entity', placeholder='Enter W&B entity')
                    local_rank = gr.Number(label='Local Rank', value=-1, info='DDP parameter, do not modify')

            with gr.Column():
                # Outputs
                gr.Markdown("### Training Output")
                with gr.Row():
                    training_output = gr.Textbox()

            with gr.Column():
                # Submit Button
                submit_button_training = gr.Button("Run Training")
                submit_button_training.click(
                    run_training,
                    inputs=[
                        weights, model_config, data_config, hyperparameters, epochs, batch_size, image_size, rectangular_training, resume_training,
                        no_save, no_validation, no_autoanchor, evolve, bucket, cache_images, image_weights, device, multi_scale, single_class,
                        optimizer, sync_batchnorm, workers, project, name, exist_ok, quad_dataloader, cosine_lr, label_smoothing, patience,
                        freeze_layers, save_period, local_rank, entity, upload_dataset, bbox_interval, artifact_alias
                    ],
                    outputs=training_output
                )

        with gr.Tab("Image Validation"):
            gr.Markdown("## Image Validation \nValidate your models to ensure accuracy and reliability. Select your model and adjust the validation settings.")

            with gr.Column():
                # Grouping file and model settings
                gr.Markdown("### File and Model Settings")
                with gr.Row():
                    data_yaml_path = gr.Textbox(label="Data YAML Path", value=os.path.join(ROOT, 'data', 'coco128.yaml'), info='dataset.yaml path', placeholder='Enter dataset.yaml path')
                    weights = gr.Textbox(value=os.path.join(ROOT, 'yolov5s.pt'), label='Weights', placeholder='Enter model paths separated by space', info='model.pt path(s)')

                # Grouping inference settings
                gr.Markdown("### Inference Settings")
                with gr.Row():
                    batch_size = gr.Number(label="Batch Size", value=32, info='Batch size')
                    image_size = gr.Number(label="Image Size (pixels)", value=640, info='inference size (pixels)')
                    device = gr.Radio(choices=['cpu', '0', '1', '2', '3'], label='Device', value='cpu', info='CUDA device, i.e. 0 or 0,1,2,3 or CPU')
                    max_dataloader_workers = gr.Number(label="Max Dataloader Workers", value=8, info='Max dataloader workers (per RANK in DDP mode)')

                # Grouping threshold and processing options
                gr.Markdown("### Threshold and Processing Options")
                with gr.Row():
                    confidence_threshold = gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, step=0.01, value=0.001, info='Confidence threshold (0-1)')
                    nms_iou_threshold = gr.Slider(label="NMS IoU Threshold", minimum=0, maximum=1, step=0.01, value=0.6, info='NMS IoU threshold (0-1)')
                    task = gr.Radio(label="Task", choices=['train', 'val', 'test', 'speed', 'study'], value='val', info='Task to run')
                    verbose = gr.Checkbox(label="Verbose", value=False, info='Report mAP by class')

                # Grouping output options
                gr.Markdown("### Output Options")
                with gr.Row():
                    save_to_txt = gr.Checkbox(label="Save to TXT", value=False, info='Save results to *.txt')
                    save_hybrid = gr.Checkbox(label="Save Hybrid", value=False, info='Save label+prediction hybrid results to *.txt')
                    save_confidences = gr.Checkbox(label="Save Confidences", value=False, info='Save confidences in --save-txt labels')
                    save_json = gr.Checkbox(label="Save JSON", value=False, info='Save a COCO-JSON results file')

                # Grouping project and additional settings
                gr.Markdown("### Project and Additional Settings")
                with gr.Row():
                    project_directory = gr.Textbox(label='Project Directory', value=os.path.join(ROOT, 'runs', 'val'), info='Save to project/name', placeholder='Enter project directory')
                    experiment_name = gr.Textbox(label="Experiment Name", value='exp', info='Save to project/name', placeholder='Enter experiment name')
                    exist_ok = gr.Checkbox(label="Exist OK", value=False, info='Existing project/name ok, do not increment')
                    use_fp16 = gr.Checkbox(label="Use FP16 Half-Precision", value=False, info='Use FP16 half-precision inference')
                    use_dnn = gr.Checkbox(label="Use DNN", value=False, info='Use OpenCV DNN for ONNX inference')

            with gr.Column():
                # Define output component
                gr.Markdown("### Validation Output")
                with gr.Row():
                    validation_output = gr.Textbox()

            with gr.Column():
                # Submit Button
                submit_button_validation = gr.Button("Run Validation")
                submit_button_validation.click(
                    run_validation,
                    inputs=[
                        data_yaml_path, weights, batch_size, image_size, confidence_threshold, nms_iou_threshold, task, device, max_dataloader_workers, single_class,
                        augment, verbose, save_to_txt, save_hybrid, save_confidences, save_json, project_directory, experiment_name, exist_ok, use_fp16, use_dnn
                    ],
                    outputs=validation_output
                )

        with gr.Tab("Image Diff Detection"):
            gr.Markdown("## Image Diff Detection \nUse this tab for differential detection, which can help in identifying changes over time or between two images.")

            with gr.Column():
                # Grouping model and source settings
                gr.Markdown("### Model and Source Settings")
                with gr.Row():
                    weights = gr.Textbox(value=os.path.join(ROOT, 'best-solar.pt'), label='Weights', placeholder='Enter model paths separated by space', info='model path(s)')
                    source = gr.Textbox(label='Source', value=os.path.join(ROOT, 'test_folder'), info='file/dir/URL/glob, 0 for webcam', placeholder='Enter source path')
                    data = gr.Textbox(label='Data', value=os.path.join(ROOT, 'data.yaml'), info='(optional) dataset.yaml path', placeholder='Enter dataset.yaml path')

                # Grouping image processing parameters
                gr.Markdown("### Image Processing Parameters")
                with gr.Row():
                    image_height = gr.Number(label='Image Height', value=640, info='Inference image size height')
                    image_width = gr.Number(label='Image Width', value=640, info='Inference image size width')
                    device = gr.Radio(choices=['cpu', '0', '1', '2', '3'], label='Device', value='cpu', info='CUDA device, i.e. 0 or 0,1,2,3 or CPU')
                
                # Grouping detection settings
                gr.Markdown("### Detection Settings")
                with gr.Row():
                    confidence_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.25, label='Confidence Threshold', info='Confidence threshold (0-1)')
                    nms_iou_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.45, label='NMS IOU Threshold', info='NMS IOU threshold (0-1)')
                    max_detections = gr.Slider(minimum=1, maximum=1000, step=1, value=1000, label='Max Detections per Image', info='Maximum detections per image')
                    classes = gr.Textbox(value=None, label='Classes', placeholder='Enter class numbers separated by space, example: 0 1 2', info='Filter by classes')

                # Grouping visual and additional options
                gr.Markdown("### Visual and Additional Options")
                with gr.Row():
                    view_image = gr.Checkbox(value=False, label='View Image', info='Show results')
                    save_text = gr.Checkbox(value=False, label='Save Text', info='Save results to *.txt')
                    save_confidence = gr.Checkbox(value=False, label='Save Confidence', info='Save confidences in --save-txt labels')
                    save_crop = gr.Checkbox(value=False, label='Save Crop', info='Save cropped prediction boxes')
                    no_save = gr.Checkbox(value=False, label='No Save', info='Do not save images/videos')
                    augment = gr.Checkbox(value=False, label='Augment', info='Augmented inference')
                    visualize = gr.Checkbox(value=False, label='Visualize', info='Visualize features')
                    update = gr.Checkbox(value=False, label='Update', info='Update all models')

                # Grouping project settings and operational controls
                gr.Markdown("### Project Settings and Operational Controls")
                with gr.Row():
                    project = gr.Textbox(value=os.path.join(ROOT, 'detect_results'), label='Project', info='Save results to project/name', placeholder='Enter project directory')
                    name = gr.Textbox(value='exp', label='Name', info='Save results to project/name', placeholder='Enter experiment name')
                    exist_ok = gr.Checkbox(value=False, label='Exist OK', info='Existing project/name ok, do not increment')
                    line_thickness = gr.Slider(minimum=1, maximum=10, step=1, value=3, label='Line Thickness', info='Bounding box thickness (pixels)')
                    hide_labels = gr.Checkbox(value=False, label='Hide Labels', info='Hide labels on images')
                    hide_confidence = gr.Checkbox(value=False, label='Hide Confidence', info='Hide confidence on images')
                    use_fp16 = gr.Checkbox(value=False, label='Use FP16 Half-Precision Inference', info='Use FP16 half-precision inference')
                    use_opencv = gr.Checkbox(value=False, label='Use OpenCV DNN for ONNX Inference', info='Use OpenCV DNN for ONNX inference')

            with gr.Column():
                # Define output components
                gr.Markdown("### Diff Detection Output")
                with gr.Row():
                    detection_message = gr.Text(label="Detection Message")
                    download_output = gr.File(label="Download Output")

            with gr.Column():
                # Submit Button
                submit_button_diff_detection = gr.Button("Run Diff Detection")
                submit_button_diff_detection.click(
                    run_diff_detection,
                    inputs=[weights, source, data, image_height, image_width, confidence_threshold, nms_iou_threshold, max_detections, device, view_image, save_text,
                            save_confidence, save_crop, no_save, classes, class_agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness,
                            hide_labels, hide_confidence, use_fp16, use_opencv],
                    outputs=[detection_message, download_output]
                )

        with gr.Tab("Model Export"):
            gr.Markdown("## Model Export \nExport your trained models for deployment. This tab allows you to set up various export options and formats.")

            with gr.Column():
                # Grouping basic export settings
                gr.Markdown("### Basic Export Settings")
                with gr.Row():
                    data_yaml_path = gr.Textbox(label="Data YAML Path", value=os.path.join(ROOT, 'data/coco128.yaml'), info='dataset.yaml path', placeholder='Enter dataset.yaml path')
                    weights_paths = gr.Textbox(label="Weights Paths", value=os.path.join(ROOT, 'yolov5s.pt'), info='model.pt path(s)', placeholder='Enter model paths separated by space')
                    batch_size = gr.Number(label="Batch Size", value=1, info='Batch size for export')
                    device = gr.Radio(choices=['cpu', '0', '1', '2', '3'], label="Device", value='cpu', info='cuda device, i.e. 0 or 0, 1, 2, 3 or cpu')

                # Grouping model specifics and optimization settings
                gr.Markdown("### Model Specifics and Optimization Settings")
                with gr.Row():
                    use_fp16 = gr.Checkbox(label="Use FP16 Half-Precision", value=False, info='FP16 half-precision export')
                    inplace = gr.Checkbox(label="Set YOLOv5 Detect() Inplace=True", value=False, info='set YOLOv5 Detect() inplace=True')
                    train_mode = gr.Checkbox(label="Model Train Mode", value=False, info='model.train() mode')
                    optimize_mobile = gr.Checkbox(label="Optimize for Mobile", value=False, info='TorchScript: optimize for mobile')
                    int8_quantization = gr.Checkbox(label="INT8 Quantization", value=False, info='CoreML/TF INT8 quantization')

                # Grouping advanced technical settings
                gr.Markdown("### Advanced Technical Settings")
                with gr.Row():
                    dynamic_axes = gr.Checkbox(label="Dynamic Axes for ONNX/TF", value=False, info='ONNX/TF: dynamic axes')
                    simplify_onnx = gr.Checkbox(label="Simplify ONNX Model", value=False, info='ONNX: simplify model')
                    onnx_opset = gr.Number(label="ONNX Opset Version", value=12, info='ONNX: opset version')
                    verbose_log = gr.Checkbox(label="Verbose Logging for TensorRT", value=False, info='TensorRT: verbose log')
                    workspace_size = gr.Number(label="Workspace Size (GB) for TensorRT", value=4, info='TensorRT: workspace size (GB)')

                # Grouping output format and threshold settings
                gr.Markdown("### Output Format and Threshold Settings")
                with gr.Row():
                    add_nms = gr.Checkbox(label="Add NMS to TensorFlow Model", value=False, info='TF: add NMS to model')
                    agnostic_nms = gr.Checkbox(label="Add Class-agnostic NMS to TensorFlow Model", value=False, info='TF: add agnostic NMS to model')
                    topk_per_class = gr.Number(label="TopK Per Class for TensorFlow.js NMS", value=100, info='TF.js NMS: topk per class to keep')
                    topk_all = gr.Number(label="TopK All for TensorFlow.js NMS", value=100, info='TF.js NMS: topk for all classes to keep')
                    iou_threshold = gr.Slider(label="IoU Threshold for TensorFlow.js NMS", minimum=0, maximum=1, step=0.01, value=0.45, info='TF.js NMS: IoU threshold')
                    confidence_threshold = gr.Slider(label="Confidence Threshold for TensorFlow.js NMS", minimum=0, maximum=1, step=0.01, value=0.25, info='TF.js NMS: confidence threshold')

                # Grouping format selection
                gr.Markdown("### Format Selection")
                with gr.Row():
                    include_formats = gr.CheckboxGroup(label="Include Formats", choices=['torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'], 
                                                      value=['torchscript', 'onnx'],
                                                      info='Select from: torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')

            with gr.Column():
                # Define output component
                gr.Markdown("### Export Output")
                with gr.Row():
                    export_output = gr.Textbox()

            with gr.Column():
                # Submit Button
                submit_button_export = gr.Button("Run Export")
                submit_button_export.click(
                    run_export,
                    inputs=[data_yaml_path, weights_paths, image_height, image_width, batch_size, device, use_fp16, inplace, train_mode, optimize_mobile,
                            int8_quantization, dynamic_axes, simplify_onnx, onnx_opset, verbose_log, workspace_size, add_nms, agnostic_nms, topk_per_class,
                            topk_all, iou_threshold, confidence_threshold, include_formats],
                    outputs=export_output
                )

if __name__ == '__main__':
    iface.launch()