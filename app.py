import gradio as gr
import os
import sys
from pathlib import Path
from detection import run_detection
from train import run_training
from val import run_validation


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Add tabbed interface for ML training parameters
main_iface = gr.Interface(
    fn=run_detection,
    inputs=[
        gr.Textbox(value=os.path.join(ROOT, 'best-solar.pt'), label='Weights', placeholder='Enter model paths separated by space', info='model path(s)'),
        gr.Textbox(label='Source', value=os.path.join(ROOT, 'test_folder'), info='file/dir/URL/glob, 0 for webcam', placeholder='Enter source path'),
        gr.Textbox(label='Data', value=os.path.join(ROOT, 'data.yaml'), info='(optional) dataset.yaml path', placeholder='Enter dataset.yaml path'),
        gr.Number(label='Image Height', value=640, info='inference image size height'),
        gr.Number(label='Image Width', value=640, info='inference image size width'),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.25, label='Confidence Threshold', info='Confidence threshold (0-1)'),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.45, label='NMS IOU Threshold', info='NMS IOU threshold (0-1)'),
        gr.Slider(minimum=1, maximum=1000, step=1, value=1000, label='Max Detections per Image', info='Maximum detections per image'),
        gr.Radio(choices=['cpu', '0', '1', '2', '3'], label='Device', value='cpu', info='cuda device, i.e. 0 or 0,1,2,3 or cpu'),
        gr.Checkbox(value=False, label='View Image', info='Show results'),
        gr.Checkbox(value=False, label='Save Text', info='Save results to *.txt'),
        gr.Checkbox(value=False, label='Save Confidence', info='Save confidences in --save-txt labels'),
        gr.Checkbox(value=False, label='Save Crop', info='Save cropped prediction boxes'),
        gr.Checkbox(value=False, label='No Save', info='Do not save images/videos'),
        gr.Textbox(value=None, label='Classes', placeholder='Enter class numbers separated by space, example: 0 1 2', info='Filter by classes'),
        gr.Checkbox(value=False, label='Class-agnostic NMS', info='Class-agnostic NMS'),
        gr.Checkbox(value=False, label='Augment', info='Augmented inference'),
        gr.Checkbox(value=False, label='Visualize', info='Visualize features'),
        gr.Checkbox(value=False, label='Update', info='Update all models'),
        gr.Textbox(value=os.path.join(ROOT, 'detect_results'), label='Project', info='Save results to project/name', placeholder='Enter project directory'),
        gr.Textbox(value='exp', label='Name', info='Save results to project/name', placeholder='Enter experiment name'),
        gr.Checkbox(value=False, label='Exist OK', info='Existing project/name ok, do not increment'),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label='Line Thickness', info='Bounding box thickness (pixels)'),
        gr.Checkbox(value=False, label='Hide Labels', info='Hide labels on images'),
        gr.Checkbox(value=False, label='Hide Confidence', info='Hide confidence on images'),
        gr.Checkbox(value=False, label='Use FP16 Half-Precision Inference', info='Use FP16 half-precision inference'),
        gr.Checkbox(value=False, label='Use OpenCV DNN for ONNX Inference', info='Use OpenCV DNN for ONNX inference'),
    ],
    outputs=[
        gr.File(label='Saved Images'),
        gr.File(label='Saved Text Files'),
    ],
    title="YOLOv5 Image Detection Interface",
    description="Detection of faults in solar panels using thermal images",
)

# Define the Gradio interface
train_iface = gr.Interface(
    fn=run_training,
    inputs=[
        gr.Textbox(label='Weights', value=os.path.join(ROOT, 'yolov5s.pt'), info='Initial weights path', placeholder='Enter single model path'),
        gr.Textbox(label='Model Config', value='', info='model.yaml path', placeholder='Enter model.yaml path'),
        gr.Textbox(label='Data Config', value=os.path.join(ROOT, 'data', 'coco128.yaml'), info='dataset.yaml path', placeholder='Enter dataset.yaml path'),
        gr.Textbox(label='Hyperparameters', value=os.path.join(ROOT, 'data', 'hyps', 'hyp.scratch-low.yaml'), info='Hyperparameters path', placeholder='Enter hyperparameters path'),
        gr.Number(label='Epochs', value=300, info='Number of epochs to train'),
        gr.Number(label='Batch Size', value=16, info='Total batch size for all GPUs, -1 for autobatch'),
        gr.Number(label='Image Size', value=640, info='train, val image size (pixels)'),
        gr.Checkbox(label='Rectangular Training', value=False, info='Rectangular training'),
        gr.Checkbox(label='Resume Training', value=False, info='Resume most recent training'),
        gr.Checkbox(label='No Save', value=False, info='Only save final checkpoint'),
        gr.Checkbox(label='No Validation', value=False, info='Only validate final epoch'),
        gr.Checkbox(label='No AutoAnchor', value=False, info='Disable AutoAnchor'),
        gr.Number(label='Evolve', value=300, info='Evolve hyperparameters for x generations'),
        gr.Textbox(label='Bucket', value='', info='gsutil bucket', placeholder='gs://bucket'),
        gr.Radio(label='Cache Images', choices=['ram', 'disk'], value='ram', info='--cache images in "ram" (default) or "disk"'),
        gr.Checkbox(label='Image Weights', value=False, info='use weighted image selection for training'),
        gr.Radio(choices=['cpu', '0', '1', '2', '3'], label='Device', value='cpu', info='cuda device, i.e. 0, 1, 2, 3 or cpu'),
        gr.Checkbox(label='Multi-Scale', value=False, info='vary img-size +/- 50%%'),
        gr.Checkbox(label='Single Class', value=False, info='train multi-class data as single-class'),
        gr.Radio(label='Optimizer', choices=['SGD', 'Adam', 'AdamW'], value='SGD', info='Optimizer selection'),
        gr.Checkbox(label='Sync BatchNorm', value=False, info='Use SyncBatchNorm, only available in DDP mode'),
        gr.Number(label='Workers', value=8, info='Max dataloader workers (per RANK in DDP mode)'),
        gr.Textbox(label='Project', value=os.path.join(ROOT, 'runs', 'train'), info='Save to project/name', placeholder='Enter project directory'),
        gr.Textbox(label='Name', value='exp', info='Save to project/name', placeholder='Enter experiment name'),
        gr.Checkbox(label='Exist OK', value=False, info='Existing project/name ok, do not increment'),
        gr.Checkbox(label='Quad DataLoader', value=False, info='Quad dataloader'),
        gr.Checkbox(label='Cosine LR Scheduler', value=False, info='Cosine LR scheduler'),
        gr.Number(label='Label Smoothing', value=0.0, info='Label smoothing epsilon'),
        gr.Number(label='Patience', value=100, info='EarlyStopping patience (epochs without improvement)'),
        gr.Textbox(value='0', label='Freeze Layers', placeholder='Enter layer numbers separated by space', info='Freeze layers: backbone=10, first3=0 1 2'),
        gr.Number(label='Save Period', value=-1, info='Save checkpoint every x epochs (disabled if < 1)'),
        gr.Number(label='Local Rank', value=-1, info='DDP parameter, do not modify'),
        gr.Textbox(label='Entity', value='', info='W&B: Entity', placeholder='Enter W&B entity'),
        gr.Checkbox(label='Upload Dataset', value=False, info='W&B: Upload data, "val" option'),
        gr.Number(label='BBox Interval', value=-1, info='W&B: Set bounding-box image logging interval'),
        gr.Textbox(label='Artifact Alias', value='latest', info='W&B: Version of dataset artifact to use', placeholder='Enter W&B artifact alias'),
    ],
    outputs="text",
    title="YOLOv5 Model Training Interface",
    description="Configure parameters to train the YOLOv5 model."
)

val_iface = gr.Interface(
    fn=run_validation,
    inputs=[
        gr.Textbox(label="Data YAML Path", value=os.path.join(ROOT, 'data', 'coco128.yaml'), info='dataset.yaml path', placeholder='Enter dataset.yaml path'),
        gr.Textbox(value=os.path.join(ROOT, 'yolov5s.pt'), label='Weights', placeholder='Enter model paths separated by space', info='model.pt path(s)'),
        gr.Number(label="Batch Size", value=32, info='Batch size'),
        gr.Number(label="Image Size (pixels)", value=640, info='inference size (pixels)'),
        gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, step=0.01, value=0.001, info='Confidence threshold (0-1)'),
        gr.Slider(label="NMS IoU Threshold", minimum=0, maximum=1, step=0.01, value=0.6, info='NMS IoU threshold (0-1)'),
        gr.Radio(label="Task", choices=['train', 'val', 'test', 'speed', 'study'], value='val', info='Task to run'),
        gr.Radio(choices=['cpu', '0', '1', '2', '3'], label='Device', value='cpu', info='cuda device, i.e. 0 or 0,1,2,3 or cpu'),
        gr.Number(label="Max Dataloader Workers", value=8, info='Max dataloader workers (per RANK in DDP mode)'),
        gr.Checkbox(label="Single Class", value=False, info='Treat as single-class dataset'),
        gr.Checkbox(label="Augment", value=False, info='Augmented inference'),
        gr.Checkbox(label="Verbose", value=False, info='Report mAP by class'),
        gr.Checkbox(label="Save to TXT", value=False, info='Save results to *.txt'),
        gr.Checkbox(label="Save Hybrid", value=False, info='Save label+prediction hybrid results to *.txt'),
        gr.Checkbox(label="Save Confidences", value=False, info='Save confidences in --save-txt labels'),
        gr.Checkbox(label="Save JSON", value=False, info='Save a COCO-JSON results file'),
        gr.Textbox(label='Project Directory', value=os.path.join(ROOT, 'runs', 'val'), info='Save to project/name', placeholder='Enter project directory'),
        gr.Textbox(label="Experiment Name", value='exp', info='Save to project/name', placeholder='Enter experiment name'),
        gr.Checkbox(label="Exist OK", value=False, info='Existing project/name ok, do not increment'),
        gr.Checkbox(label="Use FP16 Half-Precision", value=False, info='Use FP16 half-precision inference'),
        gr.Checkbox(label="Use DNN", value=False, info='Use OpenCV DNN for ONNX inference')
    ],
    outputs="text",
    title="YOLOv5 Validation Interface",
    description="Configure parameters to validate the YOLOv5 model."
)

iface = gr.TabbedInterface(
    [main_iface, train_iface, val_iface],
    ["Image Detection", "Image Training", "Image Validation"],
    title="Automatic-Faults-Detection-of-Photovoltaic-Farms-using-Thermal-Images",  # Title of the interface
)

if __name__ == '__main__':
    iface.launch()