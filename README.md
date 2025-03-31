# RF-DETR Custom Dataset

## Overview
This repository demonstrates how to use RF-DETR (Region-Free Detection Transformer) for object detection with a custom dataset. The example workflow includes:
- Installing necessary dependencies
- Running inference on sample images
- Training a custom model using a dataset from Roboflow
- Visualizing training metrics and evaluation results
- Computing performance metrics such as mAP and confusion matrix

## Installation
Ensure you have an NVIDIA GPU with CUDA installed, then run:

```bash
!nvidia-smi
!pip install -q rfdetr supervision roboflow
```

## Running Inference on Sample Images

1. Download sample images:
    ```bash
    !wget -q https://media.roboflow.com/notebooks/examples/dog-2.jpeg
    !wget -q https://media.roboflow.com/notebooks/examples/dog-3.jpeg
    ```
2. Run inference using RF-DETR:
    ```python
    from rfdetr import RFDETRBase
    from rfdetr.util.coco_classes import COCO_CLASSES
    import supervision as sv
    from PIL import Image

    image = Image.open("dog-2.jpeg")
    model = RFDETRBase()
    detections = model.predict(image, threshold=0.5)
    ```

3. Annotate and visualize detections:
    ```python
    color = sv.ColorPalette.default()
    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    
    annotated_image = bbox_annotator.annotate(image.copy(), detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)
    annotated_image.show()
    ```

## Training on a Custom Dataset

1. Download a dataset from Roboflow:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("personalprojects-dtscj").project("waste-management-dataset")
    version = project.version(2)
    dataset = version.download("coco")
    ```

2. Train RF-DETR on the dataset:
    ```python
    model = RFDETRBase()
    history = []
    
    def callback(data):
        history.append(data)
    
    model.callbacks["on_fit_epoch_end"].append(callback)
    model.train(dataset_dir=dataset.location, epochs=10, batch_size=4, lr=1e-4)
    ```

## Visualizing Training Performance

1. Plot training and validation loss:
    ```python
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(history)

    plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
    plt.plot(df['epoch'], df['test_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
    ```

2. Plot precision and recall:
    ```python
    df['avg_precision'] = df['test_coco_eval_bbox'].apply(lambda arr: arr[0])
    df['avg_recall'] = df['test_coco_eval_bbox'].apply(lambda arr: arr[6])
    
    plt.plot(df['epoch'], df['avg_precision'], label='AP')
    plt.plot(df['epoch'], df['avg_recall'], label='AR')
    plt.legend()
    plt.show()
    ```

## Evaluating Model Performance

1. Compute mean Average Precision (mAP):
    ```python
    from supervision.metrics import MeanAveragePrecision
    map_metric = MeanAveragePrecision()
    map_result = map_metric.update(predictions, targets).compute()
    map_result.plot()
    ```

2. Generate a confusion matrix:
    ```python
    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=ds.classes
    )
    _ = confusion_matrix.plot()
    ```

## Conclusion
This repository provides a complete pipeline for running object detection with RF-DETR on a custom dataset. The model is trained using data from Roboflow and evaluated with mAP and confusion matrix metrics. Future improvements could include hyperparameter tuning and augmenting training data.

