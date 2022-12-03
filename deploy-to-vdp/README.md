---
Task: Detection
Tags:
  - Detection
  - YOLOv7
---

# YOLOv7 on VDP

Instructions to import and deploy [YOLOv7](https://github.com/WongKinYiu/yolov7) (CPU) via the open-source ETL tool [VDP](https://github.com/instill-ai/vdp).

**Prerequisites**
- Docker and Docker Compose
- Python 3.8+ with an environment-management tool such as Conda
- Download [`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) and put it under the repo root directory


## Export ONNX model

See https://github.com/WongKinYiu/yolov7#export for more info.

```bash
git clone https://github.com/xiaofei-du/yolov7.git

# Create and activate a new env
conda create --name vdp-yolov7  python=3.8
conda activate vdp-yolov7

# in the root folder, install all packages in requirements.txt
pip install -r requirements.txt 
# Install onnx-simplifier not listed in general yolov7 requirements.txt
pip install onnx
pip install onnx-simplifier

# Pytorch YOLOv7 -> ONNX with grid, EfficientNMS plugin and and dynamic batch size
python export.py --weights ./yolov7.pt --grid --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --export-onnx-model-path deploy-to-vdp/infer/1/model.onnx --end2end --max-wh 640
```

The ONNX model will be exported to `/deploy-to-vdp/infer/1/model.onnx`. The `deploy-to-vdp` directory will look like the following:

```bash
├── README.md
├── infer
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
├── post
│   ├── 1
│   │   ├── labels.py
│   │   └── model.py
│   └── config.pbtxt
├── pre
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── yolov7
    ├── 1
    └── config.pbtxt
```

> **Note**
> 
> If you want to use YOLOv7 with GPU support, replace `KIND_CPU` with `KIND_GPU` in the `deploy-to-vdp/infer/config.pbtxt`.

## Deploy YOLOv7 on VDP

### Run VDP locally

```bash
git clone https://github.com/instill-ai/vdp.git && cd vdp
make all
```

If this is your first time setting up VDP, access the Console (http://localhost:3000) and you should see the onboarding page. Please enter your email and you are all set!

### Create a YOLOv7 pipeline

**Step 1: Add a HTTP source**

A `HTTP` source accepts HTTP requests with image payloads to be processed by a pipeline.

To set it up,

1. click the **Pipeline mode** ▾ drop-down and choose `Sync`,
2. click the **Source type** ▾ drop-down and choose `HTTP`, and
3. click **Next**.

**Step 2: Import YOLOv7 model**

Create a zip file `yolov7.zip` including YOLOv7 configurations and model weights.

```bash
cd deploy-to-vdp
zip -r yolov7.zip .
```

1. give your model a unique ID `yolov7`,
2. [optional] add description,
3. click the **Model source** ▾ drop-down and choose `Local`,
4. click **Upload** to upload the `yolov7.zip` file from your computer, and
click **Set up**.

See [Import the local model](https://www.instill.tech/docs/import-models/local#no-code-setup) for more details.

**Step 3: Deploy a model instance of the imported model**

Once the model is imported,

1. click the **Model instances** ▾ drop-down,
2. pick `latest`, and
3. click **Deploy** to put it online.

Step 4: Add a HTTP destination

Since we are building a SYNC pipeline, the HTTP destination is automatically paired with the HTTP source.

Just click Next.

**Step 5: Set up the pipeline**

Almost done! Just

1. give your pipeline a unique ID `yolov7`,
2. [optional] add description, and
3. click **Set up**.

Now you should see the newly created SYNC pipeline `yolov7` on the Pipeline page 🎉

### Trigger your YOLOv7 pipeline

Now that the `yolov7` pipeline is automatically activated, you can make a request to trigger the pipeline:

```bash
curl -X POST http://localhost:8081/v1alpha/pipelines/yolov7/trigger -d '{
  "inputs": [
    {
      "image_url": "https://artifacts.instill.tech/imgs/dog.jpg"
    },
    {
      "image_url": "https://artifacts.instill.tech/imgs/polar-bear.jpg"
    }
  ]
}'
```

A HTTP response will return

```json
{
    "data_mapping_indices": [
        "01GKD4G4G953Q1R0MR2HJ75J2D",
        "01GKD4G4G953Q1R0MR2K3T5Z0S"
    ],
    "model_instance_outputs": [
        {
            "model_instance": "models/yolov7/instances/latest",
            "task": "TASK_DETECTION",
            "task_outputs": [
                {
                    "index": "01GKD4G4G953Q1R0MR2HJ75J2D",
                    "detection": {
                        "objects": [
                            {
                                "category": "dog",
                                "score": 0.9628708,
                                "bounding_box": {
                                    "top": 102,
                                    "left": 324,
                                    "width": 208,
                                    "height": 404
                                }
                            },
                            {
                                "category": "dog",
                                "score": 0.92833334,
                                "bounding_box": {
                                    "top": 198,
                                    "left": 130,
                                    "width": 197,
                                    "height": 237
                                }
                            }
                        ]
                    }
                },
                {
                    "index": "01GKD4G4G953Q1R0MR2K3T5Z0S",
                    "detection": {
                        "objects": [
                            {
                                "category": "bear",
                                "score": 0.9468723,
                                "bounding_box": {
                                    "top": 455,
                                    "left": 1372,
                                    "width": 1300,
                                    "height": 2179
                                }
                            }
                        ]
                    }
                }
            ]
        }
    ]
}
```


🙌 That's it! You just built your first `SYNC` pipeline and triggered it to convert unstructured image data into structured and analyzable insight.

### What's next

By now, you should have a basic understanding of how VDP streamlines the end-to-end ETL pipelines for visual data. This tutorial only shows the tip of what VDP is capable of and is just the beginning of your VDP journey.

Check out our tutorial on [How to build a shareable object detection application with VDP and Streamlit](https://blog.instill.tech/vdp-streamlit-yolov7/).

If you have any problem at all, join our Discord to get community support.


### Shut down VDP

To shut down all running services:

```bash
make down
```
