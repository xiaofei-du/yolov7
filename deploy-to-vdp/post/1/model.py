from labels import DatasetLabels
import numpy as np
import json
import sys
import torch

from pathlib import Path
from typing import List

from triton_python_backend_utils import Tensor, InferenceResponse, \
    get_input_tensor_by_name, InferenceRequest, get_input_config_by_name, \
    get_output_config_by_name, triton_string_to_numpy

sys.path.append(Path(__file__).parent.absolute().as_posix())


def scale_coords(img1_hw, coords: torch.Tensor, img0_hw, ratio_pad=None):
    # Rescale coords (xyxy) from img1_hw to img0_hw shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_hw[0] / img0_hw[0],
                   img1_hw[1] / img0_hw[1])  # gain  = old / new
        pad = (img1_hw[1] - img0_hw[1] * gain) / \
            2, (img1_hw[0] - img0_hw[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_hw)
    return coords


def clip_coords(boxes, img_hw):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_hw[1])  # x1
    boxes[:, 1].clamp_(0, img_hw[0])  # y1
    boxes[:, 2].clamp_(0, img_hw[1])  # x2
    boxes[:, 3].clamp_(0, img_hw[0])  # y2

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'input': 'input',
            'orig_img_hw': 'orig_img_hw',
            'scaled_img_hw': 'scaled_img_hw',
        }
        self.output_names = {
            'output_bboxes': 'output_bboxes',
            'output_labels': 'output_labels'
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')

        input_configs = {k: get_input_config_by_name(
            model_config, name) for k, name in self.input_names.items()}
        for k, cfg in input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')
        if len(model_config['output']) != 2:
            raise ValueError(
                f'Expected 2 outputs, got {len(model_config["output"])}')

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        responses = []

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found '
                                     f'in request {request.request_id()}')
                # convert to PyTorch tensor
                batch_in[k] = torch.from_numpy(tensor.as_numpy())

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            batch_num = len(batch_in['orig_img_hw'])
            # batch_in['input'] tensor on (n,7) with [image_idx, x1, y1, x2, y2, cls_id, score]
            batch_input = batch_in["input"]
            max_num_bboxes_in_single_img = 0
            preds_list, classes_list = [], []
            for img_idx in range(batch_num):
                input = batch_input[batch_input[:,0]==img_idx]
                preds_list.append(input[:,[1,2,3,4,6]])  # preds tensor on (n,5) with [x1, y1, x2, y2, score]
                classes_list.append(input[:,5].int())   # classes tensor on (n,) with [cls_id]
                max_num_bboxes_in_single_img = max(max_num_bboxes_in_single_img, len(input))

            if max_num_bboxes_in_single_img == 0:
                # When no detected object at all in all imgs in the batch
                for img_idx in range(batch_num):
                    batch_out['output_bboxes'].append([[-1, -1, -1, -1, -1]])
                    batch_out['output_labels'].append(['0'])
            else:
                # The output of all imgs must have the same size for Triton to be able to output a Tensor of type self.output_dtypes
                # Non-meaningful bounding boxes have coords [-1, -1, -1, -1, -1] and label '0'
                # Loop over images in batch
                for preds, classes, orig_img_hw, scaled_img_hw in zip(preds_list, classes_list, batch_in['orig_img_hw'], batch_in['scaled_img_hw']):                
                    # Rescale bounding boxes in boxes (n, 4) back to original image size
                    preds[:, :4] = scale_coords(
                        scaled_img_hw, preds[:, :4], orig_img_hw).round()
                    # Convert from tensor to numpy array
                    preds = preds.numpy()
                    num_to_add = max_num_bboxes_in_single_img - len(preds)
                    if self.output_names['output_bboxes'] in request.requested_output_names():
                        to_add = -np.ones((num_to_add, 5))
                        if len(preds) == 0:
                            batch_out['output_bboxes'].append(to_add)
                        else:
                            batch_out['output_bboxes'].append(np.vstack((preds, to_add)))
                    
                    if self.output_names['output_labels'] in request.requested_output_names():
                        to_add = ['0'] * num_to_add
                        if len(classes) == 0:
                            batch_out['output_labels'].append(to_add)
                        else:
                            categories = [DatasetLabels(idx.item()).name.lower() for idx in classes]
                            batch_out['output_labels'].append(categories + to_add)

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses