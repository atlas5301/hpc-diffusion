from lib import PipelineSearcher
import lib
from diffusers import DiffusionPipeline
import diffusers
import torch
import add_project_to_path
import os
import copy
import json
import io
import pickle

def serialize(object):
    buffer = io.BytesIO()
    torch.save(object, buffer)
    return buffer.getvalue()

def deserialize(object):
    buffer = io.BytesIO(object)
    return torch.load(buffer)

project_dir = add_project_to_path.project_dir
# model_config_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4/unet/config.json')

def test_searcher():
    model_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4')
    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             ).to('cuda')
    tmp_models = {}
    for i in range(1):
        tmp_models[i] = lib.ModelData(i, serialize(pipeline.unet.to('cpu')), 4096, 0.06632188)
    tmpsearcher = PipelineSearcher(_baseline_pipeline= pipeline,
                                   _num_inference_steps = 30,
                                   _device_memory = 12000,
                                   _injected_models = tmp_models)
    
    
    tmpsearcher.scheduler_init()
    
    print('fully_initialized')
    # tmpsearcher.inject_pipelines([[j for i in range(30)] for j in range(len(tmpsearcher.generator.models))], expected_accuracy=0.95)
    print('injected_custom_pipeline')
    result = tmpsearcher.search(num_rounds=3, num_candidates=32, expected_accuracy=0.95, K=50)
    print('search_finished')
    print(result.accuracy, ' ', result.end2end_time, ' ', result.pipeline_info)
    
if __name__ == '__main__':
    test_searcher()