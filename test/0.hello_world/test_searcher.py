from lib import PipelineSearcher
import lib
from diffusers import DiffusionPipeline
import torch
import add_project_to_path
import os
import copy

project_dir = add_project_to_path.project_dir

def test_searcher():
    model_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4')
    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,).to('cuda')
    tmpsearcher = PipelineSearcher(_baseline_pipeline= pipeline,)
    