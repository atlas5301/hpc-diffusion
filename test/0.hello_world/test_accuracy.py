import lib
from diffusers import DiffusionPipeline
import torch
import add_project_to_path
import os

project_dir = add_project_to_path.project_dir

def test_accuracy_functional():
    model_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4')
    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,).to('cuda')
    accbench = lib.AccBenchmark(pipeline, _benchmark_sample_num = 2, _batch_size=2)
    val = accbench.acc_benchmark(pipeline.unet)
    assert(val > 0.6)
    

