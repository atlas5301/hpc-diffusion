from lib import PipelineSearcher
import lib
from diffusers import DiffusionPipeline
import torch
import add_project_to_path
import os

project_dir = add_project_to_path.project_dir
model_config_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4/unet/config.json')
model_path = ""


if __name__ == '__main__':
    model_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4')
    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,).to('cuda')
    tmpsearcher = PipelineSearcher(_baseline_pipeline= pipeline,)
    tmpsearcher.load(model_config_path, model_path)
    tmpsearcher.scheduler_init()
    tmpsearcher.inject_pipelines([[j for i in range(50)] for j in range(len(tmpsearcher.generator.models))])
    print(tmpsearcher.search())