from lib import TaskScheduler
import add_project_to_path
project_dir = add_project_to_path.project_dir


import add_project_to_path
import os
import copy
from diffusers import DiffusionPipeline
from diffusers import UNet2DConditionModel
import json
import multiprocessing as mp
import torch

def task(input, device, shared_object):
    import add_project_to_path
    try:
        project_dir = add_project_to_path.project_dir
        unet_config_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4/unet/config.json')
        f = open(unet_config_path, "r")
        config = json.load(f)
        f.close()
        model = UNet2DConditionModel(**config).to(device)
        shared_object["model"].to(device)
    except Exception as e:
        return e
    # Rest of your task here
    if (input == 3):
        raise Exception()
    return 2  # Dummy output for the sake of example


def test_multiprocess():
    project_dir = add_project_to_path.project_dir
    model_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4')

    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                                torch_dtype=torch.float16)

    # model = torch.nn.Linear(2, 2)
    model = pipeline.unet
    shared_object = {'model': model}
    inputs = list(range(8))
    scheduler = TaskScheduler(task, shared_object)
    results = scheduler.submit(inputs)
    assert (len(results) == 8)
    # results = scheduler.submit(inputs)
    # results = scheduler.submit(inputs)
    # results = scheduler.submit(inputs)
    results2 = scheduler.submit(inputs)
    # print(results)
    # scheduler.close()
    assert(results2 == [2,2,2,None,2,2,2,2])