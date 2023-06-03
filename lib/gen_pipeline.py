from __future__ import annotations
import torch
import diffusers
from diffusers import DiffusionPipeline
from diffusers import DPMSolverSinglestepScheduler
from typing import Any, Callable
import copy
import json
import os
from . import model
import pickle
import io
# from model import model.UNet2DConditionModel

def serialize(object):
    buffer = io.BytesIO()
    torch.save(object, buffer)
    return buffer.getvalue()

def deserialize(object):
    buffer = io.BytesIO(object)
    return torch.load(buffer)

def gen(my_unet, latents, t, prompt_embeds, cross_attention_kwargs, guidance_scale):
    with torch.no_grad():
        # try:
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = my_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
    
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 
        # except Exception as e:
        #     with open('debug3.txt', "w") as f:
        #         f.write(str(e)) 
    return noise_pred
    

class PipelineUtils(object):
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012)
    guidance_scale:float = 8.0
    device_memory = 32768
    
    def __init__(self,
                _scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012),
                _guidance_scale = 8.0,
                _device_memory = 32768,
                ) -> None:
        self.scheduler = _scheduler
        self.guidance_scale = _guidance_scale
        self.device_memory = _device_memory



class ModelData(object):
    name = ""
    model = None
    size = 0
    end2endtime = 0
    
    
    def __init__(self, name, model, size, end2endtime) -> None:
        self.name = name
        self.model = model
        self.size = size
        self.end2endtime = end2endtime

class MyPipeline(object):
    pipeline_info:list = []
    device_model:dict[str, model.UNet2DConditionModel] = {}
    current_memory = 0
    end2end_time = 0.0
    utils: PipelineUtils = None
    
    def __init__(self, models: dict[str, ModelData], utils, pipeline_info) -> None:
        self.pipeline_info = pipeline_info
        self.device_model = {}
        self.end2end_time = 0.0
        self.current_memory = 0
        self.utils = copy.deepcopy(utils)
        self.device: torch.device = None
        for name in self.pipeline_info:
            if (name in self.device_model):
                pass
            else:
                if (name in models):
                    
                    self.device_model[name] = deserialize(models[name].model)
                    self.current_memory += models[name].size
                    if (self.current_memory > self.utils.device_memory):
                        raise Exception("Memory limit exceeded")
                else:
                    raise Exception("Wrong model")
            self.end2end_time += models[name].end2endtime
                
    def move_to_device(self, device: torch.device):
        for model in self.device_model.values():
            model.to(device)
        self.device = device
                    
    def __call__(self, prompt_embeds, latent, extra_step_kwargs) -> Any:
        output = latent
        with torch.no_grad():
            # with open('debug2.txt', "w") as f:
            #     f.write(str(prompt_embeds.dtype))
            #     f.write(str(latent.dtype))
            #     f.write(str(extra_step_kwargs))
            tmpscheduler = self.utils.scheduler
            tmpscheduler.set_timesteps(len(self.pipeline_info), device=self.device)
            timesteps = tmpscheduler.timesteps
            for j, t in enumerate(timesteps):
                tmp1 = gen(self.device_model[self.pipeline_info[j]], output, t, prompt_embeds,None, self.utils.guidance_scale) 
                output = tmpscheduler.step(tmp1, t, output, **extra_step_kwargs).prev_sample 
             
        return output              



class PipelineGenerator(object):
    utils: PipelineUtils
    # normal_device_memory = 131072
    models:dict[str, ModelData] = {}
    
    def __init__(self,
                 _scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012),
                 _guidance_scale = 8.0,
                 _device_memory = 32768,
                 _injected_models = {}
                #  _normal_device_memory = 131072,
                ):
        self.utils = PipelineUtils(_scheduler, _guidance_scale, _device_memory)
        self.models = _injected_models
        # self.normal_device_memory = _normal_device_memory
        
    
    def load_models(self, model_json_path, config_json_path):
        '''
        Automatically load models(unets) from folder. Will read the json at first and automatically load models then.
        
        Args:
            json_path: the path of json file(contains the names, the sizes, the end2end time and the relative paths(of the json) of these models)
            
        return:
            None
        '''
        def load_model(model_json_path, path):
            with open(model_json_path, "r") as f:
                config = json.load(f)
            unet = model.UNet2DConditionModel(**config)
            if path is not None:
                sd = torch.load(path)
                unet.load_state_dict(sd)
                return unet
            return None

            # replace with actual model loading code 
        
        with open(config_json_path, 'r') as f:
            tmpmodels = json.load(f)


        # now, models is a list of dictionaries, each representing a model
        for model in tmpmodels:
            path = os.path.join(os.path.dirname(config_json_path), model['relative_path'])

            loaded_model = load_model(model_json_path, path)
            self.models[model['name']] = ModelData(model['name'], serialize(loaded_model), model['size'], model['end2end_time'])  
            
    def generate(self, pipeline_info:list[str]):
        return MyPipeline(self.models, self.utils, pipeline_info)
    
    
    