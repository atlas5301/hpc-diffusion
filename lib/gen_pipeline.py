from __future__ import annotations
import torch
import diffusers
from diffusers import DiffusionPipeline
from diffusers import DPMSolverSinglestepScheduler
from typing import Any, Callable
import copy



def gen(my_unet, latents, t, prompt_embeds, cross_attention_kwargs, guidance_scale):
    with torch.no_grad():
        latent_model_input = torch.cat([latents] * 2)
        noise_pred = my_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
    
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred



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
    generator:PipelineGenerator = None
    pipeline_info:list = []
    device_model:dict[str, ModelData] = {}
    # current_memory = 0
    end2end_time = 0
    
    def __init__(self, generator, pipeline_info) -> None:
        self.generator = generator
        self.pipeline_info = pipeline_info
        self.device_model = {}
        self.end2end_time = 0
        for name in self.pipeline_info:
            if (name in self.device_model):
                pass
            else:
                if (name in self.generator.models):
                    self.device_model[name] = self.generator.models[name]
                    self.device_model[name].model.to(self.generator.device)    
                else:
                    raise Exception("wrong model")
            self.end2end_time += self.device_model[name].end2endtime
                
    def __del__(self):
        for tmp in self.device_model:
            tmp.model.to(self.generator.normal_device)
                    
    def __call__(self, prompt_embeds, latent, extra_step_kwargs) -> Any:
        output = latent
        with torch.no_grad():
            tmpscheduler = copy.deepcopy(self.generator.scheduler)
            tmpscheduler.set_timesteps(len(self.pipeline_info), device=self.generator.device)
            timesteps = tmpscheduler.timesteps
            for j, t in enumerate(timesteps):
                tmp1 = gen(self.device_model[self.pipeline_info[j]], output, t, prompt_embeds,None, self.generator.guidance_scale)
                output = tmpscheduler.step(tmp1, t, output, **extra_step_kwargs).prev_sample 
        return output              



class PipelineGenerator(object):
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012)
    guidance_scale:float = 8.0
    device:torch.device = torch.device('cuda')
    normal_device:torch.device = torch.device('cpu')
    # device_memory = 32768
    # normal_device_memory = 131072
    models:dict = {}
    
    def __init__(self,
                 _scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012),
                 _guidance_scale = 8.0,
                 _device = torch.device('cuda'),
                 _normal_device = torch.device('cpu'),
                #  _device_memory = 32768,
                #  _normal_device_memory = 131072,
                ):
        self.scheduler = _scheduler
        self.guidance_scale = _guidance_scale
        self.device = _device
        self.normal_device = _normal_device
        # self.device_memory = _device_memory
        # self.normal_device_memory = _normal_device_memory
        
    
    def load_models(self, json_path):
        '''
        Automatically load models(unets) from folder. Will read the json at first and automatically load models then.
        
        Args:
            json_path: the path of json file(contains the names, the sizes, the end2end time and the relative paths of these models)
            
        return:
            None
        '''
        pass   
    
        
    def generate(self, pipeline_info:list[str]):
        return MyPipeline(self, pipeline_info)
    
    
    