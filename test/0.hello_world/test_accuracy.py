import lib
from diffusers import DiffusionPipeline
import torch
import add_project_to_path
import os
import copy

project_dir = add_project_to_path.project_dir


def gen(my_unet, latents, t, prompt_embeds, cross_attention_kwargs, guidance_scale, do_classifier_free_guidance):
    with torch.no_grad():
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        noise_pred = my_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
    
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def test_acc_benchmark_unet():
    model_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4')
    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,).to('cuda')
    accbench = lib.AccBenchmark(pipeline, _benchmark_sample_num = 4, _batch_size=4)
    val = accbench.acc_benchmark_single(pipeline.unet)
    print(val)
    assert(val > 0.6)
    
def test_acc_benchmark_pipeline():
    model_path = os.path.join(project_dir, 'models/stable-diffusion-v1-4')
    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,).to('cuda')
    accbench = lib.AccBenchmark(pipeline, _benchmark_sample_num = 4, _batch_size=4)
    
    def my_pipeline(prompt_embeds, latent, extra_step_kwargs):
        my_unet = pipeline.unet.to(accbench.device)
        output = latent
        
        with torch.no_grad():
            tmpscheduler = copy.deepcopy(accbench.scheduler)
            tmpscheduler.set_timesteps(accbench.num_inference_steps, device=accbench.device)
            timesteps = tmpscheduler.timesteps
            for j, t in enumerate(timesteps):
                tmp1 = gen(my_unet, output, t, prompt_embeds,accbench.cross_attention_kwargs, accbench.guidance_scale, accbench.do_classifier_free_guidance)
                output = tmpscheduler.step(tmp1, t, output, **extra_step_kwargs).prev_sample 
        return output
    
    val = accbench.acc_benchmark_pipeline(my_pipeline)
    print(val)
    assert(val > 0.6)
       
    

