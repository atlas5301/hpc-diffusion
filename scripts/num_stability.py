import torch
from diffusers import DiffusionPipeline
from diffusers import DPMSolverSinglestepScheduler
from diffusers.models.cross_attention import AttnProcessor2_0
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import copy
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim

model_path="../models/stable-diffusion-v1-4"

INT8_NOISE_FACTOR = 1/256

INT4_NOISE_FACTOR = 1/16

INT2_NOISE_FACTOR = 1/2

def calculate_ssim_latent_space(latent_vector1, latent_vector2):
    # Normalize the latent vectors to the range [0, 1]
    scaler = MinMaxScaler()
    normalized_vectors = scaler.fit_transform(np.vstack([latent_vector1, latent_vector2]))

    # Calculate the Structural Similarity Index
    ssim_value = ssim(normalized_vectors[0], normalized_vectors[1],data_range=1)

    return ssim_value

def calculate_ssim(pil_img1, pil_img2, data_range=255):
    # Ensure both images have the same size
    if pil_img1.size != pil_img2.size:
        raise ValueError("Images must have the same dimensions")

    # Convert PIL images to grayscale numpy arrays
    img1_np = np.array(pil_img1.convert('L'), dtype=np.float64)
    img2_np = np.array(pil_img2.convert('L'), dtype=np.float64)

    # Calculate the Structural Similarity Index
    ssim_value = ssim(img1_np, img2_np, data_range=data_range)

    return ssim_value

def mse_pil(pil_img1, pil_img2):
    # Ensure both images have the same size
    if pil_img1.size != pil_img2.size:
        raise ValueError("Images must have the same dimensions")

    # Convert PIL images to numpy arrays
    img1_np = np.array(pil_img1, dtype=np.float32)
    img2_np = np.array(pil_img2, dtype=np.float32)

    # Calculate the Mean Squared Error
    mse_value = np.mean((img1_np - img2_np) ** 2)
    return mse_value

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def gen_random_noise(noise_factor: float):
    return (torch.rand(1,4,64,64, device=torch.device('cuda'))*2*noise_factor - noise_factor).half()

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


if __name__ == '__main__':
    torch.manual_seed(12345)

    model_path="../models/stable-diffusion-v1-4"
    prompt = "A fairytale-inspired castle on a hill, surrounded by lush gardens and a moat with a drawbridge."
    guidance_scale = 8.0
    t=981
    num_inference_steps=30
    

    pipeline = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,)
                                            #  scheduler=DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012,beta_schedule="scaled_linear"))
        # pipeline.scheduler = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012)
    pipeline.to("cuda")
        
    pipeline.unet.set_attn_processor(AttnProcessor2_0())  
    pipeline.safety_checker = None
    # pipeline.enable_xformers_memory_efficient_attention()
    # if pipeline.cross_attention_kwargs:
        # cross_attention_kwargs=pipeline.cross_attention_kwargs
    cross_attention_kwargs=None
    
    guidance_scale=7
    my_unet = pipeline.unet
    # my_unet = torch.compile(pipeline.unet)
    _encode_prompt = pipeline._encode_prompt
    num_images_per_prompt=1
    do_classifier_free_guidance = guidance_scale > 1.0
    negative_prompt=None
    decode_latents=pipeline.decode_latents
    numpy_to_pil=pipeline.numpy_to_pil
      
    device = pipeline._execution_device
    prompt_embeds = _encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    )
    generator = None
    scheduler = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012)
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    num_channels_latents = my_unet.config.in_channels
    # if pipeline.cross_attention_kwargs:
    #     cross_attention_kwargs=pipeline.cross_attention_kwargs
    # cross_attention_kwargs = None
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, 0.0)
  
    for j in range(10):
        latents = pipeline.prepare_latents(1,num_channels_latents,512,512,prompt_embeds.dtype, torch.device('cuda'), generator)
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        
        scheduler1 = copy.deepcopy(scheduler)
        scheduler2 = copy.deepcopy(scheduler)
        
        output1 = copy.deepcopy(latents)
        output2 = copy.deepcopy(latents)
        ERR=gen_random_noise(INT4_NOISE_FACTOR)
        
        prec_list = []
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                tmp1 = gen(my_unet, output1, t, prompt_embeds,cross_attention_kwargs, guidance_scale)
                output1 = scheduler1.step(tmp1, t, output1, **extra_step_kwargs).prev_sample
                
        
                tmp2 = gen(my_unet, output2+gen_random_noise(INT4_NOISE_FACTOR)+ERR, t, prompt_embeds,cross_attention_kwargs, guidance_scale)
                output2 = scheduler2.step(tmp2, t, output2, **extra_step_kwargs).prev_sample
                
                prec_list.append(float(F.mse_loss(output1, output2,reduction='mean')))
                
                # if (i%20 == 0): 
                #     print(".",end="",flush=True)
            out1=numpy_to_pil(decode_latents(output1))
            out2=numpy_to_pil(decode_latents(output2))
            
            get_concat_h(out1[0], out2[0]).save("../tmp/"+"test"+str(j)+".png")

            
            
        output = F.mse_loss(output1, output2,reduction='mean')
        # mse_image = mse_pil(out1[0], out2[0])
        # ssim_image = calculate_ssim_latent_space(output1.cpu().view(-1), output2.cpu().view(-1))
        # ssim_image_pil = calculate_ssim(out1[0],out2[0])
        # print(output,mse_image, ssim_image_pil,flush=True)   
        # print(float(output),ssim_image_pil,flush=True)   
         
        for i in prec_list:
            print(i, end=" ")
        print("",flush=True)
    





