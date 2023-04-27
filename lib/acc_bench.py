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

prompts = [
    "A serene lakeside scene with a family of swans gliding across the water, surrounded by weeping willows and a quaint wooden gazebo.",
    "A futuristic agricultural farm on a distant planet, with robotic workers tending to alien plants, and a biodome protecting the area from harsh weather conditions.",
    "An ancient Egyptian marketplace filled with merchants selling exotic spices, precious gems, and handcrafted items, while locals and visitors barter for goods.",
    "A breathtaking underwater waterfall in a hidden oceanic cave, illuminated by shafts of sunlight streaming through the water's surface, and teeming with vibrant coral and tropical fish.",
    "An interactive art gallery where visitors can walk through holographic, immersive exhibits that respond to their movements and emotions, creating a unique and ever-changing experience.",
    "A bustling port city in the 18th century, with tall ships from various nations docked in the harbor, sailors unloading cargo, and lively markets along the waterfront.",
    "A whimsical winter wonderland with snow-covered trees, ice sculptures, and a festive ice-skating rink, illuminated by twinkling fairy lights and a warm, glowing sunset.",
    "A serene Japanese tea ceremony taking place in a traditional tatami room, with participants dressed in elegant kimonos and a masterful tea master preparing the tea.",
    "An expansive savannah landscape during the great migration, featuring herds of wildebeest, zebras, and other wildlife crossing a river, while predators lurk in the shadows.",
    "A cozy, rustic mountain lodge with a crackling fireplace, plush armchairs, and warm, inviting décor, surrounded by a snowy landscape and towering evergreen trees.",
    "A retro-futuristic diner set in the 1950s, with hovering cars parked outside, waitstaff dressed in vintage attire, and patrons enjoying milkshakes and burgers at neon-lit booths.",
    "A magical garden filled with oversized, glowing flowers, winding pathways, and mischievous fairies flitting among the foliage, creating an enchanting and otherworldly atmosphere.",
    "A picturesque vineyard in the Italian countryside, with rows of grapevines stretching across rolling hills, a rustic farmhouse, and workers harvesting ripe grapes.",
    "A breathtaking view of Earth as seen from the International Space Station, with swirling cloud formations, vibrant continents, and the awe-inspiring curvature of the planet.",
    "A lavish, Baroque-style ballroom with gilded walls, marble columns, and a beautifully painted ceiling, filled with elegantly dressed dancers waltzing to a live orchestra.",
    "A bustling, futuristic bazaar on a distant desert planet, with merchants hawking wares from various galaxies, and a diverse array of alien species browsing the stalls.",
    "An elegant, Art Deco-inspired hotel lobby with a grand staircase, intricate geometric patterns, and plush, velvet furnishings, evoking the glamour of the Roaring Twenties.",
    "A peaceful, moonlit night in a dense bamboo forest, with a gentle breeze rustling the leaves, and a solitary red-crowned crane standing gracefully in a small clearing.",
    "A lively Renaissance fair set within a medieval village, featuring jesters, knights, archery contests, and artisans showcasing their crafts, transporting visitors to a bygone era.",
    "A secluded tropical lagoon with crystal-clear turquoise waters, white sandy beaches, and lush palm trees, where a group of dolphins playfully leap and splash in the gentle waves."
]


def find_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(numbers)

    if n % 2 == 0:
        median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        median = sorted_numbers[n//2]

    return median

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

class AccBenchmark(object):
    benchmark_sample_num:int = 20
    benchmark_samples: list = []
    baseline_pipeline = None
    scheduler = None
    prompt_embeddings:list = []
    init_latents:list = []
    guidance_scale:float = 8.0
    cross_attention_kwargs=None
    device = torch.device('cuda')
    num_inference_steps = 30
    resolution = 512
    num_to_pil = None
    decode_latents = None
    prepare_extra_step_kwargs = None
    batch_size = 0
    do_classifier_free_guidance = True

    def __init__(self,
                 _baseline_pipeline,
                 _scheduler = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012),
                 _benchmark_sample_num = 20,
                 _guidance_scale = 8.0,
                 _device = torch.device('cuda'),
                 _num_inference_steps:int = 30,
                 _resolution:int =512,
                 _batch_size: int = 1
                 ) -> None:
        self.baseline_pipeline = _baseline_pipeline
        self.scheduler=_scheduler
        self.benchmark_sample_num=_benchmark_sample_num
        self.guidance_scale = _guidance_scale
        self.device = _device
        self.num_inference_steps = _num_inference_steps
        self.resolution = _resolution
        self.batch_size = _batch_size
        self.benchmark_samples = []
        self.prompt_embeddings = []
        self.init_latents = []
        self.baseline_pipeline.to(self.device)
        self.prepare_extra_step_kwargs = self.baseline_pipeline.prepare_extra_step_kwargs
        

        #then, I will start generate the baseline images
        with torch.no_grad():
        #first initialization
            baseline_unet=self.baseline_pipeline.unet.to(self.device)
            _encode_prompt = self.baseline_pipeline._encode_prompt
            num_images_per_prompt=self.batch_size
            self.do_classifier_free_guidance = self.guidance_scale > 1.0
            negative_prompt=None
            self.decode_latents=self.baseline_pipeline.decode_latents
            self.numpy_to_pil=self.baseline_pipeline.numpy_to_pil

            for i in range(self.benchmark_sample_num):
                prompt = prompts[i]
                prompt_embeds = _encode_prompt(
                    prompt,
                    self.device,
                    num_images_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                )
                self.prompt_embeddings.append(prompt_embeds)
            
                tmpscheduler = copy.deepcopy(self.scheduler)
                tmpscheduler.set_timesteps(self.num_inference_steps, device=self.device)
                timesteps = tmpscheduler.timesteps
                num_channels_latents = baseline_unet.config.in_channels  
                generator = None 
                extra_step_kwargs = self.baseline_pipeline.prepare_extra_step_kwargs(generator, 0.0)   
                latents = self.baseline_pipeline.prepare_latents(self.batch_size,num_channels_latents,self.resolution,self.resolution,prompt_embeds.dtype, self.device, generator) 
                self.init_latents.append(latents)   
                # num_warmup_steps = len(timesteps) - self.num_inference_steps * self.scheduler.order
                output1 = copy.deepcopy(latents)
                for j, t in enumerate(timesteps):
                    tmp1 = gen(baseline_unet, output1, t, prompt_embeds,self.cross_attention_kwargs, self.guidance_scale,self.do_classifier_free_guidance)
                    output1 = tmpscheduler.step(tmp1, t, output1, **extra_step_kwargs).prev_sample
                out1=self.numpy_to_pil(self.decode_latents(output1))
                self.benchmark_samples.append(out1)
        self.baseline_pipeline = None


    def acc_benchmark(self, my_unet):
        my_unet.to(self.device)
        outs = []
        for i in range(self.benchmark_sample_num):
            output = copy.deepcopy(self.init_latents[i])
            prompt_embeds = self.prompt_embeddings[i]
            generator = None 
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0) 
            with torch.no_grad():
                tmpscheduler = copy.deepcopy(self.scheduler)
                tmpscheduler.set_timesteps(self.num_inference_steps, device=self.device)
                timesteps = tmpscheduler.timesteps
                for j, t in enumerate(timesteps):
                    tmp1 = gen(my_unet, output, t, prompt_embeds,self.cross_attention_kwargs, self.guidance_scale, self.do_classifier_free_guidance)
                    output = tmpscheduler.step(tmp1, t, output, **extra_step_kwargs).prev_sample 
                out = self.numpy_to_pil(self.decode_latents(output))
                for j in range(self.batch_size):
                    outs.append(calculate_ssim(out[j], self.benchmark_samples[i][j]))
        return find_median(outs)
        

# here is an example
# if __name__ == "__main__":
#     model_path="../models/stable-diffusion-v1-4"
#     pipeline = DiffusionPipeline.from_pretrained(model_path,
#                                              torch_dtype=torch.float16,).to('cuda')
#     accbench = AccBenchmark(pipeline, _benchmark_sample_num = 20, _batch_size=2)
#     print("hey")
#     print(accbench.acc_benchmark(pipeline.unet))
            