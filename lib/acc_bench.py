import torch
import diffusers
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
import pickle
import io

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


def find_median(numbers:list):
    '''
        give the median of the list 'numbers'

        Args:
            numbers: a list of numbers

        Returns:
            the median of 'numbers'
    '''
    sorted_numbers = sorted(numbers)
    n = len(numbers)

    if n % 2 == 0:
        median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        median = sorted_numbers[n//2]

    return median

def calculate_ssim(pil_img1, pil_img2, data_range=255):
    '''
    calculate the ssim of two pil images.

    Args:
        pil_img1: the first image
        pil_img2: the second image

    Returns:
        Returns the ssim score, which ranges from -1 to 1. Higher scores indicate better similarity.
    '''
    
    # Ensure both images have the same size
    if pil_img1.size != pil_img2.size:
        raise ValueError("Images must have the same dimensions")

    # Convert PIL images to grayscale numpy arrays
    img1_np = np.array(pil_img1.convert('L'), dtype=np.float64)
    img2_np = np.array(pil_img2.convert('L'), dtype=np.float64)

    # Calculate the Structural Similarity Index
    ssim_value = ssim(img1_np, img2_np, data_range=data_range)

    return ssim_value

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

def serialize(object):
    buffer = io.BytesIO()
    torch.save(object, buffer)
    return buffer.getvalue()

def deserialize(object):
    buffer = io.BytesIO(object)
    return torch.load(buffer)

class MinAccbench(object):
    benchmark_sample_num = None,
    init_latents = None
    prompt_embeddings = None
    extra_step_kwargs = None
    
    def __init__(
        self,
        benchmark_sample_num,
        init_latents,
        prompt_embeddings,
        extra_step_kwargs,
        benchmark_samples
    ):
        
        
        self.benchmark_sample_num = copy.deepcopy(benchmark_sample_num)
        self.init_latents = serialize(copy.deepcopy(init_latents))
        self.prompt_embeddings = serialize(copy.deepcopy(prompt_embeddings))
        self.extra_step_kwargs = serialize(copy.deepcopy(extra_step_kwargs))    
        # self.decode_latents_model = decode_latents
        # self.batch_size = copy.deepcopy(batch_size)
        self.benchmark_samples = copy.deepcopy(benchmark_samples)
    
    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def acc_benchmark_pipeline(self, pipeline, device):
        '''
        This benchmark exclusively evaluates a pipeline. .

        Args:
            pipeline: The pipeline to be assessed. get the given prompt_embeds and initial latent, and output the generated latent.

        Returns:
            Returns the accuracy score, which ranges from -1 to 1. Higher scores indicate better performance.
        ''' 

        tmpinit_latents = deserialize(self.init_latents)
        tmpprompt_embeddings = deserialize(self.prompt_embeddings)
        tmpextra_step_kwargs = deserialize(self.extra_step_kwargs)    
        outputs = []
        for i in range(self.benchmark_sample_num):
            latent = copy.deepcopy(tmpinit_latents[i]).to(device)
            prompt_embeds = tmpprompt_embeddings[i].to(device)
            extra_step_kwargs = tmpextra_step_kwargs[i]
            with torch.no_grad():
                output = pipeline(prompt_embeds, latent, extra_step_kwargs)            
                outputs.append(output)       
        outs = []
        pipeline.vae.to(device)
        with torch.no_grad():
            len_outputs = len(outputs)
            for i in range(len_outputs):
                output=outputs[i]
                out = self.numpy_to_pil(pipeline.decode_latents(output)) 
                output.to('cpu')
                for j in range(len(out)):
                    outs.append(calculate_ssim(out[j], self.benchmark_samples[i][j]))
        pipeline.vae.to('cpu')
        return find_median(outs)
    

class AccBenchmark(object):
    benchmark_sample_num:int = 20
    benchmark_samples: list = []
    baseline_pipeline = None
    scheduler = None
    prompt_embeddings:list = []
    init_latents:list = []
    extra_step_kwargs:list = []
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
                 _baseline_pipeline:DiffusionPipeline,
                 _scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012),
                 _benchmark_sample_num = 20,
                 _guidance_scale = 8.0,
                 _device = torch.device('cuda'),
                 _num_inference_steps:int = 30,
                 _resolution:int =512,
                 _batch_size: int = 1
                 ) -> None:
        
        '''
        init the pipeline

        Args:
            _baseline_pipeline: The baseline pipeline. Only original diffusers.DiffusionPipeline supported.
            _scheduler: the scheduler of the pipeline(for both baseline and custom pipelines).
            _benchmark_sample_num: sample nums of the benchmark. No more than 20.
            _guidance_scale: The guidance_scale of the baseline pipeline.
            _device: the device you want the pipeline to run on.
            _num_inference_steps: the inference_steps of the baseline pipeline.
            _resolution: resolution of image output.
            _batch_size: images per sample.
            
        Returns:
            None
        '''
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
        self.extra_step_kwargs = []
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
            
            #generate PIL used for benchmark
            for i in range(self.benchmark_sample_num):
                #first, generate prompt embeddings and save them
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
            
                #set scheduler
                tmpscheduler = copy.deepcopy(self.scheduler)
                tmpscheduler.set_timesteps(self.num_inference_steps, device=self.device)
                timesteps = tmpscheduler.timesteps
                #set unet
                num_channels_latents = baseline_unet.config.in_channels  
                generator = None 
                extra_step_kwargs = self.baseline_pipeline.prepare_extra_step_kwargs(generator, 0.0)   
                self.extra_step_kwargs.append(extra_step_kwargs)
                latents = self.baseline_pipeline.prepare_latents(self.batch_size,num_channels_latents,self.resolution,self.resolution,prompt_embeds.dtype, self.device, generator) 
                self.init_latents.append(latents)   
                output1 = copy.deepcopy(latents)
                
                #denoising
                for j, t in enumerate(timesteps):
                    tmp1 = gen(baseline_unet, output1, t, prompt_embeds,self.cross_attention_kwargs, self.guidance_scale)
                    output1 = tmpscheduler.step(tmp1, t, output1, **extra_step_kwargs).prev_sample
                    
                #store the samples
                out1=self.numpy_to_pil(self.decode_latents(output1))
                self.benchmark_samples.append(out1)
        #remove the baseline_pipeline from the CUDA device to conserve DRAM.
        self.baseline_pipeline.to('cpu')
        for tmp in self.prompt_embeddings:
            tmp.to('cpu')
        for tmp in self.init_latents:
            tmp.to('cpu')


    def acc_benchmark_single(self, my_unet):
        '''
        This benchmark exclusively evaluates a single U-Net by replacing the original U-Net in the pipeline.

        Args:
            my_unet: The U-Net to be assessed.

        Returns:
            Returns the accuracy score, which ranges from -1 to 1. Higher scores indicate better performance.
        '''
        my_unet.to(self.device)
        self.baseline_pipeline.vae.to('cuda')
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
                    tmp1 = gen(my_unet, output, t, prompt_embeds,self.cross_attention_kwargs, self.guidance_scale)
                    output = tmpscheduler.step(tmp1, t, output, **extra_step_kwargs).prev_sample 
                    
                out = self.numpy_to_pil(self.decode_latents(output))
                for j in range(self.batch_size):
                    outs.append(calculate_ssim(out[j], self.benchmark_samples[i][j]))
        self.baseline_pipeline.vae.to('cpu') 
        return find_median(outs)
        
    def acc_benchmark_pipeline(self, pipeline):
        '''
        This benchmark exclusively evaluates a pipeline. .

        Args:
            pipeline: The pipeline to be assessed. get the given prompt_embeds and initial latent, and output the generated latent.

        Returns:
            Returns the accuracy score, which ranges from -1 to 1. Higher scores indicate better performance.
        '''    
        self.baseline_pipeline.vae.to('cuda')    
        outs = []
        for i in range(self.benchmark_sample_num):
            latent = copy.deepcopy(self.init_latents[i])
            prompt_embeds = self.prompt_embeddings[i]
            extra_step_kwargs = self.extra_step_kwargs[i]
            with torch.no_grad():
                output = pipeline(prompt_embeds, latent, extra_step_kwargs) 
                out = self.numpy_to_pil(self.decode_latents(output))               
                for j in range(self.batch_size):
                    outs.append(calculate_ssim(out[j], self.benchmark_samples[i][j]))
        self.baseline_pipeline.vae.to('cpu') 
        return find_median(outs)
        
    def acc_benchmark_outputs(self, outputs):
        outs = []
        self.baseline_pipeline.vae.to('cuda')
        with torch.no_grad():
            len_outputs = len(outputs)
            for i in range(len_outputs):
                output=outputs[i].to('cuda')
                out = self.numpy_to_pil(self.decode_latents(output)) 
                output.to('cpu')
                for j in range(self.batch_size):
                    outs.append(calculate_ssim(out[j], self.benchmark_samples[i][j]))
        self.baseline_pipeline.vae.to('cpu')
        return find_median(outs)
    
    def gen_MinAccbench(self):
        return MinAccbench(self.benchmark_sample_num,
                           self.init_latents,
                           self.prompt_embeddings,
                           self.extra_step_kwargs,
                           self.benchmark_samples)
    

# here is an example
# if __name__ == "__main__":
#     model_path="../models/stable-diffusion-v1-4"
#     pipeline = DiffusionPipeline.from_pretrained(model_path,
#                                              torch_dtype=torch.float16,).to('cuda')
#     accbench = AccBenchmark(pipeline, _benchmark_sample_num = 20, _batch_size=2)
#     print("hey")
#     print(accbench.acc_benchmark(pipeline.unet))
            