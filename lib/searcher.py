import torch
import diffusers
from diffusers import DiffusionPipeline
from diffusers import DPMSolverSinglestepScheduler
from typing import Any, Callable
import copy

from . import PipelineGenerator
from . import AccBenchmark


class PipelineSearcher(object):
    benchmark: AccBenchmark = None
    generator: PipelineGenerator = None
    
    
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
        
        self.generator = PipelineGenerator(
            _scheduler = _scheduler,
            _guidance_scale = _guidance_scale,
            _device = _device,
        )
        
        self.benchmark = AccBenchmark(
            _baseline_pipeline = _baseline_pipeline,
            _scheduler = _scheduler,
            _benchmark_sample_num = _benchmark_sample_num,
            _guidance_scale = _guidance_scale,
            _device = _device,
            _num_inference_steps = _num_inference_steps,
            _resolution = _resolution,
            _batch_size = _batch_size
        )
