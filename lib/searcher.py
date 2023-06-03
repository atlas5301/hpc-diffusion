import torch
import diffusers
from diffusers import DiffusionPipeline
from diffusers import DPMSolverSinglestepScheduler
from typing import Any, Callable, Set, Dict
import copy
import random
import numpy as np
import io

# from .gen_pipeline import PipelineGenerator
# from .acc_bench import AccBenchmark
from . import acc_bench
from . import gen_pipeline
from . import cuda_scheduler
def serialize(object):
    buffer = io.BytesIO()
    torch.save(object, buffer)
    return buffer.getvalue()

def deserialize(object):
    buffer = io.BytesIO(object)
    return torch.load(buffer)

def random_vector(max_n, max_value=50, num_values=6, similarity_factor=0.5):
    # Generate a list of all possible unique values
    all_values = list(range(1, max_value + 1))
    
    # Randomly select some of the unique values
    unique_values = sorted(random.sample(all_values, num_values))
    
    # Randomly select the number of elements in the vector
    n = random.randint(1, max_n)
    
    # Generate the random vector
    vector = [random.choice(unique_values) for _ in range(n)]
    
    # Sort the vector in ascending order
    vector.sort()
    
    # Adjust the order of the values based on the similarity factor
    for _ in range(int((1 - similarity_factor) * n)):
        i, j = random.sample(range(n), 2)
        vector[i], vector[j] = vector[j], vector[i]
    
    return vector

def task(pipeline_info, device, shared_object):
    with torch.no_grad():
        # try:
            generator:gen_pipeline.PipelineGenerator = shared_object['generator']
            benchmark:acc_bench.MinAccbench = shared_object['benchmark'][device]
            keys = sorted(generator.models.keys())
            pipeline_info = [keys[i] for i in pipeline_info]
            pipeline = generator.generate(pipeline_info)
            pipeline.move_to_device(device)
            result = benchmark.acc_benchmark_pipeline(pipeline, device)
            pipeline.move_to_device('cpu')
        # except Exception as e:
        #     with open('debug3.txt', "w") as f:
        #         f.write(str(e)) 
    return serialize(result)

class PipelineDescription(object):
    pipeline_info: list
    accuracy: float
    end2end_time: float
    memory_use: int
    
    def __init__(self, _pipeline_info, _accuracy, _end2end_time, _memory_use) -> None:
        self.pipeline_info = _pipeline_info
        self.accuracy = _accuracy
        self.end2end_time = _end2end_time
        self.memory_use = _memory_use

class PipelineSearcher(object):
    benchmark: acc_bench.AccBenchmark = None
    generator: gen_pipeline.PipelineGenerator = None
    taskscheduler: cuda_scheduler.TaskScheduler = None
    
    pipelines: list[PipelineDescription] = [] 
    #stores 'dicts of {"info":pipeline_info, "acc":accuracy, "time":end2end time}', only for first few elements
    
    
    def __init__(self,
                _baseline_pipeline:DiffusionPipeline,
                _scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = DPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012),
                _benchmark_sample_num = 20,
                _guidance_scale = 8.0,
                _device = torch.device('cuda'),
                _device_memory = 32768,
                _num_inference_steps:int = 50,
                _resolution:int =512,
                _batch_size: int = 1,
                _injected_models = {}
                ) -> None:
        
        self.generator = gen_pipeline.PipelineGenerator(
            _scheduler= _scheduler,
            _guidance_scale= _guidance_scale,
            _device_memory= _device_memory,
            _injected_models= _injected_models
        )
        self.generator.models.update(_injected_models)
        
        self.benchmark = acc_bench.AccBenchmark(
            _baseline_pipeline= _baseline_pipeline,
            _scheduler= _scheduler,
            _benchmark_sample_num= _benchmark_sample_num,
            _guidance_scale= _guidance_scale,
            _device= _device,
            _num_inference_steps= _num_inference_steps,
            _resolution= _resolution,
            _batch_size= _batch_size
        )
        
        
    def scheduler_init(self):
        self.pipelines = []
        
        tmpdevices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
        tmpdict = {}
        for device in tmpdevices:
            tmpdict[device] = self.benchmark.gen_MinAccbench()
        
        self.taskscheduler = cuda_scheduler.TaskScheduler(task, {'benchmark':tmpdict, 'generator':self.generator})
        
    def generate_random_pipeline_info(self, M=6, K=30, temperature=0.4):
        N = len(self.generator.models)
        M = min(N, M)
        # Determine the actual number of unique values and the actual length of the list
        # Use squaring to give higher weight to larger numbers
        actual_M = int(M * (random.random() ** 0.5)) + 1
        actual_K = int(K * (random.random() ** 0.5)) + 1

        # Generate actual_M unique values between 1 and N, and a list of size actual_K from these values
        unique_values = random.sample(range(0, N), actual_M)
        random_list = [random.choice(unique_values) for _ in range(actual_K)]
        
        # Sort the list
        random_list.sort()
        
        # Apply Fisher-Yates shuffle with bias based on temperature
        for i in range(len(random_list)):
            # Pick an index from i to len(random_list), biased by temperature
            weights = [(j - i + 1) ** temperature for j in range(i, len(random_list))]
            weights = np.array(weights) / sum(weights)
            j = np.random.choice(range(i, len(random_list)), p=weights)
            
            # Swap elements at i and j
            random_list[i], random_list[j] = random_list[j], random_list[i]
        
        # return [keys[j] for j in random_list]
        return random_list 

    def mutate(self, lst, delete_weight=3, replace_weight=1):
        operation = random.choices(['delete', 'replace'], weights=[delete_weight, replace_weight], k=1)[0]
        lambda_parameter = 1 / 10  # On average, we modify 10% of the elements
        if operation == 'delete':
            num_to_delete = min(len(lst), max(1, int(np.random.exponential(1 / lambda_parameter))))
            delete_vals = random.sample(lst, num_to_delete)
            lst = [x for x in lst if x not in delete_vals]
        else: # operation == 'replace'
            unique_vals = list(set(lst))
            num_to_replace = min(len(unique_vals), max(1, int(np.random.exponential(1 / lambda_parameter))))
            old_vals = random.sample(unique_vals, num_to_replace)
            new_vals = random.choices(range(0, len(self.generator.models)), k=num_to_replace)
            replace_dict = dict(zip(old_vals, new_vals))
            lst = [replace_dict.get(x, x) for x in lst]
        return lst


    def validate_pipeline_info(self, pipeline_info, time_limit) -> bool:
        if (pipeline_info == None):
            return False
        if (not pipeline_info):
            return False
        keys = sorted(self.generator.models.keys())
        new_pipeline_info = [keys[i] for i in pipeline_info]
        device_model = set()
        current_memory = 0
        end2end_time = 0
        for name in new_pipeline_info:
            if (name in device_model):
                pass
            else:
                if (name in self.generator.models):
                    device_model.add(name)
                    current_memory += self.generator.models[name].size
                    if (current_memory > self.generator.utils.device_memory):
                        return False
                else:
                    return False
            end2end_time += self.generator.models[name].end2endtime
            if (end2end_time > time_limit):
                return False
        return True

    def gen_pipeline_description(self, pipeline_info, accuracy) -> PipelineDescription:
        keys = sorted(self.generator.models.keys())
        new_pipeline_info = [keys[i] for i in pipeline_info]
        device_model = set()
        current_memory = 0
        end2end_time = 0
        for name in new_pipeline_info:
            if (name in device_model):
                pass
            else:
                device_model.add(name)
                current_memory += self.generator.models[name].size
            end2end_time += self.generator.models[name].end2endtime
        
        return PipelineDescription(pipeline_info, accuracy, end2end_time, current_memory)      
        
    def result_process(self, results):
        tmplist = []
        for output in results:
            if (output!=None):
                output = deserialize(output)
                if ((output!=None) and output):
                    tmplist.append(self.benchmark.acc_benchmark_outputs(output))
                else:
                    tmplist.append(None)
            else:
                tmplist.append(None)
        return tmplist

    def searcher_run_once(
        self,
        time_limit = 10.0,
        expected_accuracy = 0.95,
        num_candidates = 120,   
        mutate_rate = 0.7,
        temperature = 0.4,
        M = 6,
        K = 30,
        mutate_delete_weight = 0.7,
        max_tries_random = 10000,
    ):
        new_pipelines = []
        num_mutate = int(min(mutate_rate*num_candidates, len(self.pipelines)))
        current_time_limit = time_limit
        if (num_mutate == int(mutate_rate*num_candidates)):
            if (num_mutate > 0):
                current_time_limit = min(time_limit, self.pipelines[num_mutate-1].end2end_time)
            
        for i in range(num_mutate):
            new_pipelines.append(self.mutate(self.pipelines[i].pipeline_info, mutate_delete_weight, 1-mutate_delete_weight))
            
        num_new = num_candidates - num_mutate
        for i in range(num_new):
            tmp = []
            for i in range(max_tries_random):
                tmp = self.generate_random_pipeline_info(M, K, temperature)
                if (self.validate_pipeline_info(tmp, current_time_limit)):
                    new_pipelines.append(tmp)
                    break
        
        
        results = self.taskscheduler.submit(new_pipelines)
        with open('debug2.txt', "w") as f:
            f.write(str(results))
        results = self.result_process(results)
        len_results = len(results)
        for i in range(len_results):
            if (results[i] == None):
                continue
            if (results[i] > expected_accuracy):
                self.pipelines.append(self.gen_pipeline_description(new_pipelines[i], results[i]))
                
        self.pipelines = sorted(self.pipelines, key= lambda obj: obj.end2end_time)[:num_candidates]
        
    # def result_process(self, results):
    #     return [self.benchmark.acc_benchmark_outputs()]
        
    def inject_pipelines(self, pipeline_info_list: list, expected_accuracy = 0.95, num_candidates = 120):
        results = self.taskscheduler.submit(pipeline_info_list)
        len_results = len(results)
        with open('debug.txt', "w") as f:
            f.write(str(results))
            
        results = self.result_process(results)
            
        for i in range(len_results):
            if (results[i] != None):
                if (results[i] > expected_accuracy):
                    self.pipelines.append(self.gen_pipeline_description(pipeline_info_list[i], results[i]))
                
        self.pipelines = sorted(self.pipelines, key= lambda obj: obj.end2end_time)[:num_candidates]        
    
            
    def search(
        self,
        num_rounds = 5,
        time_limit = 10.0,
        expected_accuracy = 0.95,
        num_candidates = 120,   
        mutate_rate = 0.7,
        temperature = 0.4,
        M = 6,
        K = 30,
        mutate_delete_weight = 0.7,
        max_tries_random = 10000,        
    ):
        for i in range(num_rounds):
            self.searcher_run_once(
                time_limit= time_limit,
                expected_accuracy= expected_accuracy,
                num_candidates=num_candidates,   
                mutate_rate= mutate_rate,
                temperature= temperature,
                M= M,
                K= K,
                mutate_delete_weight= mutate_delete_weight,
                max_tries_random= max_tries_random,                 
            )
            if (self.pipelines):
                print('current_best:', self.pipelines[0])
            else:
                print('no candidates found!!')
        if (self.pipelines):
            return self.pipelines[0]
        else:
            return None       
            
                
    def load(self, model_json_path, config_json_path):
        self.generator.load_models(model_json_path, config_json_path)           
            
        
        



