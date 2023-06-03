import torch
import multiprocessing as mp
from typing import List, Dict, Any, Callable


class Worker(mp.Process):
    def __init__(self, worker_id: int, task_queue: mp.Queue, output_dict, device: str, task: Callable[[Any, str, Dict], Any], shared_object: Dict, done_event: mp.Event):
        super().__init__()
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.output_dict = output_dict
        self.device = device
        self.task = task
        self.shared_object = shared_object
        self.done_event = done_event

    def run(self):
        while True:
            task_id, task_input = self.task_queue.get()
            if task_id is None:
                if task_input is not None:  # End signal for a batch of tasks
                    self.done_event.set()
                    while self.done_event.is_set():  # Wait until the event is cleared
                        pass
                else:  # End signal for this worker
                    self.done_event.set()
                    break
            else:
                try:
                    self.output_dict[task_id] = self.task(task_input, self.device, self.shared_object)
                except Exception:
                    self.output_dict[task_id] = None

class TaskScheduler:
    def __init__(self, task: Callable[[Any, str, Dict], Any], shared_object: Dict = {}):
        try:
            mp.set_start_method('spawn', force=True)
            print("spawned")
        except RuntimeError:
            pass
        self.devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
        self.task = task
        self.manager = mp.Manager()
        self.shared_object = self.manager.dict(shared_object)
        self.output_dict = self.manager.dict()
        self.task_queue = mp.Queue()
        self.done_events = [mp.Event() for _ in range(len(self.devices))]
        self.workers = [Worker(i, self.task_queue, self.output_dict, self.devices[i], task, self.shared_object, self.done_events[i]) for i in range(len(self.devices))]
        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def submit(self, tasks: List):
        for i, task_input in enumerate(tasks):
            self.task_queue.put((i, task_input))
        # Signal the end of a batch of tasks
        for _ in self.workers:
            self.task_queue.put((None, 1))
        for event in self.done_events:  # Wait for all workers to finish
            event.wait()
        # Clear the events for the next batch of tasks
        for event in self.done_events:
            event.clear()
        return [self.output_dict[i] for i in range(len(tasks))]

    def close(self):
        for _ in self.workers:
            self.task_queue.put((None, None))
        for worker in self.workers:
            worker.join()

    def __del__(self):
        self.close()




