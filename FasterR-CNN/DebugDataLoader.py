import time
import torch
from SmokeModel import SmokeModel
from torch.profiler import profile, record_function, ProfilerActivity

class DebugDataloader:
    def __init__(self, train_dataloader, validate_dataloader, test_dataloader, debug_dataloader):
        self.debug_dataloader = debug_dataloader
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # device agnostic

    def move_targets_to_device(self, targets, device, non_block):
        if(non_block):
            targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for
                       t in targets]
        else:
            targets = [{k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for
                       t in targets]
        return targets

    def check_pin(self, iterations=1):
        no_pin_times = []
        pin_times = []

        for iter in range(iterations):
            print(f"Iteration {iter + 1}/{iterations}")

            # Without non_blocking
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_no_pin'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
            ) as prof_no_pin:
                start = time.time()
                for batch, (image_tensor, target) in enumerate(self.debug_dataloader):
                    with record_function("Data transfer (no pin_memory)"):
                        image_tensors = list(tensor.to(self.device) for tensor in image_tensor)
                        targets = self.move_targets_to_device(target, self.device, False)
                no_pin_times.append(time.time() - start)

            print(prof_no_pin.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            # With non_blocking
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_with_pin'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
            ) as prof_with_pin:
                start = time.time()
                for batch, (image_tensor, target) in enumerate(self.train_dataloader):
                    with record_function("Data transfer (pin_memory)"):
                        image_tensors = list(tensor.to(self.device, non_blocking=True) for tensor in image_tensor)
                        targets = self.move_targets_to_device(target, self.device, True)
                pin_times.append(time.time() - start)

            print(prof_with_pin.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            #tensorboard --logdir=./

        avg_no_pin_time = sum(no_pin_times) / len(no_pin_times)
        avg_pin_time = sum(pin_times) / len(pin_times)

        improvement = ((avg_no_pin_time - avg_pin_time) / avg_no_pin_time) * 100
        print(f"Average transfer time (no pin_memory): {avg_no_pin_time:.6f} seconds")
        print(f"Average transfer time (with pin_memory): {avg_pin_time:.6f} seconds")
        print(f"Average performance improvement: {improvement:.2f}%")

    def check_dataloader(self, test_runs = 1):
        for test_runs in range(test_runs):
            # Iterate through entire dataloader
            for batch_idx, (images, targets) in enumerate(self.train_dataloader):
                print(f"Batch {batch_idx}:")
            for batch_idx, (images, targets) in enumerate(self.validate_dataloader):
                print(f"Batch {batch_idx}:")
            for batch_idx, (images, targets) in enumerate(self.test_dataloader):
                print(f"Batch {batch_idx}:")

def main():
    # Setting variables for instance
    device = "cuda" if torch.cuda.is_available() else "cpu"  # device agnostic
    # Create instance of SmokeModel class
    smoke_model = SmokeModel()
    # get dataloaders from model class
    train_dataloader, validate_dataloader, test_dataloader = smoke_model.get_dataloader()
    debug_dataloader = smoke_model.get_debug_dataloader()
    # create instance for debugging dataloaders
    debug_data_loader = DebugDataloader(train_dataloader, validate_dataloader, test_dataloader, debug_dataloader)

    # CALL FUNCTIONS TO CHECK DATALOADER
    #debug_data_loader.check_dataloader()

    # CHECK PINNING PERFORMANCE
    debug_data_loader.check_pin()



# only run if this script is being run (not being called from other)
if __name__ == '__main__':
    main()