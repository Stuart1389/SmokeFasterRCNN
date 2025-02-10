### Testing model predictions
import os
#os.environ["OMP_NUM_THREADS"] = "8"
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import copy
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.benchmark as benchmark
import concurrent.futures
from GetValues import checkColab, setTestValues, setGlobalValues
from SmokeModel import SmokeModel
from tabulate import tabulate
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
import torch.ao.quantization as quantization
import numpy as np
from SmokeUtils import get_layers_to_fuse
from torch.nn.utils import prune
from torch.ao.pruning import WeightNormSparsifier
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from simplify import simplify
from super_image import PanModel
from multiprocessing import Pool
import matplotlib.patches as mpatches
import torchvision.transforms.functional as F


#DELETE
import collections
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import shutil
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.ao.pruning import WeightNormSparsifier
import torch_pruning as tp
SparseSemiStructuredTensor._FORCE_CUTLASS = True

class Tester:
    #Constructor
    def __init__(self):

        # Initialising instanced variables
        self.base_dir = checkColab()
        # scores over this will be counted towards mAP/precission/recall and will be displayed if plot
        self.confidence_threshold = 0.5
        self.benchmark = True # measure how long it takes to make average prediction
        self.ap_value = 0.5 # ap value for precision/recall e.g. if 0.5 then iou > 50% overlap = true positive

        #PLOT MAIN IMAGE
        self.draw_highest_only = False # only draw bbox with highest score on plot
        self.plot_image = False # plot images
        self.save_plots = False # save plots to model folder/plots
        self.plot_ground_truth = True # whether to plot ground truth
        self.draw_no_true_positive_only = False # only plot images with no true positives

        #SPLIT IMAGE
        self.plot_split_images = False # if using partitioned/split images, whether to display each split
        self.save_split_image_plots = False
        self.combine_bboxes = False # merge touching bbox predictions when splitting image

        # RESIZE / SCALE GROUND TRUTH
        self.use_scale = False
        self.scale_height = 224
        self.scale_width = 224

        if(self.use_scale == False):
            self.scale_height = None
            self.scale_width = None

        # device agnostic code
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # device agnostic

        self.batch_size = setTestValues("BATCH_SIZE")
        self.model_name = setTestValues("model_name")
        self.save_path = Path("savedModels/" + self.model_name)


        # initialise model
        smoke_model = SmokeModel()
        if(setTestValues("load_QAT_model")):
            # can only load qat state dict once model has been prepped
            self.model, self.qat_state_dict = smoke_model.get_model(True)
        else:
            self.model = smoke_model.get_model(True)
        print(f"Number of parameters: {self.count_parameters(self.model)}")
        #print(self.model.rpn)

        # get test dataloader
        _, _, self.test_dataloader = smoke_model.get_dataloader(True)
        self.validate_dataloader = smoke_model.get_validate_test_dataloader()

        # Paths
        if(setTestValues("test_on_val")):
            self.test_image_dir = Path(f"{self.base_dir}/Dataset/") / setTestValues("dataset") / "Validate/images"
            self.test_annot_dir = Path(f"{self.base_dir}/Dataset/") / setTestValues("dataset") / "Validate/annotations/xmls"
        else:
            self.test_image_dir = Path(f"{self.base_dir}/Dataset/") / setTestValues("dataset") / "Test/images"
            self.test_annot_dir = Path(f"{self.base_dir}/Dataset/") / setTestValues("dataset") / "Test/annotations/xmls"

        # Metrics
        self.initialise_metrics()

        # Benchmark times
        self.benchmark_times = []

        # Total time
        self.start_time = None

        # Precision/recall counters
        self.total_tp = {"small": 0, "medium": 0, "large": 0, "global": 0}
        self.total_fp = {"small": 0, "medium": 0, "large": 0, "global": 0}
        self.total_fn = {"small": 0, "medium": 0, "large": 0, "global": 0}

        # dictionary containing ground truth sizes
        self.image_gt_size = self.get_ground_truth_size(self.test_image_dir, self.test_annot_dir)
        #print(self.image_gt_size)

        self.cur_batch = None
        self.model_name = setTestValues("model_name")
        self.start_profiler = setTestValues("start_profiler")
        self.record_trace = setTestValues("record_trace")
        self.non_blocking = setTestValues("non_blocking")
        # quants
        self.static_quants = setTestValues("static_quant")
        self.calibrate_full_set = setTestValues("calibrate_full_set")
        # pytorch only supports static quants on cpu
        if(self.static_quants or setTestValues("CPU_inference")):
            self.device = "cpu"
        self.half_precission = setTestValues("half_precission")

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # !!START TESTING CHAIN!!
    # function starts testing images
    def test_dir(self):
        profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.save_path / 'profiler_test_output'),
                record_shapes=False,
                profile_memory=False,
                with_stack=self.record_trace,
                with_flops=False,
                with_modules=False
        )
        if(self.start_profiler):
            profiler.start()
        test_dataloader = self.test_dataloader # change to dataloader
        # getting predictions
        self.model.to(self.device, non_blocking=self.non_blocking)  # put model on cpu or gpu
        # set model to evaluation mode
        self.model.eval()
        #fp16
        if(self.half_precission):
            self.model = self.model.cuda().half()

        # Static quantization
        #print(self.model)
        #print(self.model.backbone.body)
        #print(self.model.backbone)

        # QUANTS!
        if(self.static_quants):
            self.quant_model()

        #self.model = self.model.cuda().half()
        if(setTestValues("prune_model")):
            self.prune_model()

        # delete currently saved plots in model folder
        # and create new dir for plots
        if(self.save_plots):
            self.make_plot_dir()

        if(self.save_split_image_plots):
            self.make_plot_dir(True)



        # Start timer
        self.start_time = time.time()
        # Walks through dir_path, for each file call predictBbox and pass file path
        for batch, (image_tensor, filename) in enumerate(test_dataloader):
            print(f"Processing batch {batch} out of {len(test_dataloader)}")
            self.get_predictions(image_tensor, filename)
            #profiler.step()
        self.get_results()
        if(self.start_profiler):
            profiler.stop()

    def make_plot_dir(self, split_images = False):
        self.plot_save_path = self.save_path / "plots"
        self.split_plot_save_path = self.save_path / "split_plots"
        if (split_images):
            save_path = self.split_plot_save_path
        else:
            save_path = self.plot_save_path
        print("Plot save path:", save_path)
        if(save_path.exists()):
            shutil.rmtree(save_path)
        save_path.mkdir(exist_ok=False)

    def prune_model(self):
        """
        pruning in pytorch is very experimental, benefits are limited, most performance gains
        are achieved through  libraries e.g. nvidia apex, tensort.
        pytorch has articles on 2:4 sparsity but this isnt really applicable to conv
        and has a bunch of constraints, based on what ive found during development
        pruning can be quite good for vision transformers but isnt very beneficial for cnns

        _________________________________________
        list of conv layers in resnet 50 common to use middle layers, e.g. 3
        conv layers dont benefit much from current sparsity implementations unfortunately
        """
        #print(self.model.backbone.body)
        layers_to_prune = [
            self.model.backbone.body.layer3[0].conv1,
            self.model.backbone.body.layer3[0].conv2,
            self.model.backbone.body.layer3[0].conv3,
            self.model.backbone.body.layer3[1].conv1,
            self.model.backbone.body.layer3[1].conv2,
            self.model.backbone.body.layer3[1].conv3,
            self.model.backbone.body.layer3[2].conv1,
            self.model.backbone.body.layer3[2].conv2,
            self.model.backbone.body.layer3[2].conv3,
        ]

        tensor_type = setTestValues("tensor_type")
        prune_amount = setTestValues("prune_amount")

        for module in layers_to_prune:
            if(setTestValues("unstructured")):
                prune.l1_unstructured(module, name="weight", amount=prune_amount)
            else:
                # dim 0 = filters/output channels, dim 1 = input channels
                # n = norm, n=1 - l1, n=2 - l2
                prune.ln_structured(module, name="weight", amount=prune_amount, dim=1, n=1)

            with torch.no_grad():
                weight_tensor = module.weight
                print("Dense weight tensor:", weight_tensor)

                if(tensor_type == "coo"):
                    indices = torch.nonzero(weight_tensor, as_tuple=True)  # coords
                    values = weight_tensor[indices]  # value
                    size = weight_tensor.size()  # getting shape

                    # converting to coo tensor
                    sparse_weight = torch.sparse_coo_tensor(
                        indices=torch.stack(indices),
                        values=values,
                        size=size,
                        dtype=weight_tensor.dtype,
                        device=weight_tensor.device
                    )
                elif tensor_type == "csr": # compressed sparse row tensor, better if there are more rows than columns
                    # get values other than 0
                    indices = torch.nonzero(weight_tensor, as_tuple=False)
                    # csr is built using rows and cols
                    rows, cols = indices[:, 0], indices[:, 1]
                    values = weight_tensor[rows, cols]
                    size = weight_tensor.size()
                    crow_indices = torch.zeros(size[0] + 1, dtype=torch.int32, device=weight_tensor.device)
                    crow_indices[1:] = torch.cumsum(torch.bincount(rows, minlength=size[0]), dim=0)

                    # create compressed sparse row tensor
                    sparse_weight = torch.sparse_csr_tensor(
                        crow_indices=crow_indices,
                        col_indices=cols,
                        values=values,
                        size=size,
                        dtype=weight_tensor.dtype,
                        device=weight_tensor.device
                    )

                prune.remove(module, name="weight") # remove hooks
                print("Sparse weight tensor:", sparse_weight)

    def quant_model(self):
        print(f"Threads available: {torch.get_num_threads()}")
        # torch.set_num_threads(torch.get_num_threads())
        module_names = list(self.model.backbone.body.named_modules())
        self.model.backbone.body.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        layers_to_fuse = get_layers_to_fuse(module_names)
        self.model.backbone.body = torch.ao.quantization.fuse_modules(self.model.backbone.body, layers_to_fuse)
        self.model.backbone.body = torch.ao.quantization.prepare(self.model.backbone.body, inplace=False)
        if (self.calibrate_full_set):
            start_time = time.time()
            for batch, (image_tensor, filename) in enumerate(self.validate_dataloader):  # or self.test_dataloader
                print(f"Calibrating batch {batch} out of {len(self.validate_dataloader)}")
                image_tensors = torch.stack(
                    [tensor.to(self.device, non_blocking=self.non_blocking) for tensor in image_tensor]
                )
                self.model.backbone.body(image_tensors)
            end_time = time.time()
            print(f"Calibration took: {end_time - start_time:.2f} seconds")
        else:
            image_tensor, filename = next(iter(self.validate_dataloader))  # quant calibration
            self.model.backbone.body(image_tensor[0].unsqueeze(0))

        self.model.backbone.body = torch.ao.quantization.convert(self.model.backbone.body)
        if (setTestValues("load_QAT_model")):
            # load quant aware state dict after converting model
            self.model.load_state_dict(self.qat_state_dict)


    # Transform images for model
    def get_transform(self):
        transforms = []
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)

    # function to calculate the average time to make a prediction
    def get_avg_time(self, benchmark_times):
        total_time = sum(benchmark_times)  # get sum
        avg_time = total_time / len(benchmark_times)  # get average
        return avg_time

    def upscale_images(self, image_tensors):
        combined_tensor = torch.stack(image_tensors, dim=0).to(self.device)
        upscale_model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=setTestValues("upscale_value"))
        upscale_model.to(self.device)
        upscale_outputs = upscale_model(combined_tensor)
        # undo stack for faster rcnn input
        formatted_tensors = list(torch.unbind(upscale_outputs, dim=0))
        return formatted_tensors

    def split_image(self, input_tensor):
        split_tensors = []

        for tensor in input_tensor:
            _, H, W = tensor.shape # c, w, h

            H_mid, W_mid = H // 2, W // 2

            # Split the tensor into 4 parts
            part1 = tensor[:, :H_mid, :W_mid]  # Top left
            part2 = tensor[:, :H_mid, W_mid:]  # Top right
            part3 = tensor[:, H_mid:, :W_mid]  # Bot left
            part4 = tensor[:, H_mid:, W_mid:]  # Bot right

            #batch = torch.stack([part1, part2, part3, part4], dim=0)
            #print("batch_shape",batch.shape)
            split_tensors.append([part1, part2, part3, part4])
        #result = torch.cat(batches, dim=0)
        return split_tensors

    def display_split_images(self, split_images, pred_dict, filename):
        part1, part2, part3, part4 = split_images
        # display split parts in matplotlib
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        for i, part in enumerate([part1, part2, part3, part4]):
            boxes_over_thresh = []
            labels_over_thresh = []
            scores_over_thresh = []
            for box, label, score in zip(pred_dict[i]["boxes"], pred_dict[i]["labels"], pred_dict[i]["scores"]):
                if(score > self.confidence_threshold):
                    print("score",score)
                    boxes_over_thresh.append(box)
                    labels_over_thresh.append(label)
                    scores_over_thresh.append(score)

            labels_str = ["Smoke" if label.item() == 1 else str(label.item()) for label in labels_over_thresh]
            boxes_tensor = torch.stack(boxes_over_thresh) if boxes_over_thresh else torch.empty((0, 4))
            output_image = draw_bounding_boxes(part, boxes_tensor, labels_str,
                                               colors="red")

            row, col = i // 2, i % 2
            output_image_np = F.to_pil_image(output_image).convert("RGB")
            axs[row, col].imshow(output_image_np)
            axs[row, col].axis('off')
            axs[row, col].set_title(f'Part {i + 1}')

        plt.tight_layout()
        if (self.save_split_image_plots):
            plot_save_path = self.split_plot_save_path / filename
            plt.savefig(plot_save_path, bbox_inches="tight")
        if(self.plot_split_images):
            plt.show()

    def adjust_boxes(self, boxes, offset_x, offset_y):
        # move bbox based on position in split
        adjusted_boxes = boxes.clone()
        adjusted_boxes[:, 0] += offset_x  # x_min
        adjusted_boxes[:, 2] += offset_x  # x_max
        adjusted_boxes[:, 1] += offset_y  # y_min
        adjusted_boxes[:, 3] += offset_y  # y_max
        return adjusted_boxes


    def int_prediction(self, image_tensor, outputs, filenames):
        # parallel processing after getting model predictions, might aswell
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for image_tensor, prediction, filename in zip(image_tensor, outputs, filenames):
                # print(f"{temp_index}! - pred:", prediction, filename)
                # temp_index += 1
                futures.append(executor.submit(self.process_predictions, image_tensor, prediction, filename))
                # self.process_predictions(image_tensor, prediction, filename)
        # print("outputs", outputs)
        concurrent.futures.wait(futures)

        if(self.plot_image or self.save_plots):
            # plt not thread safe, originally used multiprocess but it caused a bunch of overhead
            image_vls = [future.result() for future in futures]

            for result in image_vls:
                filtered_labels, filtered_boxes, filtered_scores, image, ground_truth, metrics, filename = result
                if (self.draw_no_true_positive_only):
                    if (metrics["TP"] == 0):
                        # call function to display image with overlayed bboxes
                        self.display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth,
                                                metrics,
                                                filename)
                else:
                    # call function to display image with overlayed bboxes
                    self.display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth,
                                            metrics,
                                            filename)

    def get_model_outputs(self, tensor_list, full_image_tensors = None):
        image_tensors = list(tensor.to(self.device, non_blocking=self.non_blocking) for tensor in tensor_list)

        # half precission
        if self.half_precission:
            image_tensors = [tensor.cuda().half() for tensor in image_tensors]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            # benchmarking
            if self.benchmark == True:
                # start torch.utils.benchmark
                # will run the below code a second time to measure performance
                timer = benchmark.Timer(
                    stmt="model(image_tensors)",  # specify code to be benchmarked
                    globals={"image_tensors": image_tensors, "model": self.model}
                    # pass x and model to be used by benchmark
                )

                # record time taken
                time_taken = timer.timeit(3)  # run code n times, gives average = time taken / n
                time_taken_ind = time_taken.mean / self.batch_size
                print(f"Prediction time taken: {time_taken_ind:.4f} seconds")
                self.benchmark_times.append(time_taken_ind)
            # outputs = self.model(image_tensors)
            outputs, _, _, _, _, _ = self.model(image_tensors)

            if(setTestValues("split_images")):
                # offsets
                # print(image_tensor[0].shape[1:])
                full_image_height, full_image_width = full_image_tensors[0].shape[1:]  # get  image dims
                H_mid, W_mid = full_image_height // 2, full_image_width // 2  # midpoints

                # Mapping offsets for the splits
                offsets = [
                    (0, 0),  # top left
                    (W_mid, 0),  # top right
                    (0, H_mid),  # bot left
                    (W_mid, H_mid)  # bot right
                ]

                # Accumulate outputs
                temp_combined = {
                    'boxes': torch.empty((0, 4), device='cuda:0'),
                    'labels': torch.empty((0,), dtype=torch.int64, device='cuda:0'),
                    'scores': torch.empty((0,), device='cuda:0')
                }

                vis_split = []
                #print(outputs)
                i = 0
                for output in outputs:
                    # combined dict for using with the single combined image
                    offset_x, offset_y = offsets[i]
                    adjusted_boxes = self.adjust_boxes(output['boxes'], offset_x, offset_y)
                    temp_combined['boxes'] = torch.cat((temp_combined['boxes'], adjusted_boxes), dim=0)
                    temp_combined['labels'] = torch.cat((temp_combined['labels'], output['labels']), dim=0)
                    temp_combined['scores'] = torch.cat((temp_combined['scores'], output['scores']), dim=0)

                    if(self.combine_bboxes):
                        temp_combined = self.merge_touching_boxes(temp_combined)

                    # split dicts for visualising split images
                    vis_split.append({
                        'boxes': output['boxes'],
                        'labels': output['labels'],
                        'scores': output['scores']
                    })
                    # tells us what part of the split we're on
                    i+=1

                return temp_combined, vis_split
            else:
                return  outputs

    def merge_touching_boxes(self, temp_combined, tolerance=2):
        boxes, labels, scores = temp_combined['boxes'], temp_combined['labels'], temp_combined['scores']
        # bool indexing to get bboxes with score higher than threshold
        score_thresh = scores > self.confidence_threshold
        boxes, labels, scores = boxes[score_thresh], labels[score_thresh], scores[score_thresh]

        # only use boxes where temp_combined['scores'] > self.confidence_threshold

        if boxes.shape[0] == 0:
            return temp_combined

        merged_boxes = []
        merged_labels = []
        merged_scores = []
        used = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)

        for i in range(len(boxes)):
            if used[i]:  # Skip already merged boxes
                continue
            x1, y1, x2, y2 = boxes[i]
            label = labels[i]
            score = scores[i]

            for j in range(i + 1, len(boxes)):
                if used[j] or labels[j] != label:
                    continue

                x1_j, y1_j, x2_j, y2_j = boxes[j]

                # Check if bbox edge is within tol

                horizontally_touching = (
                        (abs(x2 - x1_j) <= tolerance or abs(x2_j - x1) <= tolerance)
                        and (y1 < y2_j and y2 > y1_j)
                )
                vertically_touching = (
                        (abs(y2 - y1_j) <= tolerance or abs(y2_j - y1) <= tolerance)
                        and (x1 < x2_j and x2 > x1_j)
                )
                # Merge boxes
                if horizontally_touching or vertically_touching:
                    x1 = min(x1, x1_j)
                    y1 = min(y1, y1_j)
                    x2 = max(x2, x2_j)
                    y2 = max(y2, y2_j)

                    # sets box score to higherst val
                    score = max(score, scores[j])
                    # dont merge boxes multiple times
                    used[j] = True

            merged_boxes.append([x1, y1, x2, y2])
            merged_labels.append(label)
            merged_scores.append(score)

        temp_combined['boxes'] = torch.tensor(merged_boxes, device=boxes.device)
        temp_combined['labels'] = torch.tensor(merged_labels, dtype=torch.int64, device=boxes.device)
        temp_combined['scores'] = torch.tensor(merged_scores, device=boxes.device)

        return temp_combined

    # !!GETTING PREDICTION!!
    @torch.inference_mode()
    def get_predictions(self, image_tensor, filename):
        #print("image tensor type", type(image_tensor))
        with torch.profiler.record_function("TESTING"):
            if(setTestValues("upscale_image")):
                image_tensor = self.upscale_images(image_tensor)

            image_tensor_split = self.split_image(image_tensor)
            #print("image tensor type after", type(image_tensor))
            # print("image:", image, "image_tensor", image_tensor, "filename", filename)
            filenames = list(files for files in filename)
            image_tensors = list(tensor.to(self.device, non_blocking=self.non_blocking) for tensor in image_tensor)

            combined_outputs = []
            #combined_outputs.append(temp_combined)
            if(setTestValues("split_images")):
                for item, fname in zip(image_tensor_split, filename):
                    temp_combined, vis_split = self.get_model_outputs(item, image_tensor)
                    if(self.plot_split_images or self.save_split_image_plots):
                        self.display_split_images(item, vis_split, fname)
                    combined_outputs.append(temp_combined)
            else:
                combined_outputs = self.get_model_outputs(image_tensor)

            self.int_prediction(image_tensor, combined_outputs, filename)

    def process_predictions(self, image_tensor, predictions, filename):
        global total_tp, total_fp, total_fn
        # Normalize the image
        #x = image_tensor
        image = (255.0 * (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())).to(torch.uint8)
        # Filter predictions based on confidence score
        filtered_labels = []
        filtered_boxes = []
        filtered_scores = []
        # Get prediction score, class and bbox if above threshold
        for label, score, box in zip(predictions["labels"], predictions["scores"], predictions["boxes"]):
            if score >= self.confidence_threshold:
                filtered_labels.append(f"Smoke: {score:.3f}")
                # move to cpu to correctly display predictions with plotlib
                filtered_boxes.append(box.long().to("cpu"))
                filtered_scores.append(score)

        # annotation path for ground truth using test_annot_dir
        # CHANGE image ID to filename
        image_id = filename # temp
        #print(image_id)
        annotation_path = os.path.join(self.test_annot_dir, f"{image_id}.xml")
        ground_truth = self.parse_xml(annotation_path)

        # format predictions for mAP calculation
        predicted = {
            "boxes": torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.float),
            "scores": torch.tensor(filtered_scores) if filtered_scores else torch.empty((0,)),
            "labels": torch.tensor([1] * len(filtered_boxes)) if filtered_boxes else torch.empty((0,),
                                                                                                 dtype=torch.long),
        }

        # getting ground truth size (small, medium, large)
        # filename stored with format in dict, might change dict later
        gt_size = self.image_gt_size.get(filename + ".jpeg")
        #print(gt_size)

        # getting tp, tn, fp, fn for each image
        metrics = self.get_image_val(
            pred_score=predicted["scores"].to("cpu"),  # cpu or cry
            pred_box=predicted["boxes"].to("cpu"),
            ground_truth=ground_truth,
            confidence_threshold=self.confidence_threshold,
            ap_value=self.ap_value,
            gt_size=gt_size
        )

        # Accumulate TP, FP, FN counts for the total
        print(f"{metrics}")
        for size in ["small", "medium", "large", "global"]:
            self.total_tp[size] += metrics["TP"][size]
            self.total_fp[size] += metrics["FP"][size]
            self.total_fn[size] += metrics["FN"][size]

        if gt_size == "small":
            self.map_metricSmallA.update([predicted], [ground_truth])
            self.map_metricSmallB.update([predicted], [ground_truth])
        elif gt_size == "medium":
            self.map_metricMediumA.update([predicted], [ground_truth])
            self.map_metricMediumB.update([predicted], [ground_truth])
        elif gt_size == "large":
            self.map_metricLargeA.update([predicted], [ground_truth])
            self.map_metricLargeB.update([predicted], [ground_truth])

        # mAP update
        print(f"predicted: {predicted}, Ground truth: {ground_truth}")
        self.map_metricGlobalA.update([predicted], [ground_truth])
        self.map_metricGlobalB.update([predicted], [ground_truth])
        return filtered_labels, filtered_boxes, filtered_scores, image, ground_truth, metrics, filename

    # function for each image gets the number of true positive, false positive, etc
    def get_image_val(self, pred_score, pred_box, ground_truth, confidence_threshold, ap_value,
                      gt_size):
        # Filter predictions by confidence score
        filtered_boxes = []
        filtered_scores = []
        for score, box in zip(pred_score, pred_box):
            if score >= confidence_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)

        # Initialize counters
        tp_count = {'small': 0, 'medium': 0, 'large': 0, 'global': 0}
        fp_count = {'small': 0, 'medium': 0, 'large': 0, 'global': 0}
        fn_count = {'small': 0, 'medium': 0, 'large': 0, 'global': 0}

        # Extract ground truth boxes
        ground_truth_boxes = ground_truth['boxes']
        predicted_boxes = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.float)

        # Calculate TP, FP, FN
        if len(predicted_boxes) > 0 and len(ground_truth_boxes) > 0:
            iou_matrix = box_iou(predicted_boxes, ground_truth_boxes)
            matched_gts = set()

            for i, pred_box in enumerate(predicted_boxes):
                max_iou, max_idx = iou_matrix[i].max(0)
                if max_iou >= ap_value:
                    tp_count['global'] += 1
                    if gt_size != 'none':  # Update size counter if not none
                        tp_count[gt_size] += 1
                    matched_gts.add(max_idx.item())
                else:
                    fp_count['global'] += 1
                    if gt_size != 'none':  # Update size counter if not none
                        fp_count[gt_size] += 1

            # Ground truth boxes not matched are FN
            fn_count['global'] = len(ground_truth_boxes) - len(matched_gts)
            if gt_size != 'none':  # Update size counter if not none
                fn_count[gt_size] = len(ground_truth_boxes) - len(matched_gts)
        else:
            # No predictions
            fp_count['global'] = len(predicted_boxes)
            if gt_size != 'none':  # Update size counter if not none
                fp_count[gt_size] = len(predicted_boxes)
            fn_count['global'] = len(ground_truth_boxes)
            if gt_size != 'none':  # Update size counter if not none
                fn_count[gt_size] = len(ground_truth_boxes)

        # Return dictionary with results
        return {
            "TP": tp_count,
            "FP": fp_count,
            "FN": fn_count
        }



    # !!GETTING GROUND TRUTH AND MAP VALUES AND PRECISION/RECALL
    # function initialising mAP metrics
    def initialise_metrics(self):
        # kind of ugly but changing this would just make things way more complicated for no gain
        self.map_metricGlobalA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricGlobalB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricSmallA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricSmallB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricMediumA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricMediumB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricLargeA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricLargeB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricLocal = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])

    # Need ground truths to calculate mAP
    # Function parse xml for ground truths (copied from Dataset class)
    # and get area of ground truth to assign each a size (small, medium, large)
    def parse_xml(self, annotation_path, get_area=False):
        upscale_value = 1
        scale_x, scale_y = 1, 1
        if(self.use_scale):
            scale_y = self.scale_height
            scale_x = self.scale_width
        if(setTestValues("upscale_image")):
            # bring ground truth in line with upscaled image
            upscale_value = setTestValues("upscale_value")
        # Parsing XML file for each image to find bounding boxes
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        # Extract bounding boxes
        boxes = []
        areas = []
        for obj, size in zip(root.findall("object"), root.findall("size")):
            xml_box = obj.find("bndbox")
            image_height = float(size.find("height").text)
            image_width = float(size.find("width").text)

            if(self.scale_height != None and self.scale_width != None):
                scale_x = self.scale_width / image_width
                scale_y = self.scale_height / image_height

            xmin = float(xml_box.find("xmin").text) * upscale_value * scale_x
            ymin = float(xml_box.find("ymin").text) * upscale_value * scale_y
            xmax = float(xml_box.find("xmax").text) * upscale_value * scale_x
            ymax = float(xml_box.find("ymax").text) * upscale_value * scale_y

            if xmin >= xmax or ymin >= ymax:
                print(f"Invalid area/box coordinates: ({xmin}, {ymin}), ({xmax}, {ymax})")

            boxes.append([xmin, ymin, xmax, ymax])
            area = (xmax - xmin) * (ymax - ymin) # get area of ground truth
            if (get_area):
                areas.append(area)

        # Extract labels
        labels = []
        class_to_idx = {"smoke": 1}  # dictionary, if "smoke" return 1
        for obj in root.findall("object"):
            label = obj.find("name").text  # find name in xml
            labels.append(class_to_idx.get(label, 0))  # 0 if the class isn't found in dictionary

        # Convert boxes and labels to tensors for torchmetrics
        ground_truth = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        # Some images have more than 1 ground truth, in that case only use that images AP for global
        if get_area:
            if len(areas) == 1:
                return areas
            elif len(areas) > 1 or len(areas) == 0:
                return 0
        else:
            return ground_truth

    # function to assign each ground truth a size (small, medium or large)
    # Used to calculate size mAP's
    def get_ground_truth_size(self, test_image_dir, test_annot_dir):
        all_areas = []

        # Iterate through files in test image dir
        for root, dirs, files in os.walk(test_image_dir):
            for file_name in files:
                if file_name.endswith(".jpeg"):  # dont really need but whatever
                    # Get the corresponding XML annotation
                    get_annotation = os.path.join(test_annot_dir, file_name.replace(".jpeg", ".xml"))

                    # Parse XML to get areas (ground truth)
                    ground_truth_area = self.parse_xml(get_annotation, True)

                    # append the area to list for later
                    if ground_truth_area:
                        all_areas.append(ground_truth_area[0])  # append ground truth

        # sort areas in asc order
        sorted_areas = sorted(all_areas)

        # get boundaries for small, medium, and large sizes
        # 1/3 - small, 1/3 - medium, 1/3 - large | 33/33/33 split
        total_images = len(sorted_areas)
        small_limit = total_images // 3
        medium_limit = 2 * total_images // 3

        # Create dictionary for categorizing images by size
        categorized_images_dict = {}

        for root, dirs, files in os.walk(test_image_dir):
            for filename in files:
                if filename.endswith(".jpeg"): # dont really need but whatever
                    # Get corresponding XML annotation file
                    get_annotation = os.path.join(test_annot_dir, filename.replace(".jpeg", ".xml")) # image and annot have same names

                    # Parse XML to get areas (ground truth)
                    ground_truth_area = self.parse_xml(get_annotation, True)

                    # assigning each ground truth a size
                    if ground_truth_area:
                        area = ground_truth_area[0]
                        if area == 0:
                            size_category = "none"
                        elif area <= sorted_areas[small_limit - 1]:
                            size_category = "small"
                        elif area <= sorted_areas[medium_limit - 1]:
                            size_category = "medium"
                        else:
                            size_category = "large"
                    else:
                        size_category = "none"

                    # Store the size category in dictionary
                    categorized_images_dict[filename] = size_category

        # Count occurrences of each size
        size_counts = {"small": 0, "medium": 0, "large": 0, "none": 0}
        for size in categorized_images_dict.values():
            if size in size_counts:
                size_counts[size] += 1

        # Print count for each size
        print(f"Small: {size_counts['small']}")
        print(f"Medium: {size_counts['medium']}")
        print(f"Large: {size_counts['large']}")
        print(f"None: {size_counts['none']}")

        # Print area metrics
        # remove outliers
        q1 = np.percentile(all_areas, 25)
        q3 = np.percentile(all_areas, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_areas = [area for area in all_areas if lower_bound <= area <= upper_bound]

        # get min max and avg
        max_area = max(filtered_areas)
        min_area = min(filtered_areas)
        avg_area = sum(filtered_areas) / len(filtered_areas)

        print(f"Max Area (25th to 75th percentile): {max_area}")
        print(f"Min Area (25th to 75th percentile): {min_area}")
        print(f"Avg Area (25th to 75th percentile): {avg_area:.2f}")

        return categorized_images_dict

    # function to get precission and recall
    def calculate_total_precision_recall(self):
        # Precision and Recall for each size category
        precision_recall = {
            "small": {},
            "medium": {},
            "large": {},
            "global": {}
        }

        for size in ["small", "medium", "large", "global"]:
            tp = self.total_tp.get(size, 0)
            fp = self.total_fp.get(size, 0)
            fn = self.total_fn.get(size, 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_recall[size]["precision"] = precision
            precision_recall[size]["recall"] = recall

        return precision_recall


    # !!VISUALISATIONS!!
    def get_bbox_colours(self, filtered_labels, filtered_boxes, ground_truth, n_iou, m_iou, h_iou):
        for label, box in zip(filtered_labels, filtered_boxes):
            iou_matrix = box_iou(box.unsqueeze(0), ground_truth["boxes"])
            # if more than 1 ground truth then this will return multiple values
            # use highest
            iou_matrix = torch.max(iou_matrix)
            if(iou_matrix < 0.3):
                n_iou["label"].append(label)
                n_iou["boxes"].append(box)
            elif(iou_matrix > 0.3 and iou_matrix < 0.5):
                m_iou["label"].append(label)
                m_iou["boxes"].append(box)
            else:
                h_iou["label"].append(label)
                h_iou["boxes"].append(box)

        return n_iou, m_iou, h_iou
    def display_prediction(self, filtered_labels, filtered_boxes, filtered_scores, image, ground_truth, metrics, filename):
        if(not self.plot_image):
            matplotlib.use('Agg')
        # !!! DISPLAYING PREDICTIONS THROUGH MATPLOT LIB !!!
        # get iou to set bbox colours
        n_iou = {"boxes": [], "label": []}
        m_iou = {"boxes": [], "label": []}
        h_iou = {"boxes": [], "label": []}
        if(len(filtered_boxes) > 0):
            n_iou, m_iou, h_iou = self.get_bbox_colours(filtered_labels, filtered_boxes, ground_truth, n_iou, m_iou, h_iou)
        #filtered_boxes_tensor = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.long)
        n_iou["boxes"] = torch.stack(n_iou["boxes"]) if n_iou["boxes"] else torch.empty((0, 4), dtype=torch.long)
        m_iou["boxes"] = torch.stack(m_iou["boxes"]) if m_iou["boxes"] else torch.empty((0, 4), dtype=torch.long)
        h_iou["boxes"] = torch.stack(h_iou["boxes"]) if h_iou["boxes"] else torch.empty((0, 4), dtype=torch.long)
        # Get prediction with highest score and draw only that if draw_only_highest is true
        if filtered_scores and self.draw_highest_only:
            max_score_idx = filtered_scores.index(max(filtered_scores))  # get index pos of highest score
            highest_score_box = filtered_boxes_tensor[max_score_idx].unsqueeze(0)  # Add batch dimension
            highest_score_label = [filtered_labels[max_score_idx]]  # get highest value using index pos
            # Draw highest scoring bbox in red
            output_image = draw_bounding_boxes(image, highest_score_box, highest_score_label, colors="red")
        else:
            # Draw all predicted bbox in red
            output_image = draw_bounding_boxes(image, n_iou["boxes"], n_iou["label"], colors="red")
            output_image = draw_bounding_boxes(output_image, m_iou["boxes"], m_iou["label"], colors="yellow")
            output_image = draw_bounding_boxes(output_image, h_iou["boxes"], h_iou["label"], colors="green")

        # Convert filtered boxes to a tensor
        #filtered_boxes = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.long)
        #print("output labels", filtered_labels)

        # convert ground truth boxes to tensor
        ground_truth_boxes = [bbox.clone().detach() for bbox in ground_truth['boxes']]

        # draw ground truth bbox over predicted bbox over image
        if(self.plot_ground_truth):
            output_image = draw_bounding_boxes(output_image, torch.stack(ground_truth_boxes),
                                               ["Ground Truth"] * len(ground_truth_boxes), colors="blue")
        # registry increased ide.rest.api.request.per.minute to 100
        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))
        #print(filename)
        plt.axis('off')

        # creating legend
        r_patch = mpatches.Patch(color='red', label=f'{len(n_iou["boxes"])} mAP < 0.3/False positive')
        y_patch = mpatches.Patch(color='yellow', label=f'{len(m_iou["boxes"])} mAP > 0.3 | < 0.5/Ok detection')
        g_patch = mpatches.Patch(color='green', label=f'{len(h_iou["boxes"])} mAP > 0.5/Good detection')
        b_patch = mpatches.Patch(color='blue', label='Ground truth')
        plt.legend(handles=[r_patch, y_patch, g_patch, b_patch], loc='lower left', fontsize=10)

        if (self.save_plots):
            plot_save_path = self.plot_save_path / filename
            plt.savefig(plot_save_path, bbox_inches="tight")
        if(self.plot_image):
            plt.show()

        matplotlib.pyplot.close()

    # starting chain to display results
    def get_results(self):
        if self.benchmark:
            avg_benchmark = str(round(self.get_avg_time(self.benchmark_times), 2)) + " seconds"
        else:
            avg_benchmark = "N/A"

        # End timer
        end_time = time.time()
        elapsed_time = round(end_time - self.start_time, 2)

        # Get final mAP, sync and compute
        self.map_metricSmallA.sync()
        self.final_mapSmallA = self.map_metricSmallA.compute()

        self.map_metricSmallB.sync()
        self.final_mapSmallB = self.map_metricSmallB.compute()

        self.map_metricMediumA.sync()
        self.final_mapMediumA = self.map_metricMediumA.compute()

        self.map_metricMediumB.sync()
        self.final_mapMediumB = self.map_metricMediumB.compute()

        self.map_metricLargeA.sync()
        self.final_mapLargeA = self.map_metricLargeA.compute()

        self.map_metricLargeB.sync()
        self.final_mapLargeB = self.map_metricLargeB.compute()

        self.map_metricGlobalA.sync()
        self.final_mapA = self.map_metricGlobalA.compute()

        self.map_metricGlobalB.sync()
        self.final_mapB = self.map_metricGlobalB.compute()
        # get precission and recall
        precision_recall = self.calculate_total_precision_recall()
        # get max vram
        max_vram = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        #torch.cuda.reset_peak_memory_stats()  # reset

        # start displaying
        self.nice_layout(precision_recall, elapsed_time, avg_benchmark, max_vram)
        self.ugly_layout(precision_recall, elapsed_time, avg_benchmark, max_vram)

    # !!DISPLAYING RESULTS!!
    # function for nice results
    def nice_layout(self, precision_recall, elapsed_time, avg_benchmark, max_vram):
        # colour codes
        COLOR_SMALL = "\033[94m"  # Blue
        COLOR_MEDIUM = "\033[92m"  # Green
        COLOR_LARGE = "\033[93m"  # Yellow
        COLOR_GLOBAL = "\033[91m"  # Red
        RESET = "\033[0m"  # Reset to default
        BOLD = "\033[1m"  # Bold text

        # total time and vram
        elapsed_time_data = [[f"{BOLD}Global Time{RESET}", f"{elapsed_time:} seconds"]]

        elapsed_time_data = [
            [f"{BOLD}Global Time{RESET}", f"{elapsed_time:} seconds"],
            [f"{BOLD}Max VRAM{RESET}", f"{max_vram:.2f} MB"],
            [f"{BOLD}Benchmark Time (per image){RESET}", f"{avg_benchmark}"]
        ]

        print(f"\n{BOLD}Time and VRAM:{RESET}")
        print(tabulate(elapsed_time_data, headers=[f"{BOLD}Metric{RESET}", f"{BOLD}Value{RESET}"],
                       tablefmt="fancy_grid"))

        # nice table
        combined_data = [
            [f"{BOLD}Precision{RESET}",
             f"{precision_recall['global']['precision'] * 100:.2f}%",
             f"{precision_recall['small']['precision'] * 100:.2f}%",
             f"{precision_recall['medium']['precision'] * 100:.2f}%",
             f"{precision_recall['large']['precision'] * 100:.2f}%"
             ],

            [f"{BOLD}Recall{RESET}",
             f"{precision_recall['global']['recall'] * 100:.2f}%",
             f"{precision_recall['small']['recall'] * 100:.2f}%",
             f"{precision_recall['medium']['recall'] * 100:.2f}%",
             f"{precision_recall['large']['recall'] * 100:.2f}%"
             ],

            [f"{BOLD}mAP @ 0.5{RESET}",
             f"{self.final_mapA['map'] * 100:.2f}%",
             f"{self.final_mapSmallA['map'] * 100:.2f}%",
             f"{self.final_mapMediumA['map'] * 100:.2f}%",
             f"{self.final_mapLargeA['map'] * 100:.2f}%"
             ],

            [f"{BOLD}mAP @ 0.3{RESET}",
             f"{self.final_mapB['map'] * 100:.2f}%",
             f"{self.final_mapSmallB['map'] * 100:.2f}%",
             f"{self.final_mapMediumB['map'] * 100:.2f}%",
             f"{self.final_mapLargeB['map'] * 100:.2f}%"
             ]
        ]

        # print table
        print(f"{BOLD}Precision, Recall, and mAP Values:{RESET}")
        print(tabulate(combined_data,
                       headers=[f"{BOLD}Metric{RESET}", f"{COLOR_GLOBAL}Global{RESET}",
                                f"{COLOR_SMALL}Small{RESET}", f"{COLOR_MEDIUM}Medium{RESET}",
                                f"{COLOR_LARGE}Large{RESET}"], tablefmt="fancy_grid"))

    def ugly_layout(self, precision_recall, elapsed_time, avg_benchmark, max_vram):
        # Ugly output for copy and paste
        print("\nUgly output for excel sheet copy/paste")
        output_data = [
            [   f"\"{self.model_name}\"",
                f"\"Precision: {precision_recall['global']['precision'] * 100:.2f}%\nRecall: {precision_recall['global']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapB['map'] * 100:.2f}%\"",
                f"\"Precision: {precision_recall['small']['precision'] * 100:.2f}%\nRecall: {precision_recall['small']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapSmallA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapSmallB['map'] * 100:.2f}%\"",
                f"\"Precision: {precision_recall['medium']['precision'] * 100:.2f}%\nRecall: {precision_recall['medium']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapMediumA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapMediumB['map'] * 100:.2f}%\"",
                f"\"Precision: {precision_recall['large']['precision'] * 100:.2f}%\nRecall: {precision_recall['large']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapLargeA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapLargeB['map'] * 100:.2f}%\"",
                f"\"{max_vram}\"",
                f"\"{elapsed_time}\"",
                f"\"{avg_benchmark}\"",
                f"\"{self.batch_size}\""
            ]
        ]

        # headers
        headers = ["Model_name", "Global", "Small", "Medium", "Large", "Max size in vram (MB)", "Total time (seconds)",
                   "Benchmark (seconds), Batch Size"]

        # all this was tryna get excel to paste properly
        output = "\t".join(headers) + "\n"
        for row in output_data:
            output += "\t".join(row) + "\n"
        print(output.strip())


def main():
    # create instance of class
    tester = Tester()
    # start testing chain
    tester.test_dir()



if __name__ == '__main__':
    main()








