""" Evaluate accuracy on ImageNet dataset of PipeEdge using GraSP pruning """
import os
import argparse
import time
import torch
import numpy as np
import io
from contextlib import redirect_stdout
from typing import List
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet
from torchvision import transforms
from transformers import DeiTFeatureExtractor, ViTFeatureExtractor
from runtime import forward_hook_quant_encode, forward_pre_hook_quant_decode
from utils.data import ViTFeatureExtractorTransforms
import model_cfg
from evaluation_tools.evaluation_quant_test import *

class EnhancedReportAccuracy():
    def __init__(self, batch_size, output_dir, model_name, partition, quant) -> None:
        self.current_acc = 0.0
        self.total_acc = 0.0
        self.correct = 0
        self.tested_batch = 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.partition = partition
        self.quant = quant
        self.model_name = model_name.split('/')[1]
        self.pruning_keep_ratio = None  # Will store the pruning keep ratio
        
        # Create directory if it doesn't exist
        self.file_path = os.path.join(self.output_dir, self.model_name)
        os.makedirs(self.file_path, exist_ok=True)
        
        # Initialize files
        self.acc_file = os.path.join(self.file_path, f"result_{self.partition}_{str(self.quant)}.txt")
        self.sparsity_file = os.path.join(self.file_path, f"sparsity_{self.partition}_{str(self.quant)}.txt")
        self.final_file = os.path.join(self.file_path, f"final_{self.partition}_{str(self.quant)}.txt")
        
        # Initialize sparsity dict (raw parameter-level entries)
        self.sparsity_info = {}
        # Logical layer aggregation will be stored here
        self.logical_layers = {}

    def set_pruning_keep_ratio(self, keep_ratio):
        """Set the pruning keep ratio"""
        self.pruning_keep_ratio = keep_ratio
        
    def update(self, pred, target):
        self.correct = pred.eq(target.view(1, -1).expand_as(pred)).float().sum()
        self.current_acc = self.correct / self.batch_size
        self.total_acc = (self.total_acc * self.tested_batch + self.current_acc) / (self.tested_batch + 1)
        self.tested_batch += 1

    def report(self):
        print(f"The accuracy so far is: {100*self.total_acc:.2f}")
        file_name = os.path.join(self.output_dir, self.model_name, "result_" + self.partition + "_" + str(self.quant) + ".txt")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'a') as f:
            f.write(f"{100*self.total_acc:.2f}\n")
            
    def capture_sparsity(self, layer_info):
        """
        Capture layer density information printed during pruning.
        Expected format: "Layer: <layer_name> => Density: <density_value>"
        """
        if "Layer:" in layer_info and "Density:" in layer_info:
            try:
                # Extract layer name and density value
                layer_name = layer_info.split("Layer:")[1].split("=>")[0].strip()
                density = float(layer_info.split("Density:")[1].strip())
                # Store the density directly (later we compute sparsity as 1-density)
                counter = len(self.sparsity_info) + 1
                key = f"{counter}_{layer_name}"
                self.sparsity_info[key] = density
                with open(self.sparsity_file, 'a') as f:
                    f.write(f"{key}: {density:.6f}\n")
            except (ValueError, IndexError) as e:
                print(f"Error parsing layer info: {e}")

    def map_to_logical_layers(self):
        """
        Aggregate the raw density values from the prunable transformer layers
        into 48 logical layers. For a ViT model, we consider only linear layers 
        corresponding to self-attention and MLP components. Specifically, we:
          - Filter keys to only include those with: 
              'self_attention.query', 'self_attention.key', 'self_attention.value',
              'self_output.dense', 'intermediate.dense', 'output.dense'
          - Remove the numeric prefix and the .weight/.bias suffix to group together
            the weight and bias entries for the same sublayer.
          - For each transformer block (e.g. "vit.layers.X"), group:
              Logical Layer 1: Average of Q, K, V sublayers.
              Logical Layer 2: self_output sublayer.
              Logical Layer 3: intermediate sublayer.
              Logical Layer 4: output sublayer.
        """
        # First, build a dictionary mapping sublayer base names to list of density values.
        sublayer_dict = {}
        for key, density in self.sparsity_info.items():
            if "vit.layers." not in key:
                continue
            # Only consider pruned linear layers of interest.
            if not ( "self_attention.query" in key or "self_attention.key" in key or 
                     "self_attention.value" in key or "self_output.dense" in key or 
                     "intermediate.dense" in key or "output.dense" in key ):
                continue
            # Remove the numeric prefix.
            try:
                sublayer_full = key.split('_', 1)[1]  # e.g., "vit.layers.0.layernorm_before.weight"
            except IndexError:
                continue
            # Remove trailing .weight or .bias to get a common sublayer identifier.
            if sublayer_full.endswith('.weight'):
                base = sublayer_full[:-7]
            elif sublayer_full.endswith('.bias'):
                base = sublayer_full[:-5]
            else:
                base = sublayer_full
            if base not in sublayer_dict:
                sublayer_dict[base] = []
            sublayer_dict[base].append(density)
        # Average the densities per sublayer.
        averaged_sublayer = {base: sum(vals)/len(vals) for base, vals in sublayer_dict.items()}

        # Now group into logical layers.
        logical_layers = {}
        # Each key in averaged_sublayer is expected to be of the form "vit.layers.<block_idx>.<sublayer_name>"
        for base, avg_density in averaged_sublayer.items():
            tokens = base.split('.')
            if len(tokens) < 3:
                continue
            try:
                block_idx = int(tokens[2])
            except ValueError:
                continue
            # Determine logical group based on sublayer name.
            if "self_attention.query" in base or "self_attention.key" in base or "self_attention.value" in base:
                logical_group = 1  # Attention QKV (grouped together)
            elif "self_output" in base:
                logical_group = 2  # Attention Output
            elif "intermediate" in base:
                logical_group = 3  # MLP1
            elif "output" in base:
                logical_group = 4  # MLP2
            else:
                continue
            # Compute overall logical layer index (1-indexed).
            logical_idx = block_idx * 4 + logical_group
            if logical_idx not in logical_layers:
                logical_layers[logical_idx] = []
            logical_layers[logical_idx].append(avg_density)
        
        # Average densities within each logical layer.
        final_logical = {}
        for logical_idx, density_list in logical_layers.items():
            final_logical[logical_idx] = sum(density_list) / len(density_list)
        self.logical_layers = final_logical
        return final_logical
                
    def save_final_stats(self):
        """Save final accuracy and aggregated sparsity statistics."""
        # Map raw sparsity (which is 1 - density) into logical layers.
        # Note: Since our captured value is density, we convert to sparsity below.
        logical = self.map_to_logical_layers()
        with open(self.final_file, 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Partition: {self.partition}\n")
            if self.pruning_keep_ratio is not None:
                f.write(f"Pruning Keep Ratio: {self.pruning_keep_ratio}\n")
                f.write(f"Pruning Factor: {1.0 - self.pruning_keep_ratio:.6f}\n")
            f.write(f"Final Accuracy: {100*self.total_acc:.6f}%\n")
            f.write(f"Total Batches: {self.tested_batch}\n\n")
            
            if self.sparsity_info:
                # Report raw average sparsity (computed as 1-density)
                raw_sparsities = [1 - d for d in self.sparsity_info.values()]
                avg_raw = sum(raw_sparsities)/len(raw_sparsities)
                f.write(f"Average Raw Sparsity: {avg_raw:.6f}\n")
                f.write("Raw Layer Sparsity (per printed entry):\n")
                for key, density in self.sparsity_info.items():
                    sparsity = 1 - density
                    f.write(f"  {key}: {sparsity:.6f}\n")
            
            if self.logical_layers:
                # Compute average logical sparsity from densities
                logical_sparsity = {k: 1 - v for k, v in self.logical_layers.items()}
                avg_logical = sum(logical_sparsity.values()) / len(logical_sparsity)
                f.write(f"\nAverage Logical Layer Sparsity (aggregated over transformer blocks): {avg_logical:.6f}\n")
                f.write("Logical Layer Sparsity (48 layers):\n")
                # Optionally, print in order (from 1 to max)
                for idx in sorted(logical_sparsity.keys()):
                    f.write(f"  Logical Layer {idx}: {logical_sparsity[idx]:.6f}\n")
        # Also append the average logical sparsity to the sparsity file.
        if self.logical_layers:
            logical_sparsity = {k: 1 - v for k, v in self.logical_layers.items()}
            avg_logical = sum(logical_sparsity.values()) / len(logical_sparsity)
            with open(self.sparsity_file, 'a') as sf:
                sf.write(f"\nAverage Logical Layer Sparsity: {avg_logical:.6f}\n")

def _make_shard(model_name, model_file, stage_layers, stage, q_bits, prune):
    shard = model_cfg.module_shard_factory(model_name, model_file, stage_layers[stage][0],
                                            stage_layers[stage][1], stage, prune)
    shard.register_buffer('quant_bits', q_bits)
    shard.eval()
    return shard

def _forward_model(input_tensor, model_shards):
    num_shards = len(model_shards)
    temp_tensor = input_tensor
    for idx in range(num_shards):
        shard = model_shards[idx]
        # Decoder step: if not the first shard, decode quantization
        if idx != 0:
            temp_tensor = forward_pre_hook_quant_decode(shard, temp_tensor)
        # Unwrap tuple if needed.
        if isinstance(temp_tensor[0], tuple) and len(temp_tensor[0]) == 2:
            temp_tensor = temp_tensor[0]
        elif isinstance(temp_tensor, tuple) and isinstance(temp_tensor[0], torch.Tensor):
            temp_tensor = temp_tensor[0]
        temp_tensor = shard(temp_tensor)
        # Encoder step: if not the last shard, encode quantization
        if idx != num_shards - 1:
            temp_tensor = (forward_hook_quant_encode(shard, None, temp_tensor),)
    return temp_tensor

def prune_grasp(model, ubatch, ubatch_labels, keep_ratio):
    """
    Implements the GraSP pruning method.
    Args:
      model: the model (or shard) to be pruned.
      ubatch: a batch of calibration data (inputs).
      ubatch_labels: corresponding labels.
      keep_ratio: fraction of weights to keep (e.g., 0.9 means keep top 90% weights).
    Returns:
      A dictionary of pruned weights.
    """
    # Use cross-entropy loss for classification.
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    model.zero_grad()
    output = model(ubatch)
    loss = criterion(output, ubatch_labels)
    loss.backward(retain_graph=True)  # First-order gradients

    # Compute a scalar as the sum of element-wise product of weights and gradients.
    scalar = 0.0
    for p in model.parameters():
        if p.requires_grad:
            scalar = scalar + (p * p.grad).sum()
    model.zero_grad()
    scalar.backward()  # Second backward pass approximates H * g

    # For each weight, compute the GraSP score.
    # Here we use the absolute value of (w * p.grad) as the score.
    scores = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            scores[name] = (p * p.grad).abs()

    # Create a binary mask for each parameter tensor to keep the top keep_ratio fraction.
    masks = {}
    for name, score in scores.items():
        flattened = score.view(-1)
        total_elements = flattened.numel()
        k = int(keep_ratio * total_elements)
        if k < 1:
            k = 1
        # Get threshold such that top k scores are kept.
        threshold, _ = torch.kthvalue(flattened, total_elements - k + 1)
        mask = (score >= threshold).float()
        density = mask.sum().item() / mask.numel()
        print(f"Layer: {name} => Density: {density:.6f}")
        masks[name] = mask

    # Apply masks to prune the weights and store pruned weights.
    pruned_weights = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in masks:
            pruned_weights[name] = p.data * masks[name]
            p.data.mul_(masks[name])
    return pruned_weights

def evaluation(args, dataset_cfg):
    """ Evaluation main function """
    dataset_path = args.dataset_root
    dataset_split = args.dataset_split
    batch_size = args.batch_size
    ubatch_size = args.ubatch_size
    num_workers = args.num_workers
    partition = args.partition
    quant = args.quant
    output_dir = args.output_dir
    model_name = args.model_name
    model_file = args.model_file
    num_stop_batch = args.stop_at_batch
    prune = args.prune
    train_batch_size = args.train_batch_size
    keep_ratio = args.keep_ratio

    # Load dataset for validation
    if model_name in ['facebook/deit-base-distilled-patch16-224',
                      'facebook/deit-small-distilled-patch16-224',
                      'facebook/deit-tiny-distilled-patch16-224']:
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
        val_transform = ViTFeatureExtractorTransforms(feature_extractor)
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split), transform=val_transform)
    elif model_name.startswith('torchvision'):
        feature_extractor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split), transform=feature_extractor)
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        val_transform = ViTFeatureExtractorTransforms(feature_extractor)
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split), transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    # Initialize the accuracy reporter
    acc_reporter = EnhancedReportAccuracy(batch_size, output_dir, model_name, partition, quant[0] if quant else 0)

    if prune:
        pruned_model_file = model_cfg._MODEL_CONFIGS[model_name]['pruned_weights_file']
        print("keep ratio : ", keep_ratio, ", train_data size : ", train_batch_size)
        acc_reporter.set_pruning_keep_ratio(keep_ratio)
        
        # Load training data for calibration (pruning phase)
        train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=val_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True
        )
        for ubatch, ubatch_labels in train_loader:
            config = model_cfg.get_model_config(model_name)
            shard_config = model_cfg.ModuleShardConfig(layer_start=1, layer_end=model_cfg.get_model_layers(model_name),
                                                        is_first=True, is_last=True)
            model_file = model_cfg.get_model_default_weights_file(model_name)
            model = model_cfg._MODEL_CONFIGS[model_name]['shard_module'](config, shard_config, model_file)
            
            # Use GraSP pruning instead of SNIP.
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                weights = prune_grasp(model, ubatch, ubatch_labels, keep_ratio)
            # Capture printed density information for sparsity reporting.
            for line in output_buffer.getvalue().split('\n'):
                if "Layer:" in line and "Density:" in line:
                    acc_reporter.capture_sparsity(line)
            np.savez(pruned_model_file, **weights)
            print('GraSP pruning successfully.')
            model_file = pruned_model_file
            break

    def _get_default_quant(n_stages: int) -> List[int]:
        return [0] * n_stages
    parts = [int(i) for i in partition.split(',')]
    assert len(parts) % 2 == 0
    num_shards = len(parts) // 2
    stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
    stage_quant = [int(i) for i in quant.split(',')] if quant else _get_default_quant(len(stage_layers))

    # Construct model shards for pipeline parallelism.
    model_shards = []
    for stage in range(num_shards):
        q_bits = torch.tensor((0 if stage == 0 else stage_quant[stage - 1], stage_quant[stage]))
        shard = _make_shard(model_name, model_file, stage_layers, stage, q_bits, prune)
        shard.register_buffer('quant_bit', torch.tensor(stage_quant[stage]), persistent=False)
        model_shards.append(shard)

    # Run inference on the validation set.
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            if batch_idx == num_stop_batch and num_stop_batch:
                break
            output = _forward_model(input, model_shards)
            _, pred = output.topk(1)
            pred = pred.t()
            acc_reporter.update(pred, target)
            acc_reporter.report()
    print(f"Final Accuracy: {100*acc_reporter.total_acc}; Quant Bitwidth: {stage_quant}")
    end_time = time.time()
    print(f"total time = {end_time - start_time}")
    acc_reporter.save_final_stats()

if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Evaluation on Single GPU",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Eval configs
    parser.add_argument("-q", "--quant", type=str,
                        help="comma-delimited list of quantization bits to use after each stage")
    parser.add_argument("-pt", "--partition", type=str, default='1,22,23,48',
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; single-node default: all layers in the model")
    parser.add_argument("-o", "--output-dir", type=str, default="/home1/haonanwa/projects/PipeEdge/results")
    parser.add_argument("-st", "--stop-at-batch", type=int, default=None, help="the # of batch to stop evaluation")
    
    # Device options
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="compute device type to use, with optional ordinal, e.g.: 'cpu', 'cuda', 'cuda:1'")
    parser.add_argument("-n", "--num-workers", default=4, type=int,
                        help="the number of worker threads for the dataloader")
    # Model options
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    # Dataset options
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-tb", "--train-batch-size", default=64, type=int, help="train batch size for pruning")
    parser.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")

    dset = parser.add_argument_group('Dataset arguments')
    dset.add_argument("--dataset-name", type=str, default='ImageNet', choices=['CoLA', 'ImageNet'],
                      help="dataset to use")
    dset.add_argument("--dataset-root", type=str, default="/project/jpwalter_148/hnwang/datasets/ImageNet/",
                      help="dataset root directory (e.g., for 'ImageNet', must contain 'ILSVRC2012_devkit_t12.tar.gz' and at least one of: 'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar')")
    dset.add_argument("--dataset-split", default='val', type=str,
                      help="dataset split (depends on dataset), e.g.: train, val, validation, test")
    dset.add_argument("--dataset-indices-file", default=None, type=str,
                      help="PyTorch or NumPy file with precomputed dataset index sequence")
    dset.add_argument("--dataset-shuffle", type=bool, nargs='?', const=True, default=False,
                      help="dataset shuffle")
    dset.add_argument("--prune", type=bool, nargs='?', const=True, default=False,
                      help="Pruning method")
    dset.add_argument("--keep-ratio", type=float, default=0.9,
                      help="keep ratio for pruning")
    args = parser.parse_args()

    if args.dataset_indices_file is None:
        indices = None
    elif args.dataset_indices_file.endswith('.pt'):
        indices = torch.load(args.dataset_indices_file)
    else:
        indices = np.load(args.dataset_indices_file)
    dataset_cfg = {
        'name': args.dataset_name,
        'root': args.dataset_root,
        'split': args.dataset_split,
        'indices': indices,
        'shuffle': args.dataset_shuffle,
    }

    evaluation(args, dataset_cfg)
