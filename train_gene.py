"""
Training script for Gene Expression Latte with toy dataset for sanity check
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import math
import numpy as np
from time import time
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
#from diffusers.optimization import get_scheduler

class ToyGeneExpressionDataset(Dataset):
    def __init__(self, num_samples=1000, num_genes=1000, num_timepoints=16, num_cell_types=5):
        # Generate synthetic data
        expressions = []
        cell_types = []
        
        # Generate base expression for each cell type
        base_expressions = np.random.normal(0, 1, (num_cell_types, num_genes))
        
        # Create temporal patterns
        time = np.linspace(0, 1, num_timepoints)
        patterns = {
            'increasing': time.reshape(-1, 1),
            'decreasing': (1 - time).reshape(-1, 1),
            'oscillating': np.sin(2 * np.pi * time).reshape(-1, 1)
        }
        
        # Assign patterns to genes
        gene_patterns = np.random.choice(list(patterns.keys()), num_genes)
        
        for i in range(num_samples):
            cell_type = np.random.randint(0, num_cell_types)
            cell_types.append(cell_type)
            
            base_expr = base_expressions[cell_type]
            temporal_expr = np.zeros((num_timepoints, num_genes))
            
            for g in range(num_genes):
                pattern = patterns[gene_patterns[g]]
                temporal_expr[:, g] = base_expr[g] + pattern[:, 0] * np.random.normal(1, 0.1)
            
            noise = np.random.normal(0, 0.1, temporal_expr.shape)
            temporal_expr += noise
            expressions.append(temporal_expr)
        
        self.expressions = torch.FloatTensor(expressions)
        self.cell_types = torch.LongTensor(cell_types)
        
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        return {
            'gene_expression': self.expressions[idx],
            'cell_type': self.cell_types[idx]
        }

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    from models.latte_gene import GeneLatte_S  # Using small model for sanity check
    model = GeneLatte_S(
        num_genes=args.num_genes,
        num_timepoints=args.num_timepoints,
        num_cell_types=args.num_cell_types,
        learn_sigma=True
    ).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create EMA model
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    
    # Create optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Create dataset
    dataset = ToyGeneExpressionDataset(
        num_samples=args.num_samples,
        num_genes=args.num_genes,
        num_timepoints=args.num_timepoints,
        num_cell_types=args.num_cell_types
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}")

    # Create diffusion
    from diffusion import create_diffusion
    diffusion = create_diffusion(timestep_respacing="")

    # Training loop
    model.train()
    running_loss = 0
    start_time = time()
    
    print("Starting training...")
    for step in range(args.max_steps):
        for batch in loader:
            # Get data
            gene_expr = batch['gene_expression'].to(device)
            cell_types = batch['cell_type'].to(device)

            # Sample timesteps
            t = torch.randint(0, diffusion.num_timesteps, (gene_expr.shape[0],), device=device)

            # Calculate loss
            model_kwargs = dict(cell_types=cell_types)
            loss_dict = diffusion.training_losses(model, gene_expr, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            # Update model
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update EMA
            for p_ema, p_model in zip(ema.parameters(), model.parameters()):
                p_ema.data.mul_(0.9999).add_(p_model.data, alpha=1 - 0.9999)

            # Logging
            running_loss += loss.item()
            if (step + 1) % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time() - start_time
                print(f"Step {step+1}/{args.max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time/step: {elapsed/args.log_every:.2f}s")
                running_loss = 0
                start_time = time()

            if step >= args.max_steps:
                break

    print("Training completed!")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        # Generate samples
        num_samples = 2
        num_steps = 1000

        # Start from random noise
        x = torch.randn(num_samples, args.num_timepoints, args.num_genes).to(device)
        cell_types = torch.randint(0, args.num_cell_types, (num_samples,)).to(device)
        
        # Sample using the model
        from tqdm import tqdm
        for i in tqdm(reversed(range(0, num_steps)), desc='Sampling'):
            t_batch = torch.tensor([i] * num_samples, device=device)
            model_kwargs = dict(cell_types=cell_types)
            out = diffusion.p_sample(model, x, t_batch, model_kwargs=model_kwargs)
            x = out["sample"]  # Extract the sample from the dictionary
    
        # Print statistics about generated data
        print("\nGenerated data statistics:")
        print("Shape:", x.shape)
        print("Mean:", x.mean().item())
        print("Std:", x.std().item())
        print("Min:", x.min().item())
        print("Max:", x.max().item())

    return model, ema

def main():
    # Training arguments
    class Args:
        def __init__(self):
            # Model params
            self.num_genes = 100        # Small number for quick testing
            self.num_timepoints = 16
            self.num_cell_types = 5
            self.num_samples = 200      # Small dataset for testing
            
            # Training params
            self.learning_rate = 1e-4
            self.batch_size = 8
            self.max_steps = 100        # Small number of steps for testing
            self.log_every = 10
    
    args = Args()
    model, ema = train(args)
    
if __name__ == "__main__":
    main()