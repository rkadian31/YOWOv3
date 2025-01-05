import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import torch

def plot_grad_flow(named_parameters):
    """Basic gradient flow visualization"""
    ave_grads = []
    layers = []
    
    # Filter and process gradients
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            # Handle potentially larger gradients from HD processing
            if p.grad is not None:  # Add safety check
                grad_mean = p.grad.cpu().abs().mean().item()
                # Apply log scaling for better visualization of HD gradients
                grad_mean = np.log10(grad_mean + 1e-10)
                ave_grads.append(grad_mean)
                layers.append(n)

    plt.figure(figsize=(12, 8))  # Larger figure for better readability
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Log10(average gradient)")
    plt.title("Gradient Flow (HD Resolution)")
    plt.grid(True)
    plt.tight_layout()  # Adjust layout for better text visibility

def plot_grad_flow_v2(named_parameters):
    """Enhanced gradient flow visualization with HD optimizations"""
    ave_grads = []
    max_grads = []
    layers = []
    
    # Add memory management for HD processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                if p.grad is not None:
                    # Process gradients with HD considerations
                    grad_mean = p.grad.cpu().abs().mean().item()
                    grad_max = p.grad.cpu().abs().max().item()
                    
                    # Apply log scaling for better visualization
                    grad_mean = np.log10(grad_mean + 1e-10)
                    grad_max = np.log10(grad_max + 1e-10)
                    
                    layers.append(n)
                    ave_grads.append(grad_mean)
                    max_grads.append(grad_max)

        # Create enhanced visualization
        plt.figure(figsize=(15, 10))  # Larger figure for HD details
        
        # Plot with better visibility for HD data
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        
        # Improve readability for HD-related layers
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation=45, ha='right')
        plt.xlim(left=0, right=len(ave_grads))
        
        # Dynamically set y-axis limits based on gradient values
        y_min = min(min(ave_grads), min(max_grads))
        y_max = max(max(ave_grads), max(max_grads))
        plt.ylim(bottom=y_min-0.1, top=y_max+0.1)
        
        plt.xlabel("Layers")
        plt.ylabel("Log10(gradient)")
        plt.title("Gradient Flow Analysis (HD Resolution)")
        plt.grid(True, alpha=0.3)
        
        # Enhanced legend
        plt.legend([
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4)
        ], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        
        # Add gradient statistics
        stats_text = f'Max gradient: {10**max(max_grads):.2e}\n' \
                    f'Mean gradient: {10**np.mean(ave_grads):.2e}\n' \
                    f'Std deviation: {np.std(ave_grads):.2e}'
        plt.figtext(0.02, 0.02, stats_text, fontsize=8)
        
        plt.tight_layout()  # Adjust layout for better visibility
        plt.show()
        
    except Exception as e:
        print(f"Error in gradient visualization: {str(e)}")
    finally:
        # Cleanup
        plt.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def analyze_gradients(named_parameters, resolution=(1920, 1080)):
    """Additional function to analyze gradients specifically for HD processing"""
    stats = {
        'resolution': resolution,
        'gradient_stats': {},
        'layer_stats': {}
    }
    
    try:
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
                grad = p.grad.cpu().abs()
                
                stats['layer_stats'][n] = {
                    'mean': grad.mean().item(),
                    'max': grad.max().item(),
                    'std': grad.std().item(),
                    'shape': list(p.shape)
                }
                
        # Calculate overall statistics
        means = [s['mean'] for s in stats['layer_stats'].values()]
        stats['gradient_stats'] = {
            'overall_mean': np.mean(means),
            'overall_max': max(s['max'] for s in stats['layer_stats'].values()),
            'overall_std': np.std(means)
        }
        
        return stats
    
    except Exception as e:
        print(f"Error in gradient analysis: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    def test_visualization(model):
        stats = analyze_gradients(model.named_parameters())
        if stats:
            print("Gradient Statistics:")
            print(f"Overall Mean: {stats['gradient_stats']['overall_mean']:.2e}")
            print(f"Overall Max: {stats['gradient_stats']['overall_max']:.2e}")
            print(f"Overall Std: {stats['gradient_stats']['overall_std']:.2e}")
