import torch
from thop import profile
from model.TSN.YOWOv3 import build_yowov3

def get_info(config, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Calculate input dimensions for HD aspect ratio
    height = 1080
    width = 1920
    
    # Create input tensor with HD dimensions
    video_clip = torch.randn(
        1,                          # batch size
        3,                          # channels
        config['clip_length'],      # temporal dimension
        height,                     # height
        width                       # width
    ).to(device)

    # set eval mode
    model.trainable = False
    model.eval()

    # Profile with larger warm-up and repeat counts for accurate HD measurement
    with torch.cuda.amp.autocast(enabled=True):  # Enable AMP for memory efficiency
        flops, params = profile(
            model, 
            inputs=(video_clip, ),
            verbose=False,
            batch_size=1,
            warm_up=5,              # Increased warm-up for stability
            number=3                # Multiple runs for accuracy
        )

    print('==============================')
    print('HD Resolution (1920x1080)')
    print('FLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
    
    # Calculate memory usage
    memory_used = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print('Max GPU Memory : {:.2f} GB'.format(memory_used))
    
    # Calculate theoretical throughput
    if torch.cuda.is_available():
        gpu_flops = torch.cuda.get_device_properties(0).multi_processor_count * 1.7e12  # Approximate FLOPS for modern GPU
        theoretical_fps = min(gpu_flops / flops, 30)  # Cap at 30 FPS
        print('Theoretical Max FPS : {:.2f}'.format(theoretical_fps))
    
    print('==============================')

    # Reset model state
    model.trainable = True
    torch.cuda.empty_cache()  # Clear GPU cache

    return {
        'flops': flops,
        'params': params,
        'memory_used': memory_used if torch.cuda.is_available() else None,
        'resolution': (width, height)
    }

def compare_resolutions(config, model):
    """Compare FLOPs and memory usage between different resolutions"""
    resolutions = [
        (640, 640),    # Original
        (1920, 1080),  # HD
        (3840, 2160)   # 4K
    ]
    
    results = {}
    
    for width, height in resolutions:
        print(f'\nTesting Resolution: {width}x{height}')
        
        # Modify config for current resolution
        config['img_size'] = (width, height)
        
        # Get metrics
        metrics = get_info(config, model)
        results[f'{width}x{height}'] = metrics
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
    return results

if __name__ == "__main__":
    # Example usage
    config = {
        'clip_length': 16,  # or whatever your default is
        'img_size': (1920, 1080)  # HD default
    }
    model = build_yowov3(config)
    
    # Get detailed metrics for HD
    metrics = get_info(config, model)
    
    # Optionally compare different resolutions
    # resolution_comparison = compare_resolutions(config, model)
