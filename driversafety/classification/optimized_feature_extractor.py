from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import numpy as np
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

# Windows 11 compatible optimization imports
try:
    import torch.jit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False

try:
    # TensorRT optimization (optional)
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


def default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class OptimizedResNetFeatureExtractor:
    """High-performance ResNet-based feature extractor with GPU acceleration and multi-threading."""

    def __init__(self, variant: str = "resnet50", device: str = "auto", enable_fp16: bool = True,
                 max_workers: int = 4, optimization_method: str = "jit_trace"):
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.enable_fp16 = enable_fp16 and self.device.type == "cuda"
        self.max_workers = max_workers
        self.optimization_method = optimization_method

        print(f"Initializing OptimizedResNetFeatureExtractor on {self.device}")
        if self.enable_fp16:
            print("FP16 half-precision enabled for faster inference")
        print(f"Optimization method: {optimization_method}")
        
        # Model initialization
        if variant != "resnet50":
            raise ValueError("Only resnet50 is supported in this reference implementation")
        
        # Use new torchvision weights API (replaces deprecated pretrained=True)
        try:
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            # Fallback to explicit v1 weights if DEFAULT not available in local torchvision
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Save full model for Grad-CAM
        self.model = backbone.to(self.device).eval()
        
        # Feature extractor without final FC layer
        self.extractor = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device).eval()
        
        # Enable half-precision if requested
        if self.enable_fp16:
            self.model = self.model.half()
            self.extractor = self.extractor.half()
        
        # Apply Windows 11 compatible optimizations
        self._apply_optimizations()
        
        self.transform = default_transform()
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Warmup the models
        self._warmup()

        # Performance tracking
        self.inference_times = []
        self.lock = threading.Lock()

    def _apply_optimizations(self):
        """Apply Windows 11 compatible optimization techniques."""
        print("Applying Windows 11 compatible optimizations...")

        if self.optimization_method == "jit_trace" and JIT_AVAILABLE:
            self._apply_jit_tracing()
        elif self.optimization_method == "jit_script" and JIT_AVAILABLE:
            self._apply_jit_scripting()
        elif self.optimization_method == "tensorrt" and TENSORRT_AVAILABLE:
            self._apply_tensorrt_optimization()
        elif self.optimization_method == "channels_last":
            self._apply_channels_last_optimization()
        elif self.optimization_method == "fusion":
            self._apply_manual_fusion()
        else:
            print(f"Using baseline optimization (no {self.optimization_method} available)")
            self._apply_baseline_optimizations()

    def _apply_jit_tracing(self):
        """Apply PyTorch JIT tracing optimization (Windows compatible)."""
        try:
            print("Applying JIT tracing optimization...")

            # Create example input for tracing
            example_input = torch.randn(1, 3, 224, 224).to(self.device)
            if self.enable_fp16:
                example_input = example_input.half()

            # Trace the models
            with torch.no_grad():
                # Trace ONLY the extractor. Keep full model in eager mode for Grad-CAM target layers.
                self.extractor = torch.jit.trace(self.extractor, example_input)

            # Optimize traced extractor
            self.extractor = torch.jit.optimize_for_inference(self.extractor)

            print("✓ JIT tracing optimization successful (extractor only)")

        except Exception as e:
            print(f"JIT tracing failed: {e}, falling back to baseline")
            self._apply_baseline_optimizations()

    def _apply_jit_scripting(self):
        """Apply PyTorch JIT scripting optimization."""
        try:
            print("Applying JIT scripting optimization...")

            # Script ONLY the extractor; keep full model eager for Grad-CAM
            self.extractor = torch.jit.script(self.extractor)

            # Optimize scripted extractor
            self.extractor = torch.jit.optimize_for_inference(self.extractor)

            print("✓ JIT scripting optimization successful (extractor only)")

        except Exception as e:
            print(f"JIT scripting failed: {e}, falling back to baseline")
            self._apply_baseline_optimizations()

    def _apply_tensorrt_optimization(self):
        """Apply TensorRT optimization for NVIDIA GPUs."""
        try:
            print("Applying TensorRT optimization...")

            if not self.device.type == "cuda":
                raise RuntimeError("TensorRT requires CUDA device")

            # Create example input
            example_input = torch.randn(1, 3, 224, 224).to(self.device)
            if self.enable_fp16:
                example_input = example_input.half()

            # Convert to TensorRT
            self.extractor = torch_tensorrt.compile(
                self.extractor,
                inputs=[example_input],
                enabled_precisions={torch.float16 if self.enable_fp16 else torch.float32}
            )

            # Keep full model eager to preserve Grad-CAM target layers
            print("✓ TensorRT optimization successful (extractor only)")

        except Exception as e:
            print(f"TensorRT optimization failed: {e}, falling back to JIT tracing")
            self._apply_jit_tracing()

    def _apply_channels_last_optimization(self):
        """Apply channels-last memory format optimization."""
        try:
            print("Applying channels-last memory format optimization...")

            # Convert models to channels-last format
            self.model = self.model.to(memory_format=torch.channels_last)
            self.extractor = self.extractor.to(memory_format=torch.channels_last)

            print("✓ Channels-last optimization successful")

        except Exception as e:
            print(f"Channels-last optimization failed: {e}, using default format")

    def _apply_manual_fusion(self):
        """Apply manual kernel fusion optimizations."""
        try:
            print("Applying manual fusion optimizations...")

            # Enable cuDNN benchmarking for optimal kernel selection
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("✓ cuDNN benchmarking enabled")

            # Enable JIT fusion
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)

            print("✓ Manual fusion optimizations applied")

        except Exception as e:
            print(f"Manual fusion failed: {e}, using baseline")

    def _apply_baseline_optimizations(self):
        """Apply baseline optimizations that work on all systems."""
        print("Applying baseline optimizations...")

        # Enable cuDNN optimizations if available
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("✓ cuDNN optimizations enabled")

        # Set optimal number of threads
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(min(self.max_workers, 8))
            print(f"✓ PyTorch threads set to {torch.get_num_threads()}")

        print("✓ Baseline optimizations applied")
    
    def _warmup(self):
        """Warmup the models with dummy data for optimal performance."""
        print("Warming up models...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        if self.enable_fp16:
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            # Warmup feature extractor
            _ = self.extractor(dummy_input)
            # Warmup full model for Grad-CAM
            _ = self.model(dummy_input)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print("✓ Model warmup complete")
    
    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Optimized image preprocessing with format optimization."""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device, non_blocking=True)
        if self.enable_fp16:
            tensor = tensor.half()

        # Apply channels-last format if using that optimization
        if self.optimization_method == "channels_last":
            tensor = tensor.to(memory_format=torch.channels_last)

        return tensor
    
    @torch.no_grad()
    def extract_single(self, pil_image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
        """Extract features from a single image with optimized performance."""
        start_time = time.time()
        
        # Preprocess
        tensor = self._preprocess_image(pil_image)
        
        # Feature extraction
        features = self.extractor(tensor).squeeze().detach()
        
        # Convert to float32 for compatibility if using FP16
        if self.enable_fp16:
            features = features.float()
        
        features_numpy = features.cpu().numpy()
        
        # Synchronize for accurate timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Track performance
        inference_time = time.time() - start_time
        with self.lock:
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:  # Keep last 100 measurements
                self.inference_times.pop(0)
        
        return features_numpy, tensor
    
    def extract_batch(self, pil_images: List[Image.Image]) -> List[Tuple[np.ndarray, torch.Tensor]]:
        """Extract features from multiple images in parallel."""
        if len(pil_images) == 1:
            return [self.extract_single(pil_images[0])]
        
        # Process images in parallel using thread pool
        futures = [self.thread_pool.submit(self.extract_single, img) for img in pil_images]
        results = [future.result() for future in futures]
        
        return results
    
    def extract(self, pil_image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
        """Main extraction method - maintains compatibility with original interface."""
        return self.extract_single(pil_image)
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        with self.lock:
            if not self.inference_times:
                return {"avg_time": 0, "fps": 0, "samples": 0}
            
            avg_time = np.mean(self.inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            return {
                "avg_time": avg_time,
                "fps": fps,
                "samples": len(self.inference_times),
                "min_time": np.min(self.inference_times),
                "max_time": np.max(self.inference_times)
            }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Clear GPU cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# Backward compatibility alias
ResNetFeatureExtractor = OptimizedResNetFeatureExtractor
