"""
Optimization benchmark utility for testing different Windows 11 compatible optimization methods.
"""

import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import warnings

from driversafety.classification.optimized_feature_extractor import OptimizedResNetFeatureExtractor


class OptimizationBenchmark:
    """Benchmark different optimization methods for Windows 11 compatibility."""
    
    def __init__(self, device: str = "auto", enable_fp16: bool = True, num_warmup: int = 5, num_test: int = 20):
        self.device = device
        self.enable_fp16 = enable_fp16
        self.num_warmup = num_warmup
        self.num_test = num_test
        
        # Test methods available on Windows 11
        self.optimization_methods = [
            "baseline",
            "jit_trace", 
            "jit_script",
            "channels_last",
            "fusion"
        ]
        
        # Add TensorRT if available
        try:
            import torch_tensorrt
            self.optimization_methods.append("tensorrt")
        except ImportError:
            pass
    
    def create_test_data(self, batch_size: int = 1) -> Tuple[List[Image.Image], torch.Tensor]:
        """Create test data for benchmarking."""
        # Create random PIL images
        pil_images = []
        for _ in range(batch_size):
            # Random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_image = Image.fromarray(img_array)
            pil_images.append(pil_image)
        
        # Create tensor equivalent
        tensor_batch = torch.randn(batch_size, 3, 224, 224)
        if self.device != "cpu":
            tensor_batch = tensor_batch.cuda()
        if self.enable_fp16:
            tensor_batch = tensor_batch.half()
        
        return pil_images, tensor_batch
    
    def benchmark_method(self, method: str) -> Dict[str, float]:
        """Benchmark a specific optimization method."""
        print(f"\n🔧 Testing optimization method: {method}")
        
        try:
            # Initialize extractor with specific method
            extractor = OptimizedResNetFeatureExtractor(
                variant="resnet50",
                device=self.device,
                enable_fp16=self.enable_fp16,
                max_workers=4,
                optimization_method=method
            )
            
            # Create test data
            pil_images, tensor_batch = self.create_test_data(1)
            test_image = pil_images[0]
            
            # Warmup
            print(f"  Warming up ({self.num_warmup} iterations)...")
            for _ in range(self.num_warmup):
                with torch.no_grad():
                    _ = extractor.extract(test_image)
            
            # Synchronize if using CUDA
            if self.device != "cpu":
                torch.cuda.synchronize()
            
            # Benchmark
            print(f"  Benchmarking ({self.num_test} iterations)...")
            times = []
            
            for _ in range(self.num_test):
                start_time = time.time()
                
                with torch.no_grad():
                    features, tensor = extractor.extract(test_image)
                
                if self.device != "cpu":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            # Cleanup
            extractor.cleanup()
            
            results = {
                'method': method,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_time': std_time,
                'fps': fps,
                'success': True
            }
            
            print(f"  ✅ {method}: {avg_time*1000:.1f}ms avg ({fps:.1f} FPS)")
            return results
            
        except Exception as e:
            print(f"  ❌ {method} failed: {e}")
            return {
                'method': method,
                'avg_time': float('inf'),
                'min_time': float('inf'),
                'max_time': float('inf'),
                'std_time': float('inf'),
                'fps': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def run_full_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run benchmark on all available optimization methods."""
        print("=" * 80)
        print("WINDOWS 11 OPTIMIZATION BENCHMARK")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"FP16: {self.enable_fp16}")
        print(f"Warmup iterations: {self.num_warmup}")
        print(f"Test iterations: {self.num_test}")
        
        results = {}
        
        for method in self.optimization_methods:
            try:
                result = self.benchmark_method(method)
                results[method] = result
            except Exception as e:
                print(f"❌ Failed to benchmark {method}: {e}")
                results[method] = {
                    'method': method,
                    'success': False,
                    'error': str(e)
                }
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Sort by FPS (descending)
        successful_results = [(k, v) for k, v in results.items() if v.get('success', False)]
        successful_results.sort(key=lambda x: x[1]['fps'], reverse=True)
        
        if not successful_results:
            print("❌ No optimization methods succeeded!")
            return
        
        print(f"{'Method':<15} {'Avg Time (ms)':<15} {'FPS':<10} {'Min/Max (ms)':<20} {'Status'}")
        print("-" * 80)
        
        baseline_fps = None
        
        for method, result in successful_results:
            avg_time_ms = result['avg_time'] * 1000
            fps = result['fps']
            min_time_ms = result['min_time'] * 1000
            max_time_ms = result['max_time'] * 1000
            
            if method == "baseline":
                baseline_fps = fps
                improvement = "1.0x"
            elif baseline_fps:
                improvement = f"{fps/baseline_fps:.1f}x"
            else:
                improvement = "-"
            
            print(f"{method:<15} {avg_time_ms:<15.1f} {fps:<10.1f} {min_time_ms:.1f}/{max_time_ms:.1f}{'':>8} ✅ {improvement}")
        
        # Print failed methods
        failed_results = [(k, v) for k, v in results.items() if not v.get('success', False)]
        if failed_results:
            print("\nFailed methods:")
            for method, result in failed_results:
                error = result.get('error', 'Unknown error')
                print(f"{method:<15} {'N/A':<15} {'N/A':<10} {'N/A':<20} ❌ {error}")
        
        # Recommendation
        if successful_results:
            best_method, best_result = successful_results[0]
            print(f"\n🏆 RECOMMENDED: {best_method} ({best_result['fps']:.1f} FPS)")
            
            if baseline_fps and best_result['fps'] > baseline_fps:
                improvement = best_result['fps'] / baseline_fps
                print(f"   Performance improvement: {improvement:.1f}x over baseline")
        
        print("=" * 80)
    
    def get_best_method(self, results: Dict[str, Dict[str, float]]) -> str:
        """Get the best performing optimization method."""
        successful_results = [(k, v) for k, v in results.items() if v.get('success', False)]
        if not successful_results:
            return "baseline"
        
        # Sort by FPS and return the best
        successful_results.sort(key=lambda x: x[1]['fps'], reverse=True)
        return successful_results[0][0]


def run_optimization_benchmark():
    """Run the optimization benchmark and return results."""
    benchmark = OptimizationBenchmark(
        device="auto",
        enable_fp16=True,
        num_warmup=3,
        num_test=10
    )
    
    results = benchmark.run_full_benchmark()
    best_method = benchmark.get_best_method(results)
    
    return results, best_method


if __name__ == "__main__":
    results, best_method = run_optimization_benchmark()
    print(f"\nBest optimization method: {best_method}")
