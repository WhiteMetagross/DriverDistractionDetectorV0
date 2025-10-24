import cv2
import numpy as np
from typing import Optional, Callable
from pathlib import Path
import time
from PIL import Image

from driversafety.detection.optimized_face_detector import OptimizedDriverFaceDetector
from driversafety.classification.optimized_feature_extractor import OptimizedResNetFeatureExtractor
from driversafety.classification.behavior_classifier import BehaviorClassifier
from driversafety.visualization.optimized_gradcam import OptimizedGradCAM
from driversafety.visualization.edges import get_edge_for_visualization
from driversafety.visualization.overlays import overlay_cam_on_image


class VideoExporter:
    """
    Video export utility that processes video files and exports them with
    driver behavior analysis overlays and effects.
    """
    
    def __init__(self, detector: OptimizedDriverFaceDetector,
                 extractor: OptimizedResNetFeatureExtractor,
                 classifier: BehaviorClassifier,
                 resize_width: int = 960):
        self.detector = detector
        self.extractor = extractor
        self.classifier = classifier
        self.resize_width = resize_width

        # Initialize Grad-CAM with correct target layers
        target_layers = [extractor.model.layer4[-1]]
        self.gradcam = OptimizedGradCAM(extractor.model, target_layers)

        # Cache for last generated Grad-CAM overlay (for frame-to-frame continuity)
        self._last_gradcam_overlay = None
        
    def _get_codec_for_format(self, output_path: str):
        """
        Get appropriate codec and fourcc code for the output format.
        Returns a list of (fourcc_code, codec_name) tuples to try in order.
        """
        output_path_lower = output_path.lower()

        if output_path_lower.endswith('.mp4'):
            # For MP4, try multiple codecs in order of preference
            return [
                ('mp4v', 'MP4V'),
                ('H264', 'H.264'),
                ('MJPG', 'Motion JPEG'),
                ('XVID', 'XVID'),
            ]
        elif output_path_lower.endswith('.avi'):
            # For AVI, XVID is most reliable on Windows
            return [
                ('XVID', 'XVID'),
                ('MJPG', 'Motion JPEG'),
                ('DIVX', 'DivX'),
            ]
        elif output_path_lower.endswith('.mov'):
            # For MOV, try MP4V and MJPG
            return [
                ('mp4v', 'MP4V'),
                ('MJPG', 'Motion JPEG'),
            ]
        elif output_path_lower.endswith('.mkv'):
            # For MKV, try XVID and MJPG
            return [
                ('XVID', 'XVID'),
                ('MJPG', 'Motion JPEG'),
            ]
        else:
            # Default fallback
            return [
                ('XVID', 'XVID'),
                ('MJPG', 'Motion JPEG'),
                ('mp4v', 'MP4V'),
            ]

    def _create_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """
        Create a VideoWriter with fallback codec support.
        Returns (VideoWriter, codec_name) or (None, None) if all codecs fail.
        """
        codec_options = self._get_codec_for_format(output_path)

        for fourcc_str, codec_name in codec_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                if writer.isOpened():
                    print(f"Successfully created VideoWriter with codec: {codec_name} ({fourcc_str})")
                    return writer, codec_name
                else:
                    print(f"Failed to open VideoWriter with codec: {codec_name} ({fourcc_str})")
            except Exception as e:
                print(f"Error trying codec {codec_name} ({fourcc_str}): {e}")

        print(f"Error: Could not create VideoWriter with any available codec for: {output_path}")
        return None, None

    def export_video(self, input_path: str, output_path: str,
                    progress_callback: Optional[Callable[[int], None]] = None,
                    include_gradcam: bool = True,
                    include_classification: bool = True,
                    quality: str = "high") -> bool:
        """
        Export processed video with analysis overlays.

        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            progress_callback: Optional callback for progress updates (0-100)
            include_gradcam: Whether to include Grad-CAM heatmap overlay
            include_classification: Whether to include classification text overlay
            quality: Export quality ("low", "medium", "high")

        Returns:
            True if export successful, False otherwise
        """
        cap = None
        out = None

        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Could not open input video: {input_path}")
                return False

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Default FPS if not available

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width <= 0 or height <= 0:
                print(f"Error: Invalid video dimensions: {width}x{height}")
                cap.release()
                return False

            # Resize dimensions if needed
            if width > self.resize_width:
                scale = self.resize_width / width
                width = int(width * scale)
                height = int(height * scale)

                # Ensure dimensions are even (required for many codecs)
                if width % 2 != 0:
                    width -= 1
                if height % 2 != 0:
                    height -= 1

            print(f"Exporting video: {input_path} -> {output_path}")
            print(f"Properties: {width}x{height}, {fps} FPS, {total_frames} frames")

            # Create video writer with fallback codec support
            out, codec_name = self._create_video_writer(output_path, fps, width, height)

            if out is None:
                cap.release()
                return False

            frame_count = 0
            start_time = time.time()
            write_errors = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame if needed
                if frame.shape[1] > self.resize_width or frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))

                # Ensure frame has correct dimensions
                if frame.shape[0] != height or frame.shape[1] != width:
                    print(f"Warning: Frame dimension mismatch. Expected {width}x{height}, got {frame.shape[1]}x{frame.shape[0]}")
                    frame = cv2.resize(frame, (width, height))

                # Process frame
                try:
                    processed_frame = self._process_frame(
                        frame, include_gradcam, include_classification
                    )
                except Exception as e:
                    print(f"Warning: Frame processing failed for frame {frame_count}: {e}")
                    processed_frame = frame

                # Write frame
                # Note: out.write() returns None in many cases even when successful,
                # so we only track actual exceptions, not return values
                try:
                    out.write(processed_frame)
                    # Reset error counter on successful write (no exception)
                    write_errors = 0
                except Exception as e:
                    print(f"Error writing frame {frame_count}: {e}")
                    write_errors += 1
                    if write_errors > 10:
                        print(f"Error: Too many frame write failures ({write_errors}), aborting export")
                        break

                frame_count += 1

                # Update progress
                if progress_callback and total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    progress_callback(progress)

                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({fps_actual:.1f} FPS)")

            # Cleanup
            cap.release()
            out.release()

            elapsed = time.time() - start_time
            print(f"Export completed in {elapsed:.1f} seconds")
            print(f"Output saved to: {output_path}")
            print(f"Total frames written: {frame_count}")

            # Verify output file was created
            output_file = Path(output_path)
            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"Output file verified: {output_file.stat().st_size} bytes")
                return True
            else:
                print(f"Error: Output file was not created or is empty")
                return False

        except Exception as e:
            print(f"Error during video export: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Ensure resources are cleaned up
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            
    def _process_frame(self, frame: np.ndarray, include_gradcam: bool, 
                      include_classification: bool) -> np.ndarray:
        """
        Process a single frame with analysis overlays.
        
        Args:
            frame: Input frame (BGR format)
            include_gradcam: Whether to include Grad-CAM overlay
            include_classification: Whether to include classification text
            
        Returns:
            Processed frame with overlays
        """
        try:
            # Detect faces
            boxes = self.detector.detect(frame)
            
            if len(boxes) == 0:
                # No face detected, return original frame with text
                if include_classification:
                    cv2.putText(frame, "No face detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame
                
            # Use the first detected face
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract face region for classification
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                return frame
                
            # Convert to PIL Image for feature extraction
            face_pil = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
            
            # Extract features and classify (compatible with OptimizedResNetFeatureExtractor)
            features, input_tensor = self.extractor.extract(face_pil)
            if features is not None:
                # Handle both numpy and tensor features
                if hasattr(features, 'cpu'):
                    features_np = features.cpu().numpy()
                else:
                    features_np = features

                # BehaviorClassifier.predict() returns (idx, label, confidence)
                prediction = self.classifier.predict(features_np)
                class_idx = int(prediction[0])  # Integer class index
                class_label = str(prediction[1])  # String label
                confidence = float(prediction[2]) if len(prediction) > 2 and prediction[2] is not None else 0.0  # Float confidence

                # Add classification text overlay
                if include_classification:
                    text = f"{class_label} ({confidence:.2f})"
                    cv2.putText(frame, text, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add Grad-CAM overlay positioned in top-right corner
                if include_gradcam:
                    try:
                        # Import required modules for Grad-CAM processing
                        from pytorch_grad_cam import GradCAM
                        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                        from driversafety.visualization.edges import get_edge_for_visualization
                        from driversafety.visualization.overlays import overlay_cam_on_image

                        # Generate Grad-CAM using the same approach as the workers
                        target_layers = [self.extractor.model.layer4[-1]]
                        cam = GradCAM(model=self.extractor.model, target_layers=target_layers)
                        targets = [ClassifierOutputTarget(class_idx)]
                        heatmap = cam(input_tensor=input_tensor, targets=targets)

                        if heatmap is not None and len(heatmap) > 0:
                            # Use edge visualization for better visual quality
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            edge_img = get_edge_for_visualization(Image.fromarray(frame_rgb))
                            overlay_rgb = overlay_cam_on_image(edge_img, heatmap[0])

                            # Get frame dimensions
                            frame_height, frame_width = frame.shape[:2]

                            # Calculate overlay size: 15% of frame width (half of previous 30%)
                            overlay_width = int(frame_width * 0.15)
                            # Maintain 4:3 aspect ratio
                            overlay_height = int(overlay_width * 0.75)

                            # Resize overlay to standardized size
                            pip = cv2.resize(overlay_rgb, (overlay_width, overlay_height))

                            # Cache the overlay for frame-to-frame continuity
                            self._last_gradcam_overlay = cv2.cvtColor(pip, cv2.COLOR_RGB2BGR)

                    except Exception as e:
                        print(f"Warning: Grad-CAM generation failed: {e}")
                        import traceback
                        traceback.print_exc()

            # Apply Grad-CAM overlay (either newly generated or cached from previous frame)
            if include_gradcam and self._last_gradcam_overlay is not None:
                try:
                    frame_height, frame_width = frame.shape[:2]
                    ph, pw, _ = self._last_gradcam_overlay.shape

                    # Position in top-right corner
                    margin = 10
                    y0 = margin
                    x0 = frame_width - pw - margin
                    y1_pip, x1_pip = y0 + ph, x0 + pw

                    # Ensure overlay fits within frame boundaries
                    if y1_pip <= frame_height and x1_pip <= frame_width and x0 >= 0 and y0 >= 0:
                        frame[y0:y1_pip, x0:x1_pip] = self._last_gradcam_overlay

                except Exception as e:
                    print(f"Warning: Failed to apply Grad-CAM overlay: {e}")
                        
            return frame
            
        except Exception as e:
            print(f"Warning: Frame processing failed: {e}")
            return frame
            
    def export_frame(self, frame_data, output_path: str, 
                    include_gradcam: bool = True,
                    include_classification: bool = True) -> bool:
        """
        Export a single frame as an image.
        
        Args:
            frame_data: Frame data (numpy array or QImage)
            output_path: Path for output image file
            include_gradcam: Whether to include Grad-CAM overlay
            include_classification: Whether to include classification text
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Convert frame data to numpy array if needed
            if hasattr(frame_data, 'bits'):  # QImage
                # Convert QImage to numpy array
                width = frame_data.width()
                height = frame_data.height()
                ptr = frame_data.bits()
                arr = np.array(ptr).reshape(height, width, 3)  # Assuming RGB
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                frame = frame_data
                
            # Process frame if requested
            if include_gradcam or include_classification:
                frame = self._process_frame(frame, include_gradcam, include_classification)
                
            # Save frame
            success = cv2.imwrite(output_path, frame)
            
            if success:
                print(f"Frame exported to: {output_path}")
            else:
                print(f"Failed to save frame to: {output_path}")
                
            return success
            
        except Exception as e:
            print(f"Error exporting frame: {e}")
            return False
            
    @staticmethod
    def get_supported_formats():
        """Get list of supported video formats for export."""
        return {
            'mp4': 'MP4 Video (*.mp4)',
            'avi': 'AVI Video (*.avi)',
            'mov': 'QuickTime Movie (*.mov)',
            'mkv': 'Matroska Video (*.mkv)'
        }
        
    @staticmethod
    def get_quality_settings():
        """Get available quality settings."""
        return {
            'low': {'bitrate': '1M', 'crf': 28},
            'medium': {'bitrate': '2M', 'crf': 23},
            'high': {'bitrate': '5M', 'crf': 18}
        }
