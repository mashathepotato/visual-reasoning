import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import cv2
from scipy import interpolate
from scipy.ndimage import rotate
import random
from typing import List, Tuple, Dict, Any
import os
from pathlib import Path

class LineGenerator:
    """Generate various types of lines for dataset creation"""
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256), 
                 line_color: Tuple[int, int, int] = (0, 0, 139),  # Navy blue
                 background_color: Tuple[int, int, int] = (255, 255, 255)):  # White
        self.image_size = image_size
        self.line_color = line_color
        self.background_color = background_color
        
    def create_blank_image(self) -> np.ndarray:
        """Create a blank white image"""
        return np.full((*self.image_size, 3), self.background_color, dtype=np.uint8)
    
    def draw_straight_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                          thickness: int = 2) -> np.ndarray:
        """Generate a straight line"""
        img = self.create_blank_image()
        cv2.line(img, start, end, self.line_color, thickness)
        return img
    
    def draw_curved_line(self, control_points: List[Tuple[int, int]], 
                        thickness: int = 2, num_points: int = 100) -> np.ndarray:
        """Generate a curved line using Bezier curves"""
        img = self.create_blank_image()
        
        if len(control_points) < 2:
            return img
            
        # Create Bezier curve
        t = np.linspace(0, 1, num_points)
        points = []
        
        for i in range(len(control_points) - 1):
            if i == 0:
                # First segment
                for j in range(len(t)):
                    point = self._bezier_curve(control_points[i], control_points[i+1], t[j])
                    points.append(point)
            else:
                # Subsequent segments
                for j in range(1, len(t)):  # Skip first point to avoid duplication
                    point = self._bezier_curve(control_points[i], control_points[i+1], t[j])
                    points.append(point)
        
        # Draw the curve
        for i in range(len(points) - 1):
            cv2.line(img, 
                    (int(points[i][0]), int(points[i][1])), 
                    (int(points[i+1][0]), int(points[i+1][1])), 
                    self.line_color, thickness)
        
        return img
    
    def _bezier_curve(self, p0: Tuple[int, int], p1: Tuple[int, int], t: float) -> Tuple[float, float]:
        """Simple linear Bezier curve between two points"""
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
    
    def draw_wavy_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                      amplitude: int = 20, frequency: float = 0.1, 
                      thickness: int = 2) -> np.ndarray:
        """Generate a wavy line"""
        img = self.create_blank_image()
        
        # Calculate number of points based on distance
        distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        num_points = max(50, int(distance / 2))
        
        # Generate wavy path
        t = np.linspace(0, 1, num_points)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        # Add wave perpendicular to the line direction
        line_angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        perp_angle = line_angle + np.pi/2
        
        wave_x = amplitude * np.sin(2 * np.pi * frequency * t * num_points) * np.cos(perp_angle)
        wave_y = amplitude * np.sin(2 * np.pi * frequency * t * num_points) * np.sin(perp_angle)
        
        x += wave_x
        y += wave_y
        
        # Draw the wavy line
        for i in range(len(x) - 1):
            cv2.line(img, 
                    (int(x[i]), int(y[i])), 
                    (int(x[i+1]), int(y[i+1])), 
                    self.line_color, thickness)
        
        return img
    
    def draw_ragged_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                        roughness: float = 0.3, thickness: int = 2) -> np.ndarray:
        """Generate a ragged/jagged line"""
        img = self.create_blank_image()
        
        # Calculate number of points
        distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        num_points = max(20, int(distance / 5))
        
        # Generate base line
        t = np.linspace(0, 1, num_points)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        # Add random roughness
        max_roughness = distance * roughness
        x += np.random.normal(0, max_roughness/3, num_points)
        y += np.random.normal(0, max_roughness/3, num_points)
        
        # Draw the ragged line
        for i in range(len(x) - 1):
            cv2.line(img, 
                    (int(x[i]), int(y[i])), 
                    (int(x[i+1]), int(y[i+1])), 
                    self.line_color, thickness)
        
        return img
    
    def draw_looped_line(self, center: Tuple[int, int], radius: int, 
                        start_angle: float = 0, end_angle: float = 2*np.pi,
                        thickness: int = 2) -> np.ndarray:
        """Generate a looped/circular line"""
        img = self.create_blank_image()
        
        # Generate circular path
        num_points = max(50, int(2 * np.pi * radius / 2))
        angles = np.linspace(start_angle, end_angle, num_points)
        
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        # Draw the looped line
        for i in range(len(x) - 1):
            cv2.line(img, 
                    (int(x[i]), int(y[i])), 
                    (int(x[i+1]), int(y[i+1])), 
                    self.line_color, thickness)
        
        return img

class DatasetGenerator:
    """Generate dataset with rotated line samples"""
    
    def __init__(self, line_generator: LineGenerator):
        self.line_generator = line_generator
        self.image_size = line_generator.image_size
    
    def generate_straight_line_dataset(self, num_samples: int = 5, 
                                     rotation_step: int = 10) -> List[Dict[str, Any]]:
        """Generate dataset of straight lines with rotations"""
        dataset = []
        
        for sample_id in range(num_samples):
            # Random line parameters
            start = (random.randint(50, 100), random.randint(50, 100))
            end = (random.randint(150, 200), random.randint(150, 200))
            thickness = random.choice([2, 3, 4])
            
            # Generate base line
            base_img = self.line_generator.draw_straight_line(start, end, thickness)
            
            # Generate rotations
            for angle in range(0, 360, rotation_step):
                rotated_img = self._rotate_image(base_img, angle)
                dataset.append({
                    'sample_id': sample_id,
                    'line_type': 'straight',
                    'angle': angle,
                    'image': rotated_img,
                    'base_params': {'start': start, 'end': end, 'thickness': thickness}
                })
        
        return dataset
    
    def generate_curved_line_dataset(self, num_samples: int = 5, 
                                    rotation_step: int = 10) -> List[Dict[str, Any]]:
        """Generate dataset of curved lines with rotations"""
        dataset = []
        
        for sample_id in range(num_samples):
            # Random curve parameters
            num_control_points = random.randint(3, 6)
            control_points = []
            for _ in range(num_control_points):
                point = (random.randint(50, 200), random.randint(50, 200))
                control_points.append(point)
            
            thickness = random.choice([2, 3, 4])
            
            # Generate base curve
            base_img = self.line_generator.draw_curved_line(control_points, thickness)
            
            # Generate rotations
            for angle in range(0, 360, rotation_step):
                rotated_img = self._rotate_image(base_img, angle)
                dataset.append({
                    'sample_id': sample_id,
                    'line_type': 'curved',
                    'angle': angle,
                    'image': rotated_img,
                    'base_params': {'control_points': control_points, 'thickness': thickness}
                })
        
        return dataset
    
    def generate_wavy_line_dataset(self, num_samples: int = 5, 
                                  rotation_step: int = 10) -> List[Dict[str, Any]]:
        """Generate dataset of wavy lines with rotations"""
        dataset = []
        
        for sample_id in range(num_samples):
            # Random wavy line parameters
            start = (random.randint(50, 100), random.randint(50, 100))
            end = (random.randint(150, 200), random.randint(150, 200))
            amplitude = random.randint(10, 30)
            frequency = random.uniform(0.05, 0.2)
            thickness = random.choice([2, 3, 4])
            
            # Generate base wavy line
            base_img = self.line_generator.draw_wavy_line(start, end, amplitude, frequency, thickness)
            
            # Generate rotations
            for angle in range(0, 360, rotation_step):
                rotated_img = self._rotate_image(base_img, angle)
                dataset.append({
                    'sample_id': sample_id,
                    'line_type': 'wavy',
                    'angle': angle,
                    'image': rotated_img,
                    'base_params': {'start': start, 'end': end, 'amplitude': amplitude, 
                                  'frequency': frequency, 'thickness': thickness}
                })
        
        return dataset
    
    def generate_ragged_line_dataset(self, num_samples: int = 5, 
                                    rotation_step: int = 10) -> List[Dict[str, Any]]:
        """Generate dataset of ragged lines with rotations"""
        dataset = []
        
        for sample_id in range(num_samples):
            # Random ragged line parameters
            start = (random.randint(50, 100), random.randint(50, 100))
            end = (random.randint(150, 200), random.randint(150, 200))
            roughness = random.uniform(0.1, 0.5)
            thickness = random.choice([2, 3, 4])
            
            # Generate base ragged line
            base_img = self.line_generator.draw_ragged_line(start, end, roughness, thickness)
            
            # Generate rotations
            for angle in range(0, 360, rotation_step):
                rotated_img = self._rotate_image(base_img, angle)
                dataset.append({
                    'sample_id': sample_id,
                    'line_type': 'ragged',
                    'angle': angle,
                    'image': rotated_img,
                    'base_params': {'start': start, 'end': end, 'roughness': roughness, 
                                  'thickness': thickness}
                })
        
        return dataset
    
    def generate_looped_line_dataset(self, num_samples: int = 5, 
                                    rotation_step: int = 10) -> List[Dict[str, Any]]:
        """Generate dataset of looped lines with rotations"""
        dataset = []
        
        for sample_id in range(num_samples):
            # Random loop parameters
            center = (random.randint(100, 150), random.randint(100, 150))
            radius = random.randint(30, 60)
            start_angle = random.uniform(0, np.pi)
            end_angle = start_angle + random.uniform(np.pi, 2*np.pi)
            thickness = random.choice([2, 3, 4])
            
            # Generate base loop
            base_img = self.line_generator.draw_looped_line(center, radius, start_angle, end_angle, thickness)
            
            # Generate rotations
            for angle in range(0, 360, rotation_step):
                rotated_img = self._rotate_image(base_img, angle)
                dataset.append({
                    'sample_id': sample_id,
                    'line_type': 'looped',
                    'angle': angle,
                    'image': rotated_img,
                    'base_params': {'center': center, 'radius': radius, 'start_angle': start_angle, 
                                  'end_angle': end_angle, 'thickness': thickness}
                })
        
        return dataset
    
    def generate_mixed_dataset(self, samples_per_type: int = 3, 
                              rotation_step: int = 15) -> List[Dict[str, Any]]:
        """Generate a mixed dataset with all line types"""
        all_datasets = []
        
        # Generate datasets for each line type
        all_datasets.extend(self.generate_straight_line_dataset(samples_per_type, rotation_step))
        all_datasets.extend(self.generate_curved_line_dataset(samples_per_type, rotation_step))
        all_datasets.extend(self.generate_wavy_line_dataset(samples_per_type, rotation_step))
        all_datasets.extend(self.generate_ragged_line_dataset(samples_per_type, rotation_step))
        all_datasets.extend(self.generate_looped_line_dataset(samples_per_type, rotation_step))
        
        return all_datasets
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image around its center"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                               borderValue=self.line_generator.background_color)
        
        return rotated

class Visualizer:
    """Visualize generated datasets"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
    
    def plot_sample_rotations(self, dataset: List[Dict[str, Any]], 
                            sample_id: int, line_type: str, 
                            max_angles: int = 8) -> None:
        """Plot rotations of a specific sample"""
        # Filter dataset for specific sample and line type
        sample_data = [item for item in dataset 
                      if item['sample_id'] == sample_id and item['line_type'] == line_type]
        
        if not sample_data:
            print(f"No data found for sample_id={sample_id}, line_type={line_type}")
            return
        
        # Select angles to display
        angles = sorted([item['angle'] for item in sample_data])
        selected_angles = angles[::len(angles)//max_angles] if len(angles) > max_angles else angles
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=self.figsize)
        axes = axes.flatten()
        
        for i, angle in enumerate(selected_angles[:8]):
            if i >= len(axes):
                break
                
            # Find the data for this angle
            angle_data = next((item for item in sample_data if item['angle'] == angle), None)
            if angle_data:
                axes[i].imshow(angle_data['image'])
                axes[i].set_title(f'Angle: {angle}°')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(selected_angles), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Sample {sample_id} - {line_type.title()} Line Rotations')
        plt.tight_layout()
        plt.show()
    
    def plot_line_type_comparison(self, dataset: List[Dict[str, Any]], 
                                 angle: int = 0) -> None:
        """Plot different line types at the same angle"""
        # Filter dataset for specific angle
        angle_data = [item for item in dataset if item['angle'] == angle]
        
        if not angle_data:
            print(f"No data found for angle={angle}")
            return
        
        # Group by line type
        line_types = {}
        for item in angle_data:
            line_type = item['line_type']
            if line_type not in line_types:
                line_types[line_type] = []
            line_types[line_type].append(item)
        
        # Create subplot
        fig, axes = plt.subplots(1, len(line_types), figsize=self.figsize)
        if len(line_types) == 1:
            axes = [axes]
        
        for i, (line_type, items) in enumerate(line_types.items()):
            # Take first sample of this line type
            sample = items[0]
            axes[i].imshow(sample['image'])
            axes[i].set_title(f'{line_type.title()} Line')
            axes[i].axis('off')
        
        plt.suptitle(f'Line Types at {angle}°')
        plt.tight_layout()
        plt.show()
    
    def plot_overlay_comparison(self, dataset: List[Dict[str, Any]], 
                              sample_id: int, line_type: str, 
                              angle1: int, angle2: int) -> None:
        """Plot overlay comparison of two rotations"""
        # Find the two images
        img1 = next((item for item in dataset 
                    if item['sample_id'] == sample_id and item['line_type'] == line_type 
                    and item['angle'] == angle1), None)
        img2 = next((item for item in dataset 
                    if item['sample_id'] == sample_id and item['line_type'] == line_type 
                    and item['angle'] == angle2), None)
        
        if not img1 or not img2:
            print(f"Could not find images for comparison")
            return
        
        # Create overlay
        overlay = cv2.addWeighted(img1['image'], 0.5, img2['image'], 0.5, 0)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img1['image'])
        axes[0].set_title(f'Angle {angle1}°')
        axes[0].axis('off')
        
        axes[1].imshow(img2['image'])
        axes[1].set_title(f'Angle {angle2}°')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay (50% each)')
        axes[2].axis('off')
        
        plt.suptitle(f'Sample {sample_id} - {line_type.title()} Line Comparison')
        plt.tight_layout()
        plt.show()

# Example usage and testing functions
def test_line_generation():
    """Test individual line generation functions"""
    generator = LineGenerator()
    
    # Test straight line
    straight = generator.draw_straight_line((50, 50), (200, 200), 3)
    
    # Test curved line
    control_points = [(50, 50), (100, 150), (150, 100), (200, 200)]
    curved = generator.draw_curved_line(control_points, 3)
    
    # Test wavy line
    wavy = generator.draw_wavy_line((50, 50), (200, 200), 20, 0.1, 3)
    
    # Test ragged line
    ragged = generator.draw_ragged_line((50, 50), (200, 200), 0.3, 3)
    
    # Test looped line
    looped = generator.draw_looped_line((128, 128), 50, 0, 2*np.pi, 3)
    
    # Plot all line types
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    line_types = [straight, curved, wavy, ragged, looped]
    titles = ['Straight', 'Curved', 'Wavy', 'Ragged', 'Looped']
    
    for i, (img, title) in enumerate(zip(line_types, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_and_visualize_dataset():
    """Generate a complete dataset and visualize it"""
    # Initialize components
    line_gen = LineGenerator()
    dataset_gen = DatasetGenerator(line_gen)
    visualizer = Visualizer()
    
    # Generate mixed dataset
    print("Generating mixed dataset...")
    dataset = dataset_gen.generate_mixed_dataset(samples_per_type=2, rotation_step=30)
    print(f"Generated {len(dataset)} samples")
    
    # Visualize different line types
    print("\nVisualizing line types at 0°...")
    visualizer.plot_line_type_comparison(dataset, angle=0)
    
    # Visualize rotations for each line type
    line_types = ['straight', 'curved', 'wavy', 'ragged', 'looped']
    for line_type in line_types:
        print(f"\nVisualizing {line_type} line rotations...")
        visualizer.plot_sample_rotations(dataset, sample_id=0, line_type=line_type)
    
    # Test overlay comparison
    print("\nTesting overlay comparison...")
    visualizer.plot_overlay_comparison(dataset, sample_id=0, line_type='straight', 
                                     angle1=0, angle2=90)
    
    return dataset

if __name__ == "__main__":
    # Test individual line generation
    print("Testing line generation...")
    test_line_generation()
    
    # Generate and visualize complete dataset
    print("\nGenerating complete dataset...")
    dataset = generate_and_visualize_dataset()
