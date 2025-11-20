import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class CinemaPricingNet(nn.Module):
    def __init__(self):
        super(CinemaPricingNet, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 12)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CinemaLoss(nn.Module):
    def __init__(self, attendance_weight=0.6, profit_weight=0.4):
        super(CinemaLoss, self).__init__()
        self.attendance_weight = attendance_weight
        self.profit_weight = profit_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, actual_occupancy, current_discounts):
        classification_loss = self.ce_loss(predictions, targets)
        
        predicted_classes = torch.argmax(predictions, dim=1)
        predicted_discounts = (predicted_classes + 1) * 5
        
        attendance_benefit = torch.mean(actual_occupancy / 100)
        profit_preservation = 1 - torch.mean((predicted_discounts - 5) / 55)
        
        business_objective = (self.attendance_weight * attendance_benefit + 
                            self.profit_weight * profit_preservation)
        
        total_loss = classification_loss - business_objective
        return total_loss

class CinemaDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.tensor([
            (sample['time_slot'] - 1) / 7,
            sample['movie_expectations'],
            sample['expected_occupancy'] / 100,
            sample['actual_occupancy'] / 100,
            sample['price_reduction'] / 60
        ], dtype=torch.float32)
        
        target = torch.tensor(sample['discount_class'], dtype=torch.long)
        return features, target, sample['actual_occupancy'], sample['price_reduction']

def create_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        time_slot = np.random.randint(1, 9)
        movie_expectations = np.random.random()
        expected_occupancy = np.random.uniform(20, 95)
        actual_occupancy = np.random.uniform(10, 100)
        
        occupancy_gap = max(0, expected_occupancy - actual_occupancy)
        discount_factor = (occupancy_gap / 100) * 0.5 + (1 - movie_expectations) * 0.3
        discount_class = min(11, int(discount_factor * 12))
        price_reduction = (discount_class + 1) * 5
        
        data.append({
            'time_slot': time_slot,
            'movie_expectations': movie_expectations,
            'expected_occupancy': expected_occupancy,
            'actual_occupancy': actual_occupancy,
            'price_reduction': price_reduction,
            'discount_class': discount_class
        })
    return data

class NetworkVisualizer:
    def __init__(self, model):
        self.model = model
        self.layer_sizes = [5, 16, 16, 12]
        self.layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        self.feature_names = ['Time', 'Movie Exp', 'Exp Occ', 'Act Occ', 'Price Red']
        
        # Position neurons
        self.neuron_positions = []
        layer_x = [1, 3, 5, 7]
        
        for i, (size, x) in enumerate(zip(self.layer_sizes, layer_x)):
            y_positions = np.linspace(1, 6, size)
            positions = [(x, y) for y in y_positions]
            self.neuron_positions.append(positions)
        
        self.activations = [np.zeros(size) for size in self.layer_sizes]
        
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(0, 7)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Title
        self.ax.text(4, 6.5, 'Cinema Pricing MLP - Training & Inference', 
                    fontsize=14, ha='center', weight='bold')
        
        # Draw connections
        self.connection_lines = []
        for i in range(len(self.layer_sizes) - 1):
            for pos1 in self.neuron_positions[i]:
                for pos2 in self.neuron_positions[i + 1]:
                    line, = self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                                       'lightgray', alpha=0.4, linewidth=0.3)
                    self.connection_lines.append(line)
        
        # Draw neurons
        self.neuron_circles = []
        for layer_idx, positions in enumerate(self.neuron_positions):
            layer_circles = []
            for neuron_idx, (x, y) in enumerate(positions):
                circle = Circle((x, y), 0.08, color='lightblue', ec='black', linewidth=0.5)
                self.ax.add_patch(circle)
                layer_circles.append(circle)
                
                # Labels for input neurons
                if layer_idx == 0:
                    self.ax.text(x - 0.3, y, self.feature_names[neuron_idx], 
                               fontsize=7, ha='right', va='center')
                # Labels for output neurons
                elif layer_idx == len(self.layer_sizes) - 1:
                    self.ax.text(x + 0.3, y, f'C{neuron_idx}', 
                               fontsize=7, ha='left', va='center')
            
            self.neuron_circles.append(layer_circles)
            
            # Layer titles
            y_center = np.mean([pos[1] for pos in positions])
            self.ax.text(positions[0][0], y_center - 0.6, self.layer_names[layer_idx], 
                        fontsize=9, ha='center', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.7))
        
        # Info display
        self.info_text = self.ax.text(0.2, 0.5, '', fontsize=9, va='top',
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        
        self.loss_text = self.ax.text(7.5, 0.5, '', fontsize=9, va='top',
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
        
    def update_display(self, activations_list, mode="training", loss=None, epoch=None):
        # Ensure we have the right number of layers
        if len(activations_list) != len(self.layer_sizes):
            print(f"Warning: Expected {len(self.layer_sizes)} layers, got {len(activations_list)}")
            return
            
        # Update neuron colors based on activations - SOLID COLORS
        normalized_activations = []
        for layer_idx, activations in enumerate(activations_list):
            if activations is not None and layer_idx < len(self.neuron_circles):
                if torch.is_tensor(activations):
                    activations = activations.detach().cpu().numpy()
                if activations.ndim > 1:
                    activations = activations.flatten()
                
                # Normalize activations
                if len(activations) > 0:
                    act_min, act_max = activations.min(), activations.max()
                    if act_max > act_min:
                        normalized = (activations - act_min) / (act_max - act_min)
                    else:
                        normalized = np.ones_like(activations) * 0.5
                else:
                    normalized = np.array([])
                
                normalized_activations.append(normalized)
                
                # Color neurons with SOLID COLORS
                for neuron_idx, circle in enumerate(self.neuron_circles[layer_idx]):
                    if neuron_idx < len(normalized):
                        intensity = max(0.3, normalized[neuron_idx])  # Minimum intensity for visibility
                        if layer_idx == 0:  # Input - Green
                            color = (0, intensity, 0, 1.0)  # Solid green
                        elif layer_idx == len(self.layer_sizes) - 1:  # Output - Red
                            color = (intensity, 0, 0, 1.0)  # Solid red
                        else:  # Hidden - Blue
                            color = (0, 0, intensity, 1.0)  # Solid blue
                        circle.set_facecolor(color)
                        circle.set_alpha(1.0)  # Fully opaque
            else:
                normalized_activations.append(np.array([]))
        
        # Update connection lighting based on activations
        connection_idx = 0
        for layer_idx in range(len(self.layer_sizes) - 1):
            if layer_idx < len(normalized_activations) and (layer_idx + 1) < len(normalized_activations):
                source_activations = normalized_activations[layer_idx]
                target_activations = normalized_activations[layer_idx + 1]
                
                for source_idx in range(len(self.neuron_positions[layer_idx])):
                    for target_idx in range(len(self.neuron_positions[layer_idx + 1])):
                        if connection_idx < len(self.connection_lines):
                            # Calculate connection intensity based on source and target activations
                            source_intensity = source_activations[source_idx] if source_idx < len(source_activations) else 0.1
                            target_intensity = target_activations[target_idx] if target_idx < len(target_activations) else 0.1
                            connection_intensity = (source_intensity + target_intensity) / 2
                            
                            # Light up connection based on intensity
                            alpha = max(0.1, connection_intensity)
                            color = plt.cm.plasma(connection_intensity)
                            self.connection_lines[connection_idx].set_color(color)
                            self.connection_lines[connection_idx].set_alpha(alpha)
                            self.connection_lines[connection_idx].set_linewidth(0.3 + connection_intensity * 1.5)
                            
                        connection_idx += 1
        
        # Update text displays
        mode_display = "TRAINING" if mode == "training" else "INFERENCE"
        info = f"Mode: {mode_display}\n"
        if epoch is not None:
            info += f"Epoch: {epoch}\n"
        if loss is not None:
            info += f"Loss: {loss:.4f}"
        self.info_text.set_text(info)
        
        if loss is not None:
            self.loss_text.set_text(f"Loss: {loss:.4f}\n{'Decreasing' if loss < 1.0 else 'High'}")

def get_layer_activations(model, input_tensor):
    """Extract activations from each layer - Input, FC1 output, FC2 output, FC3 output"""
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy().copy())
    
    # Register hooks for each layer
    hooks = []
    hooks.append(model.fc1.register_forward_hook(hook_fn))
    hooks.append(model.fc2.register_forward_hook(hook_fn))
    hooks.append(model.fc3.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Return: [input, fc1_out, fc2_out, fc3_out] - exactly 4 layers
    input_acts = input_tensor.detach().cpu().numpy()
    return [input_acts] + activations

def create_visualization():
    # Initialize model and data
    model = CinemaPricingNet()
    criterion = CinemaLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_data = create_data(100)
    train_dataset = CinemaDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Setup visualizer
    visualizer = NetworkVisualizer(model)
    visualizer.setup_plot()
    
    # Collect animation frames
    frames = []
    
    print("="*60)
    print("CINEMA PRICING NEURAL NETWORK - TRAINING DATA")
    print("="*60)
    print(f"Model Architecture: {visualizer.layer_sizes}")
    print(f"Total Training Samples: {len(train_data)}")
    print(f"Batch Size: 8")
    print("="*60)
    
    # Training phase
    model.train()
    for epoch in range(100):
        epoch_losses = []
        batch_count = 0
        
        print(f"\n--- EPOCH {epoch + 1}/15 ---")
        
        for features, targets, actual_occ, current_disc in train_loader:
            optimizer.zero_grad()
            
            # Print training data details
            print(f"Batch {batch_count + 1}:")
            for i in range(min(3, len(features))):  # Show first 3 samples
                sample_data = features[i].numpy()
                print(f"  Sample {i+1}: Time={sample_data[0]:.3f}, MovieExp={sample_data[1]:.3f}, "
                      f"ExpOcc={sample_data[2]:.3f}, ActOcc={sample_data[3]:.3f}, PriceRed={sample_data[4]:.3f}")
                print(f"           Target Class: {targets[i].item()}, Actual Occupancy: {actual_occ[i]:.1f}%")
            
            # Get activations for first sample
            sample_activations = get_layer_activations(model, features[0:1])
            
            # Forward and backward pass
            outputs = model(features)
            loss = criterion(outputs, targets, actual_occ, current_disc)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            print(f"  Batch Loss: {loss.item():.4f}")
            
            # Store frame every few batches
            if batch_count % 2 == 0:
                frames.append({
                    'activations': [act[0] if len(act.shape) > 1 else act for act in sample_activations],
                    'mode': 'training',
                    'loss': loss.item(),
                    'epoch': epoch + 1
                })
            
            batch_count += 1
            if batch_count >= 3:
                break
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    
    print("\n" + "="*60)
    print("SWITCHING TO INFERENCE MODE")
    print("="*60)
    
    # Inference phase
    model.eval()
    test_inputs = [
        [0.7, 0.2, 0.6, 0.3, 0.2],  # Low occupancy scenario
        [0.3, 0.8, 0.8, 0.9, 0.1],  # High occupancy scenario
        [0.5, 0.5, 0.5, 0.5, 0.15], # Medium scenario
    ]
    
    test_scenarios = ["Low Occupancy", "High Occupancy", "Medium Scenario"]
    
    for idx, test_input in enumerate(test_inputs):
        test_tensor = torch.tensor(test_input, dtype=torch.float32).unsqueeze(0)
        activations = get_layer_activations(model, test_tensor)
        
        with torch.no_grad():
            output = model(test_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        
        print(f"\nTest {idx + 1} - {test_scenarios[idx]}:")
        print(f"  Input: {test_input}")
        print(f"  Predicted Discount Class: {predicted_class}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Suggested Discount: {(predicted_class + 1) * 5}%")
        
        for _ in range(2):  # Multiple frames per inference
            frames.append({
                'activations': [act[0] if len(act.shape) > 1 else act for act in activations],
                'mode': 'inference',
                'loss': None,
                'epoch': None
            })
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATION...")
    print("="*60)
    
    # Animation function
    def animate(frame_num):
        if frame_num < len(frames):
            frame = frames[frame_num]
            visualizer.update_display(
                frame['activations'],
                frame['mode'],
                frame['loss'],
                frame['epoch']
            )
        return []
    
    # Create and save animation
    anim = animation.FuncAnimation(
        visualizer.fig, animate,
        frames=len(frames),
        interval=600,
        blit=False,
        repeat=True
    )
    
    # Save video (will use pillow since ffmpeg unavailable)
    anim.save('mlp_visualization.gif', writer='pillow', fps=2)
    plt.close()
    
    return "Video saved as mlp_visualization.gif"

if __name__ == "__main__":
    result = create_visualization()
    print(f"\n{result}")
    print("Visualization complete!")