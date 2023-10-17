import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_neural_network(ax, layer_sizes):
    """
    Plot a simple feedforward neural network graph using matplotlib.
    Args:
    - ax: a matplotlib axis.
    - layer_sizes: a list of layer sizes.
    """
    n_layers = len(layer_sizes)
    v_spacing = 0.3  # vertical spacing
    h_spacing = 2.0  # horizontal spacing
    
    # Draw input layer
    input_center = (0, 0)
    ax.text(input_center[0], input_center[1], 'Input\nLayer', ha='center', va='center', fontsize=12)
    for i in range(layer_sizes[0]):
        circle = patches.Circle(input_center, 0.2, fc='skyblue', ec='black', zorder=4)
        ax.add_patch(circle)
        input_center = (input_center[0], input_center[1] - v_spacing)
    
    prev_layer_size = layer_sizes[0]
    prev_layer_center = (0, -0.15*(prev_layer_size-1))
    
    # Draw hidden and output layers
    for i in range(1, n_layers):
        layer_center = (prev_layer_center[0] + h_spacing, -0.15*(layer_sizes[i]-1))
        for j in range(layer_sizes[i]):
            circle = patches.Circle(layer_center, 0.2, fc='skyblue', ec='black', zorder=4)
            ax.add_patch(circle)
            
            # Draw arrows from previous layer
            for k in range(prev_layer_size):
                line = patches.FancyArrowPatch((prev_layer_center[0]+0.2, prev_layer_center[1] - k*v_spacing), 
                                               (layer_center[0]-0.2, layer_center[1] - j*v_spacing),
                                               mutation_scale=10, color='gray', zorder=1)
                ax.add_patch(line)
            
            layer_center = (layer_center[0], layer_center[1] - v_spacing)
        
        if i == n_layers-1:
            ax.text(prev_layer_center[0] + h_spacing/2, prev_layer_center[1] + 2*v_spacing, 'Output\nLayer', ha='center', va='center', fontsize=12)
        else:
            ax.text(prev_layer_center[0] + h_spacing/2, prev_layer_center[1] + 2*v_spacing, 'Hidden\nLayer', ha='center', va='center', fontsize=12)
        
        prev_layer_center = (prev_layer_center[0] + h_spacing, -0.15*(layer_sizes[i]-1))
        prev_layer_size = layer_sizes[i]

# Plot neural network
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
plot_neural_network(ax, [4, 256, 128, 64, 32, 16, 2])
plt.show()