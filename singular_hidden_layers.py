import random
import numpy as np

# Simple feedforward and backprop neural network with single neuron layers
# Architecture: Input -> H1 -> H2 -> Output

# Initialize training data
label = 1
input_val = random.random()

# Initialize weights and biases randomly
iw = random.random()   # input to H1 weight
ib = random.random()   # input to H1 bias

h1w = random.random()  # H1 to H2 weight
h1b = random.random()  # H1 to H2 bias

h2w = random.random()  # H2 to Output weight
h2b = random.random()  # H2 to Output bias

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def feedforward(input_val, iw, ib, h1w, h1b, h2w, h2b):
    """
    Forward pass through the network
    Returns all intermediate values needed for backpropagation
    """
    # Layer 1 (Input -> H1)
    zh1 = input_val * iw + ib  # weighted input to H1
    ah1 = sigmoid(zh1)          # activation of H1
    
    # Layer 2 (H1 -> H2)
    zh2 = ah1 * h1w + h1b      # weighted input to H2
    ah2 = sigmoid(zh2)          # activation of H2
    
    # Output Layer (H2 -> Output)
    zo1 = ah2 * h2w + h2b      # weighted input to Output
    ao1 = sigmoid(zo1)          # activation of Output (prediction)
    
    return ao1, ah2, ah1, zo1, zh2, zh1

def backpropagate(input_val, label, ao1, ah2, ah1, iw, ib, h1w, h1b, h2w, h2b):
    """
    Backward pass to compute gradients
    Returns the changes (deltas) for all weights and biases
    """
    # Cost function (Mean Squared Error)
    cost = (label - ao1) ** 2
    
    # ========== OUTPUT LAYER GRADIENTS ==========
    # Chain rule: d(cost)/d(h2w) = d(cost)/d(ao1) * d(ao1)/d(zo1) * d(zo1)/d(h2w)
    
    dcost_dao1 = -2 * (label - ao1)  # derivative of cost w.r.t. output
    dao1_dzo1 = ao1 * (1 - ao1)      # sigmoid derivative
    dzo1_dh2w = ah2                   # derivative of zo1 w.r.t. h2w
    
    change_h2w = -dcost_dao1 * dao1_dzo1 * dzo1_dh2w
    
    # For bias: d(cost)/d(h2b) - same chain but d(zo1)/d(h2b) = 1
    dzo1_dh2b = 1
    change_h2b = -dcost_dao1 * dao1_dzo1 * dzo1_dh2b
    
    # ========== HIDDEN LAYER 2 GRADIENTS ==========
    # Chain rule: d(cost)/d(h1w) = d(cost)/d(ao1) * d(ao1)/d(zo1) * d(zo1)/d(ah2) * d(ah2)/d(zh2) * d(zh2)/d(h1w)
    
    dzo1_dah2 = h2w                   # derivative of zo1 w.r.t. ah2
    dah2_dzh2 = ah2 * (1 - ah2)      # sigmoid derivative for H2
    dzh2_dh1w = ah1                   # derivative of zh2 w.r.t. h1w
    
    change_h1w = -dcost_dao1 * dao1_dzo1 * dzo1_dah2 * dah2_dzh2 * dzh2_dh1w
    
    # For H2 bias: same chain but d(zh2)/d(h1b) = 1
    dzh2_dh1b = 1
    change_h1b = -dcost_dao1 * dao1_dzo1 * dzo1_dah2 * dah2_dzh2 * dzh2_dh1b
    
    # ========== HIDDEN LAYER 1 GRADIENTS ==========
    # Chain rule: d(cost)/d(iw) = d(cost)/d(ao1) * d(ao1)/d(zo1) * d(zo1)/d(ah2) * d(ah2)/d(zh2) * d(zh2)/d(ah1) * d(ah1)/d(zh1) * d(zh1)/d(iw)
    
    dzh2_dah1 = h1w                   # derivative of zh2 w.r.t. ah1
    dah1_dzh1 = ah1 * (1 - ah1)      # sigmoid derivative for H1
    dzh1_diw = input_val              # derivative of zh1 w.r.t. iw
    
    change_iw = -dcost_dao1 * dao1_dzo1 * dzo1_dah2 * dah2_dzh2 * dzh2_dah1 * dah1_dzh1 * dzh1_diw
    
    # For input bias: same chain but d(zh1)/d(ib) = 1
    dzh1_dib = 1
    change_ib = -dcost_dao1 * dao1_dzo1 * dzo1_dah2 * dah2_dzh2 * dzh2_dah1 * dah1_dzh1 * dzh1_dib
    
    return change_iw, change_ib, change_h1w, change_h1b, change_h2w, change_h2b, cost

def train(input_val, label, iw, ib, h1w, h1b, h2w, h2b, learning_rate=0.1, epochs=1000):
    """
    Train the network for specified number of epochs
    """
    for epoch in range(epochs):
        # Forward pass
        ao1, ah2, ah1, zo1, zh2, zh1 = feedforward(input_val, iw, ib, h1w, h1b, h2w, h2b)
        
        # Backward pass
        change_iw, change_ib, change_h1w, change_h1b, change_h2w, change_h2b, cost = \
            backpropagate(input_val, label, ao1, ah2, ah1, iw, ib, h1w, h1b, h2w, h2b)
        
        # Update weights and biases
        iw += learning_rate * change_iw
        ib += learning_rate * change_ib
        h1w += learning_rate * change_h1w
        h1b += learning_rate * change_h1b
        h2w += learning_rate * change_h2w
        h2b += learning_rate * change_h2b
        
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.6f}, Prediction = {ao1:.6f}")
    
    return iw, ib, h1w, h1b, h2w, h2b

# Train the network
print(f"Initial input: {input_val:.4f}, Target: {label}")
print(f"Initial weights: iw={iw:.4f}, h1w={h1w:.4f}, h2w={h2w:.4f}")
print(f"Initial biases: ib={ib:.4f}, h1b={h1b:.4f}, h2b={h2b:.4f}")

print("\nTraining...\n")

iw, ib, h1w, h1b, h2w, h2b = train(input_val, label, iw, ib, h1w, h1b, h2w, h2b)

# Test final prediction
final_pred, _, _, _, _, _ = feedforward(input_val, iw, ib, h1w, h1b, h2w, h2b)
print(f"\nFinal prediction: {final_pred:.6f}")
print(f"Final weights: iw={iw:.4f}, h1w={h1w:.4f}, h2w={h2w:.4f}")
print(f"Final biases: ib={ib:.4f}, h1b={h1b:.4f}, h2b={h2b:.4f}")
