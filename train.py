import torch
import torch.nn as nn

# --- 1. Define the Final Model Architecture ---
# This is the "brain" that does the personality prediction
class PersonalityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PersonalityModel, self).__init__()
        # input_size is 1536 (768 from text + 768 from audio)
        self.layer1 = nn.Linear(input_size, 128)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, combined_features):
        x = self.layer1(combined_features)
        x = self.activation(x)
        output = self.output_layer(x)
        return output

# --- Main execution ---
if __name__ == "__main__":
    
    # --- 2. Load Preprocessed Data ---
    print("Loading preprocessed data...")
    try:
        features = torch.load("train_features.pt")
        labels = torch.load("train_labels.pt")
        print("Loaded features and labels.")
        print(f"Feature shape: {features.shape}")
        print(f"Label shape: {labels.shape}")
    except FileNotFoundError:
        print("Error: 'train_features.pt' or 'train_labels.pt' not found.")
        print("Please 'git pull' or ask your friend to generate and push them.")
        exit()
    
    # --- 3. Setup Model, Loss, and Optimizer ---
    INPUT_SIZE = features.shape[1]  # Get size from data (should be 1536)
    OUTPUT_SIZE = labels.shape[1] # Get size from data (should be 5)
    
    model = PersonalityModel(INPUT_SIZE, OUTPUT_SIZE)
    
    # Mean Squared Error is used for regression (predicting a score)
    loss_function = nn.MSELoss()
    
    # Adam is a standard, effective optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- 4. Training Loop ---
    print("Starting training...")
    num_epochs = 50000  # Number of times to loop over the data
    
    for epoch in range(num_epochs):
        # Forward pass: Get model's predictions
        predictions = model(features)
        
        # Calculate loss: How wrong were the predictions?
        loss = loss_function(predictions, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Calculate new gradients
        optimizer.step()       # Update model weights
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print("Training complete!")

    # --- 5. Save the Final Trained Model ---
    torch.save(model.state_dict(), "personality_model_final.pth")
    print("Saved trained model to personality_model_final.pth")