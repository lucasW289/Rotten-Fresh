import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torchinfo import summary
import data_loader
import model as model_factory

# --- CHANGE 1: Hardcoded Config (No need to upload YAML files) ---
CFG = {
    "experiment_name": "/Shared/Fruit_Freshness_v1", # Changed to a safe shared path
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "/dbfs/FileStore/tables/dataset",     # <--- ASSUMING YOU UPLOADED DATA HERE
    "img_height": 224,
    "img_width": 224,
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.001
}

def main():
    print(f"🚀 Running on Databricks using {CFG['device']}")

    # --- CHANGE 2: Simplified MLflow Setup ---
    # In Databricks, you don't need set_tracking_uri or tokens. It's automatic.
    mlflow.set_experiment(CFG["experiment_name"])
    
    # --- Setup Device ---
    device = torch.device(CFG["device"])

    # --- Load Data ---
    # NOTE: Ensure you uploaded your dataset to DBFS or use a sample for testing
    # If you haven't uploaded data yet, this part will fail. 
    # For now, let's wrap this in a try/except or assume data is there.
    try:
        train_loader, val_loader = data_loader.get_dataloaders(
            CFG["data_dir"], CFG["batch_size"], CFG["img_height"], CFG["img_width"]
        )
    except Exception as e:
        print(f"⚠️ Data Load Error: {e}")
        print("Did you upload the dataset to DBFS?")
        return

    # --- Build Model ---
    model = model_factory.build_model().to(device)
    
    # --- Train ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG["learning_rate"])

    print("🚀 Starting Training...")
    with mlflow.start_run():
        mlflow.log_params(CFG)

        for epoch in range(CFG["epochs"]):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Simple log per epoch
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")
            mlflow.log_metric("loss", epoch_loss, step=epoch)

        # --- CHANGE 3: Save to DBFS (Permanent Storage) ---
        # /dbfs/FileStore is a special path that maps to blob storage
        save_dir = "/dbfs/FileStore/models"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, "fruit_model.pth")
        torch.save(model.state_dict(), save_path)
        
        print(f"💾 Model saved permanently to {save_path}")
        mlflow.log_artifact(save_path)

if __name__ == "__main__":
    main()