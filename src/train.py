import yaml
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torchinfo import summary
import data_loader
import model as model_factory

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"⚙️  Configuration Loaded: {cfg['experiment_name']}")

    # 2. Setup System
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000"))
    mlflow.set_experiment(cfg["experiment_name"])
    device = torch.device(cfg["device"])

    # 3. Load Data & Model
    train_loader, val_loader = data_loader.get_dataloaders(
        cfg["data_dir"], cfg["batch_size"], cfg["img_height"], cfg["img_width"]
    )
    
    model = model_factory.build_model().to(device)

    # 4. [REQUIREMENT] Model Summary
    print("\n" + "="*40)
    print("📊 MODEL SUMMARY")
    # Simulate a dummy input to check shapes
    summary(model, input_size=(1, 3, cfg["img_height"], cfg["img_width"]))
    print("="*40 + "\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # 5. Training Loop
    print("🚀 Starting Training...")
    with mlflow.start_run():
        mlflow.log_params(cfg)

        for epoch in range(cfg["epochs"]):
            # Train
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
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Logs
            epoch_loss = running_loss / len(train_loader)
            acc = 100 * correct / total
            print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.2f}%")
            
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)

        # 6. Save Model
        os.makedirs("models", exist_ok=True)
        save_path = "models/fruit_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model saved to {save_path}")
        mlflow.log_artifact(save_path)

if __name__ == "__main__":
    main()