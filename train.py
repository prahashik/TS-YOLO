import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class TSYOLOLoss(nn.Module):
    def __init__(self, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super(TSYOLOLoss, self).__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        pred_boxes, pred_obj, pred_cls = predictions
        target_boxes, target_obj, target_cls = targets
        loss_box = self.mse_loss(pred_boxes, target_boxes)
        loss_obj = self.bce_loss(pred_obj, target_obj)
        loss_cls = self.ce_loss(pred_cls, target_cls)
        total_loss = self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls
        return total_loss, {"loss_box": loss_box.item(), "loss_obj": loss_obj.item(), "loss_cls": loss_cls.item()}

def dataloader_pre_step_hook(batch_data):
    images, targets = batch_data
    return images, targets

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        images, targets = dataloader_pre_step_hook(batch)
        images = images.to(device)
        if isinstance(targets, tuple) or isinstance(targets, list):
            targets = tuple(t.to(device) for t in targets)
        else:
            targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training TS-YOLO on {device}")
    
    # Initialize dummy model or TS-YOLO model with Temporal-Spectral modules
    try:
        from models.yolo import TSYOLO
        model = TSYOLO().to(device)
    except ImportError:
        # Fallback dummy model structure
        class DummyTSYOLO(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
                self.box_head = nn.Linear(16, 4)
                self.obj_head = nn.Linear(16, 1)
                self.cls_head = nn.Linear(16, 80)
            def forward(self, x):
                feat = self.backbone(x).mean(dim=[2, 3])
                return self.box_head(feat), self.obj_head(feat), self.cls_head(feat)
        model = DummyTSYOLO().to(device)

    criterion = TSYOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy DataLoader
    dummy_dataset = [(torch.randn(3, 640, 640), (torch.randn(4), torch.tensor([1.0]), torch.tensor(0))) for _ in range(10)]
    dataloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
    
    epochs = 5
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }
        torch.save(checkpoint, f"checkpoints/tsyolo_epoch_{epoch+1}.pt")
        print(f"Saved checkpoint to checkpoints/tsyolo_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
