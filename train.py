
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from model import SteeringAnglePredictor, TwinLiteNet
from data_processing import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4
num_epochs 
learning_rate 
k_folds = 10

root_dir = "/home/wyl123/minidata/driving_dataset/"
txt_file = "/home/wyl123/minidata/driving_dataset/data.txt"

transform = transforms.Compose([
    transforms.Resize((320, 180)),
    transforms.ToTensor(),
])

custom_dataset = CustomDataset(root_dir, txt_file, num_samples=10000, transform=transform)

best_val_loss = float('inf')
best_fold = -1
best_model_state = None

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(custom_dataset)):
    print(f'Fold {fold+1}')
    train_subset = torch.utils.data.Subset(custom_dataset, train_idx)
    val_subset = torch.utils.data.Subset(custom_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    seg_model = TwinLiteNet()
    seg_model = torch.nn.DataParallel(seg_model)
    seg_model.load_state_dict(torch.load('pretrained/best.pth'))
    seg_model = seg_model.cuda()
    seg_model.eval()
    
    steering_model = SteeringAnglePredictor(input_dim=180).to(device)
    criterion = RMSELoss()
    optimizer = optim.Adam(steering_model.parameters(), lr=learning_rate)
    
    prev_masks = None  
    prev_masks = None
    batch_counter = 0
    all_predictions = []
    val_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        all_outputs = []

        # Train the model
        for images, angles in train_loader:
            images = images.float().to(device)
            angles = angles.float().to(device)

            with torch.no_grad():
                seg_output1, _ = seg_model(images)
            masks = (seg_output1 > 0.5).float()

            masks = masks.repeat(1, images.size(1), 1, 1)
            masks = masks[:, :images.size(1), :, :]
            masks = F.interpolate(masks, size=(320, 180), mode='nearest')

            masked_images = images * masks

            outputs = steering_model(masked_images)

            loss = criterion(outputs, angles.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = outputs.squeeze().cpu().detach().numpy()
            all_predictions.append(predictions)

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')

        # Validate the model
        val_loss = 0.0
        with torch.no_grad():
            for images, angles in val_loader:
                images = images.float().to(device)
                angles = angles.float().to(device)

                seg_output1, _ = seg_model(images)
                masks = (seg_output1 > 0.5).float()
                masks = masks.repeat(1, images.size(1), 1, 1)
                masks = masks[:, :images.size(1), :, :]
                masks = F.interpolate(masks, size=(320, 180), mode='nearest')
                masked_images = images * masks

                outputs = steering_model(masked_images)

                loss = criterion(outputs, angles.unsqueeze(1))
                val_losses.append(loss.item())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation RMSE: {val_loss:.4f}')

        # Update best model if necessary
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_fold = fold
            best_model_state = steering_model.state_dict()

# Save the best model state
if best_model_state is not None:
    torch.save
