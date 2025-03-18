from utils import *
from dataset import *
from model import *
from loss import *
import tqdm
import wandb

def get_dataloader(data_path):
    data = read_data(data_path)
    dataset = ContourDataset(data, transforms=train_transforms)
    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn, drop_last=False)

train_path = DATA_DIR + '/train'
train_loader = get_dataloader(train_path)

val_path = DATA_DIR + '/val'
val_loader = get_dataloader(val_path)

test_path = DATA_DIR + '/test'
test_loader = get_dataloader(test_path)

S = 13
BOX = 5
CLS = 1
H, W = 416, 416
OUTPUT_THRESH = 0.7

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO(S=S, BOX=BOX, CLS=CLS)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
epochs = 45
imgs, targets = next(iter(train_loader))

wandb.init(project="train_yolo_scratch", name="test1")
best_loss = 1e18


for epoch in range(epochs):
    model.train()
    print(f"---------------------Epoch {epoch} training---------------------")
    iters = 0
    for imgs, targets in train_loader:

        tensor_imgs, tensor_targets = torch.stack(imgs), torch.stack(targets)
        output = model(tensor_imgs.to(device))
        loss, aux = custom_loss(output_tensor=output, target_tensor=tensor_targets.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1

        if (iters % 10 == 0):
            print(f"Epoch {epoch}, Iteration {iters}, Total Loss: {loss.item()}, MSE : {aux[2].item()}")
        
        #wandb.log({"epoch": epoch, "loss": loss.item(), "area_loss": aux[0], "mse_loss": aux[2]})
    
    total_loss = 0
    area_loss = 0
    obj_loss = 0
    mse_loss = 0
    model.eval()
    with torch.no_grad():
        for imgs, targets in val_loader:
            tensor_imgs, tensor_targets = torch.stack(imgs), torch.stack(targets)
            output = model(tensor_imgs.to(device))
            loss, aux = custom_loss(output_tensor=output, target_tensor=tensor_targets.to(device))
            total_loss += loss.item()
            area_loss += aux[0].item()
            obj_loss += aux[1].item()
            mse_loss += aux[2].item()
    
    total_loss /= len(val_loader)
    area_loss /= len(val_loader)    
    obj_loss /= len(val_loader)
    mse_loss /= len(val_loader)
    print(f"Epoch {epoch}, Validation Loss: {total_loss}, Area_loss: {area_loss}, Obj_loss: {obj_loss} MSE: {mse_loss}")
    print()

    wandb.log({"epoch": epoch, "loss": total_loss, "area_loss": area_loss, "obj_loss": obj_loss, "mse_loss": mse_loss})
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), 'best_model_scratch.pth')
        print('Model saved')


