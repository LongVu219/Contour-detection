from utils import *

DATA_DIR = '/home/longvv/Contour-detection/yolo_dataset'
TRAIN_DIR = '/home/longvv/Contour-detection/yolo_dataset/train'
S = 13
BOX = 5
CLS = 1
H, W = 416, 416
OUTPUT_THRESH = 0.7

def get_label_from_textline(text):
    coor = text.split(" ")
    return float(coor[0])

scale = 770

def get_xywh_from_textline(text):
    coor = text.split(" ")
    coor = [float(x) for x in coor]
    
    if len(coor) > 4:    
        
        for i in range (1, 5):
            coor[i] = coor[i] * scale
        return [coor[1] - coor[3]/2, coor[2] - coor[4]/2, coor[3], coor[4]] 
    else:
        return None

def read_data(folder_path, max_contour = 10):
    data = []

    images_folder = os.path.join(folder_path, 'images')
    labels_folder = os.path.join(folder_path, 'labels')

    for filename in os.listdir(labels_folder):

        filepath = os.path.join(labels_folder, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for i, cur_line in enumerate(lines):
                img_data = {
                    'file_path': os.path.join(images_folder, filename.replace('.txt', '.png')),
                    'box_nb': len(lines),
                    'boxes': [],
                    'labels': []
                }

                box_nb = img_data['box_nb']

                #print(box_nb)

                for j in range(box_nb):
                    rect = get_xywh_from_textline(lines[j].replace("\n", ""))
                    labels = get_label_from_textline(lines[j].replace("\n", ""))
                    if rect is not None:
                        img_data['boxes'].append(rect)
                        img_data['labels'].append(labels)
                if len(img_data['boxes']) > 0:
                    data.append(img_data)
                
    return data

class ContourDataset(Dataset):
    def __init__(self, data, img_dir=TRAIN_DIR, transforms=None):
        self.data = data
        self.img_dir = img_dir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        img_data = self.data[id]
        img_fn = f"{img_data['file_path']}"
        boxes = img_data["boxes"]
        box_nb = img_data["box_nb"]

        assert len(boxes) > 0

        labels = torch.tensor(img_data["labels"])
        img = cv2.imread(img_fn).astype(np.float32)/255.0
        
        try:
            if self.transforms:
                sample = self.transforms(**{
                    "image":img,
                    "bboxes": boxes,
                    "labels": labels,
                })
                img = sample['image']
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        except:
            return self.__getitem__(randint(0, len(self.data)-1))

        #print(img_data['file_path'], boxes, box_nb)
        target_tensor = self.boxes_to_tensor(boxes.type(torch.float32), labels)
        return img, target_tensor
    
    def boxes_to_tensor(self, boxes, labels):
        """
        Convert list of boxes (and labels) to tensor format
        Output:
            boxes_tensor: shape = (Batchsize, S, S, Box_nb, (4+1+CLS))
        What a day to be alive
        """
        boxes_tensor = torch.zeros((S, S, BOX, 5+CLS))
        cell_w, cell_h = W/S, H/S
        for i, box in enumerate(boxes):
            x,y,w,h = box
            # normalize xywh with cell_size
            x,y,w,h = x/cell_w, y/cell_h, w/cell_w, h/cell_h
            center_x, center_y = x+w/2, y+h/2
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            if grid_x < S and grid_y < S:
                boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(BOX * [[center_x-grid_x,center_y-grid_y,w,h]])
                boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(BOX * [1.])
                boxes_tensor[grid_y, grid_x, :, 5] = labels[i]
        return boxes_tensor

def collate_fn(batch):
    return tuple(zip(*batch))

def target_tensor_to_boxes(boxes_tensor):
    '''
    Recover target tensor (tensor output of dataset) to bboxes
    Input:
        boxes_tensor: bboxes in tensor format - output of dataset.__getitem__
    Output:
        boxes: list of box, each box is [x,y,w,h]
    '''
    cell_w, cell_h = W/S, H/S
    boxes = []
    for i in range(S):
        for j in range(S):
            for b in range(BOX):
                data = boxes_tensor[i,j,b]
                x_center,y_center, w, h, obj_prob, cls_prob = data[0], data[1], data[2], data[3], data[4], data[5]
                prob = obj_prob*cls_prob
                if prob > OUTPUT_THRESH:
                    x, y = x_center+j-w/2, y_center+i-h/2
                    x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                    box = [x,y,w,h]
                    boxes.append(box)
    return boxes

from albumentations.pytorch.transforms import ToTensor, ToTensorV2
import albumentations as A
train_transforms = A.Compose([
        A.Resize(height=416, width=416),
        A.RandomSizedCrop(min_max_height=(350, 416), height=416, width=416, p=0.4),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ],
    bbox_params={
        "format":"coco",
        'label_fields': ['labels']
})