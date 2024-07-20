import os
import json
import pandas as pd
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
from resize_lesions import generate_train_test
from resnet.model import resnet50
from my_dataset import MyDataSet


if __name__ == '__main__':
    print(torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(128),
                                         transforms.CenterCrop(114),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    roots = "../nii/"
    assert os.path.exists(roots), "file: '{}' dose not exist.".format(roots)
    images,files_id = generate_train_test(roots)

    validate_dataset = MyDataSet(images =images,
                                 transform=data_transform)

    batch_size = 16
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    # create model
    model = resnet50(num_classes=1000).to(device)

    # load model weights
    weights_path = "./resnet50-pre.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    # in_channel = model.fc.in_features
    # model.fc = model.Linear(in_channel, 2)
    # model.to(device)
    # params = [p for p in model.parameters() if p.requires_grad]


    # prediction
    model.eval()
    probs = []
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images = val_data
            outputs = model(val_images.to(device))
            outputs = outputs.to("cpu").numpy()

            # Flatten the output features
            outputs = outputs.reshape(outputs.shape[0], -1)

            probs.extend(outputs)


    # Convert to DataFrame
    probs = pd.DataFrame(probs)
    # Concatenate along the columns
    result = pd.concat([files_id, probs], axis=1)

    # Save to Excel
    result.to_excel('transfer_learning.xlsx', index=False)



    # with torch.no_grad():
    #     for val_data in tqdm(validate_loader):
    #         val_images = val_data
    #         outputs = model(val_images.to(device))
    #         probs.append(outputs.to("cpu").numpy())
    # probs = pd.DataFrame(probs)
    # pd.concat([files_id,probs],axis=1).to_excel('train_prob.xlsx')


