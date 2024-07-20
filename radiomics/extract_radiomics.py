from radiomics import featureextractor
import os
import pandas as pd

def extract_radiomics(roots:str,params: str):
    assert os.path.exists(roots), "dataset root:{} does not exist.".format(roots)
    floders = os.listdir(roots)
    path = []
    label_path = []
    files_id = []

    for folder in floders:
        files = os.listdir(os.path.join(roots, folder).replace("\\", "/"))
        files_id.append(folder)
        for file in files:
            if "image.nii" in file:
                file_path = os.path.join(roots, folder, file).replace("\\", "/")
                path.append(file_path)

        for file in files:
            if "mask.nii" in file:
                file_path = os.path.join(roots, folder, file).replace("\\", "/")
                label_path.append(file_path)


    print("path:", len(path))
    print("label_path", len(label_path))
    print("files_id", len(files_id))

    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    featureVector = []

    for i in range(len(path)):
        print(i)
        print("image:", path[i])
        print("label:", label_path[i])
        print("file:", files_id[i])
        result = extractor.execute(path[i], label_path[i], label=255)
        featureVector.append(result)

    names = pd.DataFrame(files_id)
    features = pd.DataFrame(featureVector)
    part_full = pd.concat((names, features), axis=1)
    part_full.to_excel('cl_radiomics.xlsx')


if __name__ == '__main__':
    roots = "./nii/"
    params = './exampleCT_filter.yaml'
    extract_radiomics(roots, params)

