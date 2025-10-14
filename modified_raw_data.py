import pandas as pd
from pathlib import Path

def modified_boundary(original_file_path):
    normal_cell = ['NILM-Inflammatory cells', 'NILM-Parabasal', 'NILM-glands', 'NILM-intermediate& Superficial']
    lesion_cell = ['ASC-H', 'ASCUS', 'HSIL', 'LSIL']

    df = pd.read_csv(original_file_path)

    two_class = []
    five_class = []

    for _, row in df.iterrows():

        five_role = row.copy()
        if five_role['type'] in normal_cell:
            five_role['type'] = 'NILM'
        five_class.append(five_role)

        two_role = row.copy()
        if two_role['type'] in normal_cell:
            two_role['type'] = 'NILM'
        elif two_role['type'] in lesion_cell:
            two_role['type'] = 'Lesion'
        two_class.append(two_role)

    df_five_class = pd.DataFrame(five_class)
    df_two_class = pd.DataFrame(two_class)

    out_dir = original_file_path.parent.parent
    out_name = original_file_path.name
    df_five_class.to_csv(out_dir / "boundaries(csv)_5_class" / out_name)
    df_two_class.to_csv(out_dir / "boundaries(csv)_2_class" / out_name)


original_folder_path = Path('/project/aip-xli135/jeff418/YOLO/raw_data/boundaries(csv)_7_class')
for file in original_folder_path.iterdir():
    modified_boundary(file)
