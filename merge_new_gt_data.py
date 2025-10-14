from pathlib import Path
import sys

def check_file_existent(folder_path, file_list):
    checked_folder = Path(folder_path)
    not_exist = []
    missing = 0
    
    for file in file_list:
        file_path = checked_folder / file  
        if not file_path.exists():         
            not_exist.append(file)
            missing = 1
            
        #print(file_path)
    return not_exist, missing

def gather_all_txt_file(folder_path):
    input_folder = Path(folder_path)
    data_files = list(input_folder.glob("*.txt"))
    
    str_data_files = [f.name for f in data_files]

    #print(str_data_files)
    return str_data_files

def filter_incorrect_files(str_data_files, incorrect_case):
    filtered_name = [f for f in str_data_files if f not in incorrect_case]
    return filtered_name

def filter_name_for_merge(str_data_file):
    parts = str_data_file.split("_")
    new_name = "_".join(parts[:3]) + ".txt"
    #print(new_name)
    return new_name # single str for the name

def merge_data(added_gt_data_folder, old_gt_data_folder, file_name):
    new_data_folder = Path(added_gt_data_folder)
    new_data_file_path = new_data_folder / file_name

    old_data_folder = Path(old_gt_data_folder)
    old_data_file_name = filter_name_for_merge(file_name)
    old_data_file_train = old_data_folder / 'train' / old_data_file_name
    old_data_file_val = old_data_folder / 'val' / old_data_file_name
    
    if old_data_file_train.exists():
        old_data_file_path = old_data_file_train
    else:
        return 
        #DO NOT CHANGE VAL FILE
        #old_data_file_path = old_data_file_val

    with open(new_data_file_path, "r") as f_new:
        new_lines = []
        for line in f_new:
            parts = line.strip().split()
            if len(parts) >= 5:
                new_lines.append(" ".join(parts[:5])) 
        new_content = "\n".join(new_lines)

    with open(old_data_file_path, "a") as f_old:
        f_old.write("\n" + new_content)

def merge_incorrect_data(added_gt_data_folder, old_gt_data_folder, file_name):
    new_data_folder = Path(added_gt_data_folder)
    new_data_file_path = new_data_folder / file_name

    old_data_folder = Path(old_gt_data_folder)
    old_data_file_name = filter_name_for_merge(file_name)
    old_data_file_train = old_data_folder / 'train' / old_data_file_name
    old_data_file_val = old_data_folder / 'val' / old_data_file_name
    
    if old_data_file_train.exists():
        old_data_file_path = old_data_file_train
    else:
        return 
        #DO NOT CHANGE VAL FILE
        #old_data_file_path = old_data_file_val

    with open(new_data_file_path, "r") as f_new:
        new_lines = []
        for line in f_new:
            parts = line.strip().split()
            if len(parts) >= 5:
                new_lines.append(" ".join(["1"] + parts[1:5]))
        new_content = "\n".join(new_lines)

    with open(old_data_file_path, "a") as f_old:
        f_old.write("\n" + new_content) 

def full_merge_process():
    missing_files, incorrect_case_error = check_file_existent(added_gt_data_folder, incorrect_case)

    if incorrect_case_error:
        print('check incorrect case input: \n')
        print(missing_files)
        sys.exit()

    new_data_files_name = gather_all_txt_file(added_gt_data_folder)
    filtered_new_data_files_name = filter_incorrect_files(new_data_files_name, incorrect_case)
    for f in filtered_new_data_files_name:
        merge_data(added_gt_data_folder, old_gt_data_folder, f)
    for l in incorrect_case:
        merge_incorrect_data(added_gt_data_folder, old_gt_data_folder, l)


#=====================================

def class_filter(file_path):
    kept_lines = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "0" and len(parts) >= 5:
            #if len(parts) >= 5:
                kept_lines.append(" ".join(parts[:5]))
    return kept_lines

def update_file(file_path, kept_lines):
    with open(file_path, "w") as f:
        f.write("\n".join(kept_lines))

def delete_data_instance(images_folder, file_path):
    image_name = file_path.stem
    image_path = images_folder / f'{image_name}.jpg'
    file_path.unlink()
    image_path.unlink()

def lesion_filter_process():
    target_data_folder = Path(lesion_filter_folder)
    images_train_folder = target_data_folder / 'images' / 'train'
    images_val_folder = target_data_folder / 'images' / 'val'
    labels_train_folder = target_data_folder / 'labels' / 'train'
    labels_val_folder = target_data_folder / 'labels' / 'val'

    for f in labels_train_folder.iterdir():
        file_name = f.name
        file_path = labels_train_folder / file_name
        kept_lines = class_filter(file_path)
        if kept_lines:
            update_file(file_path, kept_lines)
        else:
            delete_data_instance(images_train_folder, file_path)
    
    for f in labels_val_folder.iterdir():
        file_name = f.name
        file_path = labels_val_folder / file_name
        kept_lines = class_filter(file_path)
        if kept_lines:
            update_file(file_path, kept_lines)
        else:
            delete_data_instance(images_val_folder, file_path)

 #===================================== count labels ===================================== 
target_folders_to_count = [
    "/project/aip-xli135/jeff418/YOLO/yolo_data_downscale_2_class",
    "/project/aip-xli135/jeff418/YOLO/yolo_data_downscale_2_class_sept_8",
    "/project/aip-xli135/jeff418/YOLO/yolo_data_downscale_2_class_sept_11"
]
txt_output_path = Path("/project/aip-xli135/jeff418/YOLO/labels_count.txt")


def count_labels(target_folder_to_count: Path):
    labels_folder = target_folder_to_count / "labels"
    train_folder = labels_folder / "train"
    val_folder = labels_folder / "val"

    train_normal_count = 0
    train_lesion_count = 0
    val_normal_count = 0
    val_lesion_count = 0

    for f in train_folder.glob("*.txt"):
        with open(f, 'r') as t:
            for line in t:
                parts = line.strip().split()
                if parts and parts[0] == "0":
                    train_lesion_count += 1
                elif parts:
                    train_normal_count += 1

    for f in val_folder.glob("*.txt"):
        with open(f, 'r') as t:
            for line in t:
                parts = line.strip().split()
                if parts and parts[0] == "0":
                    val_lesion_count += 1
                elif parts:
                    val_normal_count += 1

    total_normal = train_normal_count + val_normal_count
    total_lesion = train_lesion_count + val_lesion_count
    total_labels = total_lesion + total_normal

    with open(txt_output_path, "a") as out:
        out.write("# ------------------------------------------------------------\n")
        out.write(f"Dataset: {target_folder_to_count.name}\n")
        out.write(f"Train - Lesion: {train_lesion_count}, Normal: {train_normal_count}\n")
        out.write(f"Val   - Lesion: {val_lesion_count}, Normal: {val_normal_count}\n")
        out.write(f"Total - Lesion: {total_lesion}, Normal: {total_normal}, All: {total_labels}\n")



def count_labels_all_folder():
    txt_output_path.write_text("")
    for target_folder in target_folders_to_count:
        count_labels(Path(target_folder))


def main():
    #full_merge_process()
    #lesion_filter_process()
    count_labels_all_folder()



added_gt_data_folder = '/project/aip-xli135/jeff418/YOLO/lesion_false_positives_sept_11'

#replicate the previous data folder, change the name and put its path below
#and since the following folder have the exactly content as the original folder, we only need to apped new data to the following folder
old_gt_data_folder = Path('/project/aip-xli135/jeff418/YOLO/yolo_data_downscale_2_class_sept_11/labels')


lesion_filter_folder = '/project/aip-xli135/jeff418/YOLO/yolo_data_downscale_2_class_sept_9_only_lesion'


incorrect_case = [
    # s075 patches
    "s075_patch_00665_0_12_38_46.txt",
    "s075_patch_00740_114_153_145_182.txt",
    "s075_patch_00750_214_172_241_192.txt",
    "s075_patch_00782_28_220_73_256.txt",
    "s075_patch_00783_220_0_256_54.txt",
    "s075_patch_00784_0_10_14_57.txt",
    "s075_patch_00824_146_172_177_208.txt",
    "s075_patch_00903_2_172_33_203.txt",
    "s075_patch_00995_114_230_153_251.txt",
    "s075_patch_01186_27_153_49_172.txt",
    "s075_patch_01194_34_91_58_122.txt",
    "s075_patch_01224_112_0_156_36.txt",
    "s075_patch_01227_126_178_187_223.txt",
    "s075_patch_01267_53_88_85_133.txt",
    "s075_patch_01310_223_204_253_242.txt",
    "s075_patch_01311_35_125_65_156.txt",
    "s075_patch_01311_38_187_105_219.txt",
    "s075_patch_01353_104_0_143_27.txt",

    # s099 patches
    "s099_patch_00069_221_153_240_177.txt",
    "s099_patch_00101_174_92_188_104.txt",
    "s099_patch_00102_139_85_213_146.txt",
    "s099_patch_00113_110_198_139_221.txt",
    "s099_patch_00433_207_0_237_29.txt",
    "s099_patch_00448_138_185_160_215.txt",
    "s099_patch_00472_0_53_28_103.txt",
    "s099_patch_00513_133_0_171_15.txt",
    "s099_patch_00515_113_203_137_231.txt",
    "s099_patch_00516_209_44_248_94.txt",
    "s099_patch_00529_38_77_99_122.txt",
    "s099_patch_00668_74_67_100_109.txt",
    "s099_patch_00748_71_222_106_239.txt",
    "s099_patch_00842_86_219_118_253.txt",
    "s099_patch_00843_44_93_92_138.txt",
    "s099_patch_00860_153_144_176_159.txt",
    "s099_patch_00868_39_183_57_198.txt",
    "s099_patch_00868_121_199_145_216.txt",
    "s099_patch_00870_169_229_198_249.txt",
    "s099_patch_00915_3_54_58_91.txt",
    "s099_patch_00925_2_51_32_82.txt",
    "s099_patch_01005_10_47_34_74.txt",
    "s099_patch_01046_36_175_57_205.txt",
    "s099_patch_01047_97_94_118_112.txt",
    "s099_patch_01051_26_213_48_250.txt",
    "s099_patch_01051_107_217_123_237.txt",
    "s099_patch_01051_200_123_224_151.txt",
    "s099_patch_01105_97_129_124_175.txt",
    "s099_patch_01176_189_120_231_153.txt",
    "s099_patch_01217_106_220_148_256.txt",
    "s099_patch_01346_88_39_114_56.txt",
    "s099_patch_01383_121_231_149_248.txt",
    "s099_patch_01426_172_154_189_175.txt",
    "s099_patch_01426_176_24_197_55.txt",
    "s099_patch_01427_23_238_43_256.txt",

    # s156 patch
    "s156_patch_00402_0_59_42_102.txt",

    # s450 patch
    "s450_patch_00232_100_129_139_156.txt",

    # s470 patches
    "s470_patch_00024_3_166_54_221.txt",
    "s470_patch_00219_1_94_48_120.txt",
    "s470_patch_00257_222_226_249_253.txt",
    "s470_patch_00258_86_238_132_256.txt",
    "s470_patch_00258_106_0_148_28.txt",
    "s470_patch_00276_143_119_172_154.txt",
    "s470_patch_00297_36_91_74_118.txt",
    "s470_patch_00318_152_75_184_98.txt",
    "s470_patch_00320_9_139_30_158.txt",
    "s470_patch_00336_157_149_195_177.txt",
    "s470_patch_00359_97_118_127_155.txt",
    "s470_patch_00360_51_98_109_146.txt",
    "s470_patch_00422_17_20_33_38.txt",
    "s470_patch_00563_123_15_154_53.txt",
    "s470_patch_00595_102_0_141_17.txt",
    "s470_patch_00801_192_37_233_72.txt",
    "s470_patch_00812_93_7_131_52.txt",
    "s470_patch_00834_4_52_40_83.txt",
    "s470_patch_00837_0_238_33_256.txt",
    "s470_patch_00912_79_55_135_96.txt",

    # s501 patches
    "s501_patch_00908_0_181_66_216.txt",
    "s501_patch_01069_147_0_184_18.txt",
    "s501_patch_01071_22_166_42_196.txt",
    "s501_patch_01264_134_59_157_89.txt",
    "s501_patch_01278_20_108_46_142.txt",
    "s501_patch_01307_60_0_106_11.txt"
]

#for sept_8:
# incorrect_case = [
#     "s099_patch_00915_4_55_58_92.txt",
#     "s470_patch_00801_188_20_231_72.txt",
#     "s501_patch_00908_50_42_91_74.txt",
#     "s501_patch_01069_146_0_187_27.txt",
#     "s501_patch_01113_117_12_171_46.txt",
#     "s511_patch_00108_58_0_147_54.txt",
#     "s511_patch_00109_41_27_90_67.txt",
#     "s511_patch_00875_174_101_203_137.txt",
#     "s511_patch_01493_188_156_217_199.txt",
#     "s598_patch_00137_65_181_113_226.txt"
# ]

if __name__ == "__main__":
    main()


