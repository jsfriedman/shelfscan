import os
import glob

def run():
    # iterate through each .txt file in dir_to_search's label dir
    labels_dir = r'D:\Code\shelfscan\datasets\test\labels'
    images_dir = r'D:\Code\shelfscan\datasets\test\images'
    annotation_file_paths_to_check = glob.glob(labels_dir+"\*")
    # load 1 line of txt file
    for annotation_file_path in annotation_file_paths_to_check:
        file = open(annotation_file_path, 'r')
        file_lines = file.readlines()
        for line in file_lines:
            # if line length == 5, then get the filename
            if len(line.split(" ")) <= 6:
                print("suspicious line")
                # correlator = os.path.basename(annotation_file_path)[:-4]
                # matching_image_files = glob.glob(images_dir+'\*'+correlator+'*')
                # file.close()
                # os.remove(annotation_file_path)
                # print(f"removing {annotation_file_path}")
                # for matching_image_file in matching_image_files:
                #     os.remove(matching_image_file)
                #     print(f"removing {matching_image_file}")

if __name__ == '__main__':
    print('running code')
    run()