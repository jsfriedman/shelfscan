import json
import os

def find_bounding_boxes(annotation_file, annotation_dir):
    # Load the annotation file
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Map image IDs to filenames
    id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    # Set to store unique image ids with bounding box annotations
    bounding_box_image_ids = set()

    # Loop over all annotations
    for annotation in data['annotations']:
        # If the annotation has the 'bbox' field and 'segmentation' field is not empty, it's a polygon
        if 'bbox' in annotation and 'segmentation' in annotation and annotation['segmentation']:
            points = annotation['segmentation'][0]
            if len(points) != 8:  # More than 4 points with x,y for each point, it's a polygon
                continue

        # If the annotation has the 'bbox' field but no 'segmentation' field or 'segmentation' field is empty, it's a bounding box
        elif 'bbox' in annotation and ('segmentation' not in annotation or not annotation['segmentation']):
            bounding_box_image_ids.add(annotation['image_id'])

    # Get unique filenames for the image ids
    bounding_box_image_filenames = [id_to_filename[image_id] for image_id in bounding_box_image_ids]
    # for file_name in bounding_box_image_filenames:
    #     print(f'removing {file_name}')
    #     os.remove(annotation_dir+f'\{file_name}')

    return bounding_box_image_filenames



if __name__ == "__main__":
    annotation_dir = r'D:\Downloads\Book spine instance segmentation.v1i.coco\train'
    annotation_file = r'D:\Downloads\Book spine instance segmentation.v1i.coco\train\_annotations.coco.json'
    bounding_box_image_ids = find_bounding_boxes(annotation_file, annotation_dir)

    print("Found bounding boxes in the following images:", bounding_box_image_ids)