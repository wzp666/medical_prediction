import os

input_dir = r"D:\Projects\medical_prediction_py_service\disease_segmentation_version5\segment_img_input"

def delete(img_dir=input_dir):
    img_dir = img_dir + "/0"
    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
