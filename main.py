# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from flask import Flask
from flask import json

import minio_util
import prediction_2classes
import prediction_5classes
import utils
from disease_segmentation_version5 import predict_segment

app = Flask(__name__)
model_5c, device_5c = prediction_5classes.init_model()
model_2c, device_2c = prediction_2classes.init_model()
model_segment, trainer = predict_segment.init_model()
minio_client = minio_util.init_minio_client()
minio_bucket = 'segment-imgs'


@app.route('/classification5/<string:img_path>', methods=["GET"])
def classification5(img_path):
    if img_path == '':
        return None
    file_name, y_score, prediction_class = prediction_5classes.main(img_path=img_path, device=device_5c, model=model_5c)
    data = json.jsonify({'file_name': file_name, 'y_score': y_score.tolist(), 'prediction_class': prediction_class})
    return data


@app.route('/classification2/<string:img_path>', methods=["GET"])
def classification2(img_path):
    if img_path == '':
        return None
    file_name, y_score, prediction_class = prediction_2classes.main(img_path=img_path, device=device_2c, model=model_2c)
    data = json.jsonify({'file_name': file_name, 'y_score': y_score.tolist(), 'prediction_class': prediction_class})
    return data


@app.route('/segment/<string:img_path>', methods=["GET"])
def segment(img_path):
    if img_path == '':
        return None
    file_path, file_name, orginal_path, all_vessel_num, calc_vessel_num, max_dia, mean_dia = predict_segment.main(
        model_segment, trainer)
    orginal_name = file_name + "_orginal"
    url_oringal_img = minio_util.upload_data(minio_client=minio_client, file_path=orginal_path, file_name=orginal_name,
                                             minio_bucket=minio_bucket)
    url_segment_img = minio_util.upload_data(minio_client=minio_client, file_path=file_path, file_name=file_name,
                                             minio_bucket=minio_bucket)
    utils.delete()
    data = json.jsonify(
        {'segment_url': url_segment_img, 'original_url': url_oringal_img, 'all_vessel_num': all_vessel_num,
         'calc_vessel_num': calc_vessel_num, 'max_dia': max_dia, 'mean_dia': mean_dia})
    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
