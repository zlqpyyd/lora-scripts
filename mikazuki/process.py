
import sys
import os
import shutil
from typing import Optional

from mikazuki.tasks import tm
from mikazuki.log import log
import mikazuki.utils as utils 

def pre_run_train(toml_path: str,
              trainer_file: str = "./sd-scripts/train_network.py",
              multi_gpu: bool = False,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", str(cpu_threads),
        trainer_file,
        "--config_file", toml_path,
    ]
    if multi_gpu:
        args.insert(3, "--multi_gpu")

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    try:
        task = tm.create_task(args, customize_env)
        return task
    except Exception as e:
        log.error(f"An error occurred when training / 创建训练进程时出现致命错误: {e}")


from mikazuki.tagger.interrogator import (available_interrogators,
                                          on_interrogate)



def run_inter(j):
    print(j)
    interrogator = available_interrogators.get(j['interrogator_model'], available_interrogators["wd14-convnextv2-v2"])
    on_interrogate(
        image=None,
        batch_input_glob=j['path'],
        batch_input_recursive=False,
        batch_output_dir="",
        batch_output_filename_format="[name].[output_extension]",
        batch_output_action_on_conflict=j['batch_output_action_on_conflict'],
        batch_remove_duplicated_tag=True,
        batch_output_save_json=False,
        interrogator=interrogator,
        threshold=j['threshold'],
        additional_tags=j['additional_tags'],
        exclude_tags=j['exclude_tags'],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=j['replace_underscore'],
        replace_underscore_excludes=j['replace_underscore_excludes'],
        escape_tag=j['escape_tag'],
        unload_model_after_running=True
    )
    return {"status": "success"}


import cv2
from PIL import Image

def autofocus_and_crop(input_path, output_path):
    # 读取图像
    img = cv2.imread(input_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅度
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # 计算梯度幅度的平均值作为对焦度
    focus = int(gradient_magnitude.mean())

    # 根据对焦度裁剪图像
    height, width, _ = img.shape
    min_dim = min(height, width)
    center = (width // 2, height // 2)
    crop_size = 512
    x1 = max(0, center[0] - crop_size//2)
    y1 = max(0, center[1] - crop_size//2)
    x2 = min(width, center[0] + crop_size//2)
    y2 = min(height, center[1] + crop_size//2)
    cropped_img = img[y1:y2, x1:x2]

    # 调整图像大小为 512x512
    resized_img = cv2.resize(cropped_img, (512, 512))

    # 保存处理后的图像
    cv2.imwrite(output_path, resized_img)

    return focus
import cv2
import numpy as np
from PIL import Image

def get_face(path):
    face_cascade = cv2.CascadeClassifier('/root/lora-scripts_dev/mikazuki/haarcascade_frontalface_default.xml')
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测脸部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        faces = faces[np.argsort(faces[:, 1])]
        return faces[0], img
    else:
        print('没有识别到人脸')


# 遍历指定目录下的所有图片文件
def traverse_images(directory):
    image_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".JPEG"):
                image_list.append(os.path.join(root, file))
    return image_list

def resize_image(input_image_path, output_image_path):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    while True:
        if width > 1024 or height > 1024:
            width= int(width/2)
            height=int(height/2)
            size = (width, height)
        else:
            size = (width, height)
            break
    new_image = original_image.resize(size)
    new_image.save(output_image_path)

def copy_512(path):
    import os
    #import glob
    # 获取目录下所有的图片文件路径
    image_files = traverse_images(path)

    # 打印所有找到的图片文件路径
    for image_file in image_files:
        input_image_path=image_file
        output_image_path = image_file
        resize_image(input_image_path, output_image_path)
        image = get_face(image_file)
        if image is not None:
            x, y, w, h = image[0]
            img = image[1]
            img_w = img.shape[0]
            img_h = img.shape[1]
            min_wh = 512#int(min(min(x * 2 + w, y * 2 + h + 6), min(img_w, img_h)))
            save_x = max(x - int((min_wh - w) / 2), 0)
            save_y = max(y - int((min_wh - h) / 2) - 6, 0)
            save_img = np.copy(img)[save_y: save_y + min_wh, save_x: save_x + min_wh]
            cv2.imwrite(image_file, save_img)
        else:
            autofocus_and_crop(input_image_path, output_image_path)
        #cv2.waitKey(0)
        # print("Focus measure: ", focus_measure)
        
def copy_mirrored_img(directory):
    # 遍历目录下所有图片文件
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".JPG") or filename.endswith(".PNG") or filename.endswith(".JPEG"):
            img_path = os.path.join(directory, filename)
    
            # 打开图像文件
            img = Image.open(img_path)
            # 将图像转换为RGB模式
            img = img.convert('RGB')
            # 镜像复制图像
            mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
            # 保存镜像复制后的图像
            mirrored_img.save(os.path.join(directory, 'mirrored_' + filename))
        

def run_train(task_id: str, lora_name: str, train_dir: str, doppelganger_id, sd_service_url: str, notify_url: str, j):
    # 裁剪512*512
    copy_512(j['path'])
    copy_mirrored_img(j['path'])
    run_inter(j)
    task = tm.get_task(task_id)
    tm.execute_task(task_id)
    try:
        tm.wait_for_process(task_id)
        result = task.communicate()
        if result.returncode != 0:
            log.error(f"Training failed / 训练失败")
        else:
            log.info(f"Training finished / 训练完成")
            log.info(f"Uploading start / 开始上传模型")
            model_url = utils.upload_to_oss('./output/'+lora_name+'.safetensors')
            log.info(f"Uploading finished / 上传模型完成")
            log.info(f"rm temp file / 开始删除本地文件")
            os.remove('./output/'+lora_name+'.safetensors')
            shutil.rmtree(train_dir)
            log.info(f"rm temp file / 删除本地文件完成")
            log.info(f"notify start / 通知应用层 {model_url}")
            utils.notify_finish(notify_url=notify_url, doppelganger_id=doppelganger_id, sd_service_url=sd_service_url, model_url=model_url, notify_status="success")
            log.info(f"notify finished / 通知应用层完成")
    except Exception as e:
        log.error(f"An error occurred when training / 创建训练进程时出现致命错误: {e}")
        utils.notify_finish(doppelganger_id=doppelganger_id, sd_service_url=sd_service_url, model_url=None, notify_status="fail")
