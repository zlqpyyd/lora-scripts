import glob
import importlib.util
import os
import subprocess
import sys
import re
import shutil
import tkinter
from tkinter.filedialog import askopenfilename, askdirectory
from typing import Optional
from mikazuki.log import log
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import requests

python_bin = sys.executable


def validate_data_dir(path):
    if not os.path.exists(path):
        log.error(f"Data dir {path} not exists, check your params")
        return False

    dir_content = os.listdir(path)

    if len(dir_content) == 0:
        log.error(f"Data dir {path} is empty, check your params")

    subdirs = [f for f in dir_content if os.path.isdir(os.path.join(path, f))]

    if len(subdirs) == 0:
        log.warn(f"No subdir found in data dir")

    ok_dir = [d for d in subdirs if re.findall(r"^\d+_.+", d)]

    if len(ok_dir) == 0:
        log.warning(f"No leagal dataset found. Try find avaliable images")
        imgs = get_total_images(path, False)
        captions = glob.glob(path + '/*.txt')
        log.info(f"{len(imgs)} images found, {len(captions)} captions found")
        if len(imgs) > 0:
            num_repeat = suggest_num_repeat(len(imgs))
            dataset_path = os.path.join(path, f"{num_repeat}_zkz")
            os.makedirs(dataset_path)
            for i in imgs:
                shutil.move(i, dataset_path)
            if len(captions) > 0:
                for c in captions:
                    shutil.move(c, dataset_path)
            log.info(f"Auto dataset created {dataset_path}")
        else:
            log.error("No image found in data dir")
            return False

    return True


def suggest_num_repeat(img_count):
    if img_count <= 10:
        return 7
    elif 10 < img_count <= 50:
        return 5
    elif 50 < img_count <= 100:
        return 3

    return 1


def check_training_params(data):
    potential_path = [
        "train_data_dir", "reg_data_dir", "output_dir"
    ]
    file_paths = [
        "sample_prompts"
    ]
    for p in potential_path:
        if p in data and not os.path.exists(data[p]):
            return False

    for f in file_paths:
        if f in data and not os.path.exists(data[f]):
            return False
    return True


def get_total_images(path, recursive=True):
    if recursive:
        image_files = glob.glob(path + '/**/*.jpg', recursive=True)
        image_files += glob.glob(path + '/**/*.jpeg', recursive=True)
        image_files += glob.glob(path + '/**/*.png', recursive=True)
    else:
        image_files = glob.glob(path + '/*.jpg')
        image_files += glob.glob(path + '/*.jpeg')
        image_files += glob.glob(path + '/*.png')
    return image_files


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run(command,
        desc: Optional[str] = None,
        errdesc: Optional[str] = None,
        custom_env: Optional[list] = None,
        live: Optional[bool] = True,
        shell: Optional[bool] = None):

    if shell is None:
        shell = False if sys.platform == "win32" else True

    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=shell, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=shell, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(command, desc=None, live=False):
    return run(f'"{python_bin}" -m pip {command}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def check_run(file: str) -> bool:
    result = subprocess.run([python_bin, file], capture_output=True, shell=False)
    log.info(result.stdout.decode("utf-8").strip())
    return result.returncode == 0

def tk_window():
    window = tkinter.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()

def open_file_selector(
        initialdir,
        title,
        filetypes) -> str:
    try:
        tk_window()
        filename = askopenfilename(
            initialdir=initialdir, title=title,
            filetypes=filetypes
        )
        return filename
    except:
        return ""


def open_directory_selector(initialdir) -> str:
    try:
        tk_window()
        directory = askdirectory(
            initialdir=initialdir
        )
        return directory
    except:
        return ""

def download_oss(oss_url, local_dir):
    bucket_name = oss_url[oss_url.find('//')+2:oss_url.find('.')]
    tempendpoint = oss_url[oss_url.find('//')+2:]
    endpoint = tempendpoint[:tempendpoint.find('/')].replace(bucket_name+".",'')
    oss_key = oss_url.replace("https://"+bucket_name+"."+endpoint+"/","")
    file_name = oss_key[oss_key.rfind('/')+1:]
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    bucket.get_object_to_file(oss_key, local_dir+"/"+file_name)
   
def upload_to_oss(local_path):
    file_name = local_path[local_path.rfind('/')+1:]
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, 'https://oss-cn-shanghai.aliyuncs.com', 'kidlife-model')
    bucket.put_object_from_file('lora/'+file_name, local_path)
    return 'https://kidlife-model.oss-cn-shanghai.aliyuncs.com/lora/'+file_name

def notify_finish(notify_url, doppelganger_id, sd_service_url, model_url, notify_status):
    resp = requests.post(notify_url, json={'status':notify_status,'doppelgangerId':doppelganger_id,'modelUrl':model_url,'sdServiceUrl':sd_service_url})
    print(resp)
    print(resp.text)
