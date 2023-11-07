
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


def run_train(task_id: str, lora_name: str, train_dir: str, doppelganger_id, sd_service_url: str, notify_url: str):
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
