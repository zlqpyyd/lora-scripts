import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import starlette.responses as starlette_responses
import toml
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import mikazuki.utils as utils
import mikazuki.process as process
from mikazuki.log import log
from mikazuki.models import TaggerInterrogateRequest
from mikazuki.tagger.interrogator import (available_interrogators,
                                          on_interrogate)
from mikazuki.tasks import tm

app = FastAPI()

avaliable_scripts = [
    "networks/extract_lora_from_models.py",
    "networks/extract_lora_from_dylora.py"
]

# fix mimetype error in some fucking systems
_origin_guess_type = starlette_responses.guess_type


def _hooked_guess_type(*args, **kwargs):
    url = args[0]
    r = _origin_guess_type(*args, **kwargs)
    if url.endswith(".js"):
        r = ("application/javascript", None)
    elif url.endswith(".css"):
        r = ("text/css", None)
    return r


starlette_responses.guess_type = _hooked_guess_type

# cors middleware
if os.environ.get("ENABLE_APP_CORS") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8004"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    # response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.post("/api/run")
async def create_toml_file(request: Request):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}.toml")
    toml_data = await request.body()
    j = json.loads(toml_data.decode("utf-8"))
    #download images
    print("train_request,", j)
    train_images = j["train_images"]
    local_dir = j["train_data_dir"] + "/" + str(j["repeats"]) + "_" + j["output_name"];
    if len(train_images)!=0:
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        for imageUrl in train_images:
            print("downloading image,", imageUrl, local_dir)
            utils.download_oss(imageUrl, local_dir)
    
    if not utils.validate_data_dir(j["train_data_dir"]):
        return {
            "status": "fail",
            "detail": "训练数据集路径不存在或没有图片，请检查目录。"
        }

    suggest_cpu_threads = 8 if len(utils.get_total_images(j["train_data_dir"])) > 100 else 2
    trainer_file = "./sd-scripts/train_network.py"

    model_train_type = j.pop("model_train_type", "sd-lora")
    if model_train_type == "sdxl-lora":
        trainer_file = "./sd-scripts/sdxl_train_network.py"
    elif model_train_type == "sd-dreambooth":
        trainer_file = "./sd-scripts/train_db.py"
    elif model_train_type == "sdxl-finetune":
        trainer_file = "./sd-scripts/sdxl_train.py"

    multi_gpu = j.pop("multi_gpu", False)

    def is_promopt_like(s):
        for p in ["--n", "--s", "--l", "--d"]:
            if p in s:
                return True
        return False

    sample_prompts = j.get("sample_prompts", None)
    if sample_prompts is not None and not os.path.exists(sample_prompts) and is_promopt_like(sample_prompts):
        sample_prompts_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}-promopt.txt")
        with open(sample_prompts_file, "w", encoding="utf-8") as f:
            f.write(sample_prompts)
        j["sample_prompts"] = sample_prompts_file
        log.info(f"Wrote promopts to file {sample_prompts_file}")

    with open(toml_file, "w") as f:
        f.write(toml.dumps(j))

    #coro = asyncio.to_thread(process.run_train, toml_file, trainer_file, multi_gpu, suggest_cpu_threads)
    #asyncio.create_task(coro)

    task = process.pre_run_train(toml_file, trainer_file, multi_gpu, suggest_cpu_threads)
    coro = asyncio.to_thread(process.run_train, task.task_id, j["output_name"], local_dir, j["doppelganger_id"], j["sd_service_url"], j["notify_url"])
    asyncio.create_task(coro)

    return {"status": "success","task_id":task.task_id}


@app.post("/api/run_script")
async def run_script(request: Request, background_tasks: BackgroundTasks):
    paras = await request.body()
    j = json.loads(paras.decode("utf-8"))
    script_name = j["script_name"]
    if script_name not in avaliable_scripts:
        return {"status": "fail"}
    del j["script_name"]
    result = []
    for k, v in j.items():
        result.append(f"--{k}")
        if not isinstance(v, bool):
            value = str(v)
            if " " in value:
                value = f'"{v}"'
            result.append(value)
    script_args = " ".join(result)
    script_path = Path(os.getcwd()) / "sd-scripts" / script_name
    cmd = f"{utils.python_bin} {script_path} {script_args}"
    background_tasks.add_task(utils.run, cmd)
    return {"status": "success"}


@app.post("/api/interrogate")
async def run_interrogate(req: TaggerInterrogateRequest, background_tasks: BackgroundTasks):
    interrogator = available_interrogators.get(req.interrogator_model, available_interrogators["wd14-convnextv2-v2"])
    background_tasks.add_task(on_interrogate,
                              image=None,
                              batch_input_glob=req.path,
                              batch_input_recursive=False,
                              batch_output_dir="",
                              batch_output_filename_format="[name].[output_extension]",
                              batch_output_action_on_conflict=req.batch_output_action_on_conflict,
                              batch_remove_duplicated_tag=True,
                              batch_output_save_json=False,
                              interrogator=interrogator,
                              threshold=req.threshold,
                              additional_tags=req.additional_tags,
                              exclude_tags=req.exclude_tags,
                              sort_by_alphabetical_order=False,
                              add_confident_as_weight=False,
                              replace_underscore=req.replace_underscore,
                              replace_underscore_excludes=req.replace_underscore_excludes,
                              escape_tag=req.escape_tag,
                              unload_model_after_running=True
                              )
    return {"status": "success"}

# @app.get("/api/schema/{name}")
# async def get_schema(name: str):
#     with open(os.path.join(os.getcwd(), "mikazuki", "schema", name), encoding="utf-8") as f:
#         content = f.read()
#         return Response(content=content, media_type="text/plain")


@app.get("/api/pick_file")
async def pick_file(picker_type: str):
    if picker_type == "folder":
        coro = asyncio.to_thread(utils.open_directory_selector, os.getcwd())
    elif picker_type == "modelfile":
        file_types = [("checkpoints", "*.safetensors;*.ckpt;*.pt"), ("all files", "*.*")]
        coro = asyncio.to_thread(utils.open_file_selector, os.getcwd(), "Select file", file_types)

    result = await coro
    if result == "":
        return {
            "status": "fail"
        }

    return {
        "status": "success",
        "path": result
    }


@app.get("/api/tasks")
async def get_tasks():
    return tm.dump()

@app.get("/api/tasks/query/{task_id}")
async def query_task(task_id: str):
    task = tm.get_task(task_id)
    if not task:
        return
    return {"id": task.task_id, "status": task.status.name}

@app.get("/api/tasks/terminate/{task_id}")
async def terminate_task(task_id: str):
    tm.terminate_task(task_id)
    return {"status": "success"}


@app.get("/")
async def index():
    return FileResponse("./frontend/dist/index.html")

app.mount("/", StaticFiles(directory="frontend/dist"), name="static")
