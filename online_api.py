import os
from gevent import monkey

monkey.patch_all()

from online_trainer import RMOnlineTrainer, config
from bottle import route, run, request, response
from threading import Thread
import time
app = RMOnlineTrainer()


@route("/infer", method="POST")
def infer():
    req = dict(request.json)
    history = req.get("history", "")
    response = req.get("response", "")
    score = app.inference(history, response)
    return {"score": score}


@route("/train", method="POST")
def train():
    req = dict(request.json)
    data_list = req.get("data_list", [])
    batch_size = req.get("batch_size", config.train.batch_size)
    save_ckpt = req.get("save_ckpt", False)
    save_ckpt_dir = req.get(
        "save_ckpt_dir", f"{config.train.save_ckpt_dir}/auto_save_rm.pth"
    )

    thread = Thread(
        target=app.train,
        kwargs={
            "data_list": data_list,
            "batch_size": batch_size,
            "save_ckpt": save_ckpt,
            "save_ckpt_dir": save_ckpt_dir,
        },
    )
    thread.start()


@route("/save_checkpoint", method="POST")
def save_checkpoint():
    req = dict(request.json)
    save_path = req.get("save_ckpt_dir", "")
    app.save_checkpoint(save_path)
    return {"status": "ok"}


@route("/test_server", method="GET")
def test_server():
    return {"status": "ok"}


run(host=config.host, port=config.port, server="paste")
