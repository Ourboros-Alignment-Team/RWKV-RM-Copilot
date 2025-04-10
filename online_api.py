import os
from gevent import monkey
monkey.patch_all()

from online_trainer import RMOnlineTrainer,config
from bottle import route, run, request, response
from threading import Thread

app=RMOnlineTrainer()

@route('/infer', method='POST')
def infer():
    req = dict(request.json)
    history = req.get('history', '')
    response = req.get('response', "")
    score=app.inference(history, response)
    return {"score":score}

@route('/train', method='POST')
def train():
    req = dict(request.json)
    history = req.get('history', '')
    response_list = req.get('response_list', [])
    score_list = req.get('score_list', [])
    save_hist_dir = req.get('save_hist_dir', '')
    thread = Thread(
        target=app.train,
        kwargs={
            'history': history,
            'response_list': response_list,
            'score_list': score_list,
            'save_history': True,
            'save_hist_dir': save_hist_dir,
        }
    )
    thread.start()
    
    
@route('/save_checkpoint', method='POST')
def save_checkpoint():
    req = dict(request.json)
    save_path = req.get('save_ckpt_dir', '')
    app.save_checkpoint(save_path)
    return {"status": "ok"}


run(host=config.host, port=config.port, server="paste")
