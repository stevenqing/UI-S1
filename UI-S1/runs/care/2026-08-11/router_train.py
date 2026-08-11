import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from router_data import ARMS, BENCHMARKS, VUS, build_router_data, deterministic_permutations, fit_source_statistics, fit_standardizer, load_label_folds, load_public, load_source_metadata, torch_batch
from router_model import AcquisitionRouter, permute_router_batch, router_loss


RUN_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUN_DIR / "configs/care_prereg.yaml"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_model(input_dim, config):
    values = config["router_A1"]["model"]
    return AcquisitionRouter(input_dim, values["width"], values["layers"], values["heads"], values["dropout"])


def evaluate_loss(model, data, device, batch_size=1024):
    model.eval(); total=0.0
    normalization=torch.as_tensor(float(data.weights.sum()),device=device)
    with torch.no_grad():
        for start in range(0,len(data),batch_size):
            indices=np.arange(start,min(len(data),start+batch_size)); batch=torch_batch(data,indices,device)
            loss,_=router_loss(model,batch,normalization); total+=float(loss)
    return total


def train_epoch(model, data, optimizer, epoch, config, seed, device):
    model.train(); values=config["router_A1"]["optimizer"]
    indices=np.random.default_rng(seed+epoch).permutation(len(data)); batch_size=values["batch_size"]
    permutations=deterministic_permutations(data.row_ids,data.benchmarks,epoch,seed)
    normalization=torch.as_tensor(float(data.weights.sum()),device=device)
    optimizer.zero_grad(set_to_none=True); total=0.0
    for start in range(0,len(indices),batch_size):
        selected=indices[start:start+batch_size]; batch=torch_batch(data,selected,device)
        batch=permute_router_batch(batch,torch.as_tensor(permutations[selected],device=device))
        loss,_=router_loss(model,batch,normalization); loss.backward(); total+=float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(),values["gradient_clip_norm"]); optimizer.step()
    return total


def train_checkpoint(train, validation, config, seed, device):
    standardizer=fit_standardizer(train); train=standardizer.transform(train); validation=standardizer.transform(validation)
    set_seed(seed); model=make_model(train.features.shape[-1],config).to(device); values=config["router_A1"]["optimizer"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=values["learning_rate"],weight_decay=values["weight_decay"])
    best=float("inf"); best_epoch=0; best_state=None; stale=0; history=[]
    for epoch in range(1,values["maximum_epochs"]+1):
        train_loss=train_epoch(model,train,optimizer,epoch,config,seed,device); val_loss=evaluate_loss(model,validation,device)
        history.append({"epoch":epoch,"training_loss":train_loss,"validation_loss":val_loss})
        if val_loss < best-values["minimum_improvement"]:
            best=val_loss; best_epoch=epoch; best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=values["patience"]: break
    if best_state is None: raise ValueError("CARE A1 checkpoint selection failed")
    return best_epoch,{"selected_epoch":best_epoch,"selected_validation_loss":best,"epochs_run":len(history),"history":history}


def train_fixed(train, epochs, config, seed, device):
    standardizer=fit_standardizer(train); train=standardizer.transform(train); set_seed(seed)
    model=make_model(train.features.shape[-1],config).to(device); values=config["router_A1"]["optimizer"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=values["learning_rate"],weight_decay=values["weight_decay"]); history=[]
    for epoch in range(1,epochs+1): history.append({"epoch":epoch,"training_loss":train_epoch(model,train,optimizer,epoch,config,seed,device)})
    return model,standardizer,history


def predict(model,data,standardizer,device):
    data=standardizer.transform(data); model.eval(); output=[]
    with torch.no_grad():
        for start in range(0,len(data),1024):
            indices=np.arange(start,min(len(data),start+1024)); logits=model(torch_batch(data,indices,device).features).float().cpu().numpy()
            for offset,index in enumerate(indices): output.append({"benchmark":data.benchmarks[index],"row_id":data.row_ids[index],"fold":int(data.folds[index]),"group":data.groups[index],"selected_arm":ARMS[int(np.argmax(logits[offset]))],"arm_logits":logits[offset].tolist()})
    return output


def static_arms(data,config):
    order=config["router_A1"]["static_tie_order"]; output={}
    for benchmark in BENCHMARKS:
        indices=[i for i,name in enumerate(data.benchmarks) if name==benchmark]
        rates={arm:float(data.targets[indices,ARMS.index(arm)].mean()) for arm in ARMS}
        selected=max(ARMS,key=lambda arm:(rates[arm],-order.index(arm)))
        output[benchmark]={"arm":selected,"development_pass_at_12":rates[selected],"all_arms":rates}
    return output


def load_test_after_pretest(outer_fold,pretest_path):
    if not pretest_path.is_file(): raise PermissionError("CARE-K6 test labels sealed before pretest")
    record=json.loads(pretest_path.read_text())
    if record.get("status")!="PASS_A1_SELECTION_FROZEN" or record.get("outer_fold")!=outer_fold or outer_fold in record.get("opened_development_folds",[]): raise PermissionError("CARE-K6 invalid A1 pretest")
    return load_label_folds([outer_fold])


def run_outer(outer_fold,output,device):
    config=yaml.safe_load(CONFIG_PATH.read_text()); public=load_public(); source_metadata=load_source_metadata(); dev_folds=[fold for fold in range(5) if fold!=outer_fold]; dev_labels=load_label_folds(dev_folds)
    checkpoints=[]
    for validation_fold in dev_folds:
        train_folds=[fold for fold in dev_folds if fold!=validation_fold]
        statistics=fit_source_statistics(source_metadata,dev_labels,train_folds)
        train=build_router_data(public,dev_labels,train_folds,source_metadata,statistics,leave_one=True); validation=build_router_data(public,dev_labels,[validation_fold],source_metadata,statistics)
        epoch,report=train_checkpoint(train,validation,config,config["seed"]+outer_fold*100+validation_fold,device)
        checkpoints.append({"validation_fold":validation_fold,"train_folds":train_folds,**report})
    epochs=max(1,int(math.floor(float(np.median([row["selected_epoch"] for row in checkpoints]))+0.5)))
    final_statistics=fit_source_statistics(source_metadata,dev_labels,dev_folds)
    dev=build_router_data(public,dev_labels,dev_folds,source_metadata,final_statistics,leave_one=True); static=static_arms(dev,config)
    model,standardizer,history=train_fixed(dev,epochs,config,config["seed"]+outer_fold*100+99,device)
    pretest=output.with_name(f"outer-{outer_fold}.pretest.json")
    atomic_json(pretest,{"schema_version":1,"status":"PASS_A1_SELECTION_FROZEN","outer_fold":outer_fold,"opened_development_folds":dev_folds,"selected_epochs":[row["selected_epoch"] for row in checkpoints],"final_epochs":epochs,"static_arms":static})
    test_labels=load_test_after_pretest(outer_fold,pretest); test=build_router_data(public,test_labels,[outer_fold],source_metadata,final_statistics); predictions=predict(model,test,standardizer,device)
    vus=json.loads((VUS/"set_ranker_adjudication.json").read_text())["outputs"]; rows=[]
    for index,prediction in enumerate(predictions):
        benchmark=prediction["benchmark"]; row_id=prediction["row_id"]; selected=prediction["selected_arm"]; static_arm=static[benchmark]["arm"]
        target=test.targets[index]
        rows.append({**prediction,"static_arm":static_arm,"selected_pass":bool(target[ARMS.index(selected)]),"static_pass":bool(target[ARMS.index(static_arm)]),"oracle_pass":bool(target.max()),"selected_safe":bool(vus[benchmark][selected]["safe"][row_id]),"static_safe":bool(vus[benchmark][static_arm]["safe"][row_id])})
    result={"schema_version":1,"status":"PASS_A1_OUTER_COMPLETE","outer_fold":outer_fold,"checkpoints":checkpoints,"final_epochs":epochs,"static_arms":static,"final_history":history,"rows":rows}
    atomic_json(output,result); print(json.dumps({"outer_fold":outer_fold,"final_epochs":epochs,"static_arms":static,"rows":len(rows)},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--outer-fold",type=int,choices=range(5),required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args()
    if not Path('/proc/2274').exists(): raise RuntimeError('protected PID 2274 absent')
    if args.output.exists() or args.output.with_name(f"outer-{args.outer_fold}.pretest.json").exists(): raise FileExistsError(args.output)
    run_outer(args.outer_fold,args.output,torch.device(args.device))


if __name__=="__main__": main()
