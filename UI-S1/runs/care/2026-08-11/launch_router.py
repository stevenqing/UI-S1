import os
import subprocess
import sys
from pathlib import Path


RUN_DIR=Path(__file__).resolve().parent
PYTHON=RUN_DIR.parents[2]/'.venv-scaleup/bin/python'


def main():
    if not Path('/proc/2274').exists(): raise RuntimeError('protected PID 2274 absent')
    out=RUN_DIR/'router'; logs=out/'logs'; out.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True); processes=[]
    for fold in range(5):
        output=out/f'outer-{fold}.json'
        if output.exists() or (out/f'outer-{fold}.pretest.json').exists(): raise FileExistsError(output)
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(fold); env['OMP_NUM_THREADS']='1'; log_path=logs/f'outer-{fold}.log'; log=log_path.open('w',buffering=1)
        process=subprocess.Popen([str(PYTHON),str(RUN_DIR/'router_train.py'),'--outer-fold',str(fold),'--output',str(output)],cwd=RUN_DIR,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((fold,process,log,log_path)); print(f'started CARE A1 outer={fold} gpu={fold} pid={process.pid}',flush=True)
    failures=[]
    for fold,process,log,path in processes:
        code=process.wait(); log.close(); print(f'finished CARE A1 outer={fold} exit={code}',flush=True)
        if code: failures.append((fold,code,str(path)))
    if failures: print(failures,file=sys.stderr); raise SystemExit(1)
    if not Path('/proc/2274').exists(): raise RuntimeError('protected PID 2274 disappeared')
    print('CARE_A1_ALL_OUTERS_PASS',flush=True)


if __name__=='__main__': main()
