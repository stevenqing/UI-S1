import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


RUN_DIR=Path(__file__).resolve().parent
ARMS=("C_uni","C_cond","C_rand","C_self")


def paired(rows,left,right,resamples,seed):
    by_fold={}
    for row in rows: by_fold.setdefault(row['fold'],{}).setdefault(row['group'],[]).append(row)
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(resamples):
        selected=[]
        for fold in sorted(by_fold):
            groups=sorted(by_fold[fold])
            for group in rng.choice(groups,size=len(groups),replace=True): selected.extend(by_fold[fold][group])
        samples.append(float(np.mean([int(row[left])-int(row[right]) for row in selected])))
    point=float(np.mean([int(row[left])-int(row[right]) for row in rows]))
    return {'point_delta':point,'ci_99':[float(np.quantile(samples,.005)),float(np.quantile(samples,.995))],'wins':sum(row[left] and not row[right] for row in rows),'losses':sum(row[right] and not row[left] for row in rows),'rows':len(rows),'resamples':resamples,'seed':seed}


def main():
    config=yaml.safe_load((RUN_DIR/'configs/care_prereg.yaml').read_text()); outers=[]
    for fold in range(5):
        path=RUN_DIR/f'router/outer-{fold}.json'; pre=RUN_DIR/f'router/outer-{fold}.pretest.json'
        if not path.is_file() or not pre.is_file(): raise FileNotFoundError(path)
        value=json.loads(path.read_text()); seal=json.loads(pre.read_text())
        if value['status']!='PASS_A1_OUTER_COMPLETE' or seal['status']!='PASS_A1_SELECTION_FROZEN' or fold in seal['opened_development_folds']: raise ValueError(f'CARE A1 invalid outer {fold}')
        outers.append(value)
    rows=[row for outer in outers for row in outer['rows']]
    if len(rows)!=2080+1581 or len({(row['benchmark'],row['row_id']) for row in rows})!=len(rows): raise ValueError('CARE A1 row coverage mismatch')
    comparisons={}; choices={}; regret={}
    for index,benchmark in enumerate(('mind2web','screenspot_pro')):
        selected=[row for row in rows if row['benchmark']==benchmark]
        comparisons[benchmark]={
            'pass_selected_minus_static':paired(selected,'selected_pass','static_pass',10000,20261401+index),
            'safe_selected_minus_static':paired(selected,'selected_safe','static_safe',10000,20261501+index),
        }
        choices[benchmark]=dict(Counter(row['selected_arm'] for row in selected))
        oracle=float(np.mean([row['oracle_pass'] for row in selected])); static=float(np.mean([row['static_pass'] for row in selected])); routed=float(np.mean([row['selected_pass'] for row in selected]))
        regret[benchmark]={'oracle_pass':oracle,'static_pass':static,'routed_pass':routed,'oracle_gain':oracle-static,'captured_gain':routed-static,'fraction_oracle_gain_captured':(routed-static)/(oracle-static) if oracle>static else None}
    mde=config['mde']; mind=comparisons['mind2web']['pass_selected_minus_static']; screen=comparisons['screenspot_pro']['pass_selected_minus_static']
    one_positive=mind['ci_99'][0]>0 or screen['ci_99'][0]>0
    other_safe=(screen['point_delta']>=-mde['screenspot_pro'] and screen['ci_99'][1]>=0) if mind['ci_99'][0]>0 else (mind['point_delta']>=-mde['mind2web'] and mind['ci_99'][1]>=0)
    step_safe=all(comparisons[b]['safe_selected_minus_static']['point_delta']>=-mde[b] for b in comparisons)
    result={'schema_version':1,'status':'PASS_A1_ADJUDICATED','outcome':'PROCEED_TO_E0' if one_positive and other_safe and step_safe else 'CLOSE_ROUTING','gates':{'A1_one_pass_ci_positive':one_positive,'A1_other_pass_noninferior':other_safe,'A1_no_safe_step_mde_loss':step_safe},'comparisons':comparisons,'choice_counts':choices,'regret':regret,'outer_epochs':[outer['final_epochs'] for outer in outers],'static_arms':[outer['static_arms'] for outer in outers]}
    (RUN_DIR/'router_adjudication.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__': main()
