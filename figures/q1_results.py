"""Regenerate Q1 revision tables and scientific figures from complete recorded data."""
from pathlib import Path
from fractions import Fraction as F
import json,sys,hashlib
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'artifacts'

def load(name):
 x=json.loads((A/name).read_text());assert x.get('complete',x.get('passed',False));return x

def quant(a):return dict(zip(['p50','p90','p99','max'],map(float,np.quantile(a,[.5,.9,.99,1]))))
def ratio(row,a,b):return float(F(*map(int,row['arms'][a]['cost_exact']))/F(*map(int,row['arms'][b]['cost_exact'])))
def table(caption,label,columns,header,rows):
 return '\\begin{table}[!t]\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{@{}'+columns+'@{}}\\toprule\n'+header+'\\\\\\midrule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{table}\n'

def main():
 f=load('q1_factorial_final.json');i=load('q1_integrated.json');b=load('q1_budget.json');fa=load('q1_families.json');rp=load('q1_repair.json')
 s={'inputs':{x:hashlib.sha256((A/x).read_bytes()).hexdigest() for x in ['q1_factorial_final.json','q1_integrated.json','q1_budget.json','q1_families.json','q1_repair.json']},'arms':{},'pairs':[],'integrated':{},'budget':[],'families':[]}
 labels={'affine_bubble':'Affine + bubble','affine_green':'Affine + Green','sdp_bubble':'SDP + bubble','sdp_green':'SDP + Green','sdp_gap':'SDP + gap','sdp_selector':'SDP + gap/Green','sdp_green_allaxes':'SDP + Green, all axes','iterated_cut_qp':'Iterated cut QP','workspace_sdp_green':'Workspace Green'}
 rt=[]
 for name in f['protocol']['arms']:
  records=[r['arms'][name] for r in f['rows']];good=[r for r in records if r['ok']];gaps=[r['normalized_gap_upper'] for r in good];t=quant([r['ms'] for r in records])
  s['arms'][name]=dict(n=len(records),success=len(good),plain=sum(r.get('plain',False) for r in good),time=t,gap_median=float(np.median(gaps)),gap_max=max(gaps),statuses={st:sum(r.get('status',r.get('reason','CERTIFIED_PHYSICAL'))==st for r in records) for st in set(r.get('status',r.get('reason','CERTIFIED_PHYSICAL')) for r in records)})
  rt.append([labels[name],f'{len(good)}/240',f'{100*np.median(gaps):.1f}',f'{t["p50"]:.1f}'])
 (P/'table_q1_factorial.tex').write_text(table('All 240 cases, including failures. Gap is median normalized certificate upper-gap view (\\%) on successes only; time is median recovery ms on all 240 calls, excluding archived SDP acquisition.','tab:q1factor','lrrr','Arm & Strict & Gap & ms',rt))
 rt=[]
 for a,bn,label in [('sdp_bubble','affine_bubble','SDP/affine, bubble'),('sdp_green','affine_green','SDP/affine, Green'),('affine_green','affine_bubble','Green/bubble, affine'),('sdp_green','sdp_bubble','Green/bubble, SDP')]:
  rr=[r for r in f['rows'] if r['arms'][a]['ok'] and r['arms'][bn]['ok']];rat=[ratio(r,a,bn) for r in rr];repaired=[r for r in rr if not r['arms'][a].get('plain',False) and not r['arms'][bn].get('plain',False)]
  q=dict(a=a,b=bn,n=len(rr),ratio_p50=float(np.median(rat)),wins=sum(x<1-1e-12 for x in rat),ties=sum(abs(x-1)<1e-12 for x in rat),losses=sum(x>1+1e-12 for x in rat),repaired_n=len(repaired),repaired_p50=float(np.median([ratio(r,a,bn) for r in repaired])));s['pairs'].append(q)
  rt.append([label,str(q['n']),f'{q["ratio_p50"]:.3f}',f'{q["wins"]}/{q["ties"]}/{q["losses"]}'])
 (P/'table_q1_pairs.tex').write_text(table('Paired energy ratios on joint successes; below one favors numerator. W/T/L are lower/equal/higher energy counts (display equality threshold $10^{-12}$).','tab:q1pairs','lrrr','Comparison & $n$ & Median & W/T/L',rt))
 rt=[]
 for key,label in [('physical_return_ms','Physical emission'),('objective_return_ms','Then objective emission')]:
  q=quant([r[key] for r in i['rows']]);s['integrated'][key]=q;rt.append([label]+[f'{q[k]:.1f}' for k in ['p50','p90','p99']])
 s['integrated'].update(n=len(i['rows']),physical=sum(r['physical_ok'] for r in i['rows']),objective=sum(r['objective_ok'] for r in i['rows']),setup=quant([r['ms'] for r in i['setup']]),dispatch={x:sum(r['numerical']['dispatch_reason']==x for r in i['rows']) for x in sorted(set(r['numerical']['dispatch_reason'] for r in i['rows']))})
 (P/'table_q1_integrated.tex').write_text(table('Sequential two-emission service on 1,000 existing snapshots. Per-call elapsed ms include all failed attempts, checks and serialization; setup is separate.','tab:q1integrated','lrrr','Endpoint & p50 & p90 & p99',rt))
 rt=[]
 for budget in b['protocol']['budget']:
  for warm in b['protocol']['warm']:
   rr=[r for r in b['rows'] if r['budget']==budget and r['warm']==warm];q=quant([r['objective_return_ms'] for r in rr]);n=sum(r['numerical']['source']=='factor' for r in rr)
   x=dict(budget=budget,factor_warm=warm,n=len(rr),local=n,physical=sum(r['physical_ok'] for r in rr),objective=sum(r['objective_ok'] for r in rr),time=q,dispatch={x:sum(r['numerical']['dispatch_reason']==x for r in rr) for x in sorted(set(r['numerical']['dispatch_reason'] for r in rr))});s['budget'].append(x)
   rt.append([str(budget),'on' if warm else 'off',f'{n}/{len(rr)}',f'{q["p50"]:.1f}',f'{q["p99"]:.1f}'])
 (P/'table_q1_budget.tex').write_text(table('Factor-iteration/warm-state ablation: same 80 snapshots per setting. Conic graph and warm-start policy are unchanged. Time is the second emission ms.','tab:q1budget','rlrrr','Budget & Factor warm & Local & p50 & p99',rt))
 rt=[]
 for label in sorted(set(r['family_label'] if 'family_label' in r else r['id'].split('/')[0] for r in fa['rows'])):
  rr=[r for r in fa['rows'] if (r.get('family_label',r['id'].split('/')[0]))==label];fam=rr[0]['family'];success=sum(r['arms']['affine_green']['ok'] for r in rr);q=quant([r['total_ms'] for r in rr]);s['families'].append(dict(label=label,family=fam,n=len(rr),success=success,time=q));rt.append([f"({fam['d']},{fam['N']},{fam['eta']})",f'{success}/{len(rr)}',f'{q["p50"]:.1f}'])
 (P/'table_q1_families.tex').write_text(table('Additional exact-certified families, all $k=1,\\ell=0$: 12 matched deterministic scenes each. Time includes fresh exact/model setup and recovery.','tab:q1families','lrr','$(d,N,\\eta)$ & Strict & p50 ms',rt))
 rr=[r for r in rp['rows'] if r['selected_index'] is not None];conds=[r['candidates'][r['selected_index']]['condition_Z'] for r in rr];loss=[c['loss_from_best'] for r in rr for c in r['candidates'] if c['valid']];s['repair']=dict(points=len(rr),selected_indices={str(j):sum(r['selected_index']==j for r in rr) for j in range(6)},selected_condition=quant(conds),loss=quant(loss),max_shift=max(max(r['shifts']) for r in rr))
 macros=dict(QOnePhysical=str(s['integrated']['physical']),QOneObjective=str(s['integrated']['objective']),QOnePhysMedian=f"{s['integrated']['physical_return_ms']['p50']:.1f}",QOneObjMedian=f"{s['integrated']['objective_return_ms']['p50']:.1f}",QOneSetup=f"{s['integrated']['setup']['p50']:.1f}",QOneAllAxes=str(s['arms']['sdp_green_allaxes']['success']),QOneWorkspace=str(s['arms']['workspace_sdp_green']['success']),QOneQP=str(s['arms']['iterated_cut_qp']['success']),QOneRepairPoints=str(len(rr)),QOneRepairCond=f"{np.median(conds):.2g}",QOneRepairCondMax=f"{max(conds):.2g}",QOneRepairLoss=f"{max(loss):.2g}")
 (P/'results_q1.tex').write_text('% Generated from recorded Q1 data.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
 (P/'claims_q1.json').write_text(json.dumps(s,indent=2)+'\n')
 plot(f,i,b)

def plot(f,i,b):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from math import comb
 import figstyle;figstyle.use()
 fig,ax=plt.subplots(1,3,figsize=(figstyle.TEXT_W,2.55),layout='constrained')
 r=next(r for r in f['rows'] if r['n']==2 and r['arms']['sdp_green']['ok'] and not r['arms']['sdp_green'].get('plain',False));a=r['arms']['sdp_green'];t=next(t for t in a['attempts'] if t['ok'] and t['mode']==a['method'] and t['axis']==a['axis'] and t['sign']==a['sign'])
 def decode(x):return np.array([float(F(*map(int,v))) for v in x['values']]).reshape(x['shape'])
 for payload,label,color in [(t['reference'],'SDP reference','#D55E00'),(a['Gamma'],'Certified curve','#007C83')]:
  G=decode(payload);s=np.linspace(0,1,151);D=G.shape[-1]-1;B=np.array([comb(D,j)*s**j*(1-s)**(D-j) for j in range(D+1)]);xy=np.concatenate([g@B for g in G],axis=1);ax[0].plot(*xy,label=label,color=color)
 for c,rad in r['obstacles']:ax[0].add_patch(plt.Circle(c,rad,color='.5',alpha=.25))
 for x in [-2,2]:ax[0].axvline(x,color='.5',ls=':',lw=.6)
 ax[0].set(aspect='equal',xlabel='x (scene units)',ylabel='y (scene units)',title='(a) Recorded '+r['id'])
 rr=[r for r in f['rows'] if r['arms']['sdp_green']['ok'] and r['arms']['affine_green']['ok']];xx=[r['arms']['affine_green']['cost'] for r in rr];yy=[r['arms']['sdp_green']['cost'] for r in rr];ax[1].scatter(xx,yy,s=7,color='#007C83',alpha=.7);lim=[min(xx+yy)*.9,max(xx+yy)*1.1];ax[1].plot(lim,lim,color='.5',ls='--');ax[1].set(xscale='log',yscale='log',xlabel='Affine + Green energy',ylabel='SDP + Green energy',title=f'(b) Same-mode pairs (n={len(rr)})')
 for ref,color in [('affine','#007C83'),('sdp','#D55E00')]:
  rr=[r for r in f['rows'] if r['arms'][ref+'_green']['ok'] and r['arms'][ref+'_bubble']['ok']];v=np.sort([ratio(r,ref+'_green',ref+'_bubble') for r in rr]);ax[2].plot(v,np.arange(1,len(v)+1)/len(v),label=f'{('Affine' if ref=='affine' else 'SDP')} (n={len(v)})',color=color)
 ax[2].axvline(1,color='.5',ls='--');ax[2].set(xscale='log',xlabel='Energy Green / bubble',ylabel='Fraction of joint successes',title='(c) Same-reference mode pairs');ax[2].legend(fontsize=6,frameon=False)
 figstyle.save(fig,P/'fig_q1_factorial');plt.close(fig)
 fig,ax=plt.subplots(1,3,figsize=(figstyle.TEXT_W,2.5),layout='constrained')
 for key,label,color in [('physical_return_ms','Physical','#007C83'),('objective_return_ms','Then objective','#D55E00')]:
  v=np.sort([r[key] for r in i['rows']]);ax[0].plot(v,np.arange(1,len(v)+1)/len(v),label=label,color=color)
 ax[0].set(xlabel='Elapsed time (ms)',ylabel='Fraction of all 1,000 calls',title='(a) Measured service boundaries');ax[0].legend(fontsize=6,frameon=False)
 groups=[];labels=[]
 for budget in b['protocol']['budget']:
  for warm in b['protocol']['warm']:groups.append([r['objective_return_ms'] for r in b['rows'] if r['budget']==budget and r['warm']==warm]);labels.append(str(budget)+('/on' if warm else '/off'))
 ax[1].boxplot(groups,tick_labels=labels,flierprops=dict(markersize=2));ax[1].tick_params(axis='x',rotation=45);ax[1].set(xlabel='Factor budget / warm use',ylabel='Second emission (ms)',title='(b) All 80 calls per setting')
 causes=sorted(set(r['numerical']['dispatch_reason'] for r in i['rows']));counts=[sum(r['numerical']['dispatch_reason']==c for r in i['rows']) for c in causes];ax[2].barh([c.replace('_',' ') for c in causes],counts,color='#007C83');ax[2].set(xlabel='Calls (one cause each)',title='(c) Primary dispatcher outcome');ax[2].tick_params(axis='y',labelsize=6)
 figstyle.save(fig,P/'fig_q1_runtime');plt.close(fig)
if __name__=='__main__':main()
