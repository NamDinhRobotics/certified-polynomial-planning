"""Regenerate the recovery research addendum's table and scientific figure."""
import figstyle
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def render(write=False):
    r=json.loads((ROOT/'artifacts/revision_green_recovery.json').read_text())['rows']
    q=lambda k:np.percentile([x[k] for x in r],[50,90,99,100])
    claims=dict(count=len(r),methods={k:sum(x['method']==k for x in r) for k in ('plain','escape','green')},
        old={k:dict(accepted=sum(x['old'][k]['accepted'] for x in r),strict=sum(x['old'][k]['strict'] for x in r)) for k in ('plain','onecut','guide_qp')},
        relative_bound_percent=(100*q('certified_relative_suboptimality')).tolist(),accumulated_recovery_ms=q('combined_recovery_ms').tolist(),
        repair_only_median_bound_percent=100*float(np.median([x['certified_relative_suboptimality'] for x in r if not x['plain_strict']])),
        stage1_worst_cost=max(x['cost'] for x in json.loads((ROOT/'artifacts/revision_escape_fold.json').read_text())['rows'] if x['ok']),
        final_worst_cost=max(x['cost'] for x in r),
        note='Retrospective adaptive replay; accumulated timing adds separately measured stages, not a new live closed-loop benchmark.')

    text=r'''\begin{table}[ht]
    \centering
    \caption{Exact physical recovery on the same 1,000 F2 geometries. Historical acceptance and the strict zero-tolerance replay are different predicates.}
    \label{tab:recovery}
    \begin{tabular}{lrr}
    \toprule Method & Historical acceptance & Strict $p>0$ \\
    \midrule
    '''
    for k,name in [('plain','Original SDP projection'),('onecut','Original one-cut'),('guide_qp','Original guide QP')]:
     a=claims['old'][k];text+=f"{name} & {a['accepted']} & {a['strict']} \\\\\n"
    text+=r'''Current solver projection & -- & 463 \\
    Projection + gap-mode fold & -- & 981 \\
    Projection + gap/Green modes & -- & 1000 \\
    \bottomrule
    \end{tabular}
    \end{table}
    '''
    files={'claims_recovery.json':json.dumps(claims,indent=2),'table_recovery.tex':text}
    if write:
        for name,body in files.items():(HERE/name).write_text(body)
    return files


def plot():
    r=json.loads((ROOT/'artifacts/revision_green_recovery.json').read_text())['rows']
    figstyle.use()
    fig,ax=plt.subplots(1,3,figsize=(figstyle.TEXT_W,2.7),layout='constrained')
    labels=['Projection','+ Gap','+ Green'];counts=[463,981,1000]
    ax[0].bar(labels,counts,color=['#8595a8','#478ab9','#18876c']);ax[0].set_ylim(0,1120);ax[0].set_ylabel('Certified physical curves / 1,000');ax[0].set_title('(a) Recovery availability')
    for i,c in enumerate(counts):ax[0].text(i,c+15,str(c),ha='center')
    for rows,label,color in [(r,'All 1,000','#18876c'),([x for x in r if not x['plain_strict']],'537 repaired','#a85931')]:
     x=np.sort([100*z['certified_relative_suboptimality'] for z in rows]);ax[1].plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=color,lw=2,ls='-' if label=='All 1,000' else '--')
    ax[1].set_xlabel('Suboptimality upper bound (%)');ax[1].set_ylabel('Fraction of cases');ax[1].set_title('(b) Energy guarantee');ax[1].legend(fontsize=9);ax[1].grid(alpha=.15)
    a=[x['gap_mode_cost'] for x in r if x['gap_mode_cost'] is not None];b=[x['cost'] for x in r]
    for x,label,col in [(a,'Gap mode (981)','#478ab9'),(b,'Combined (1,000)','#18876c')]:
     x=np.sort(x);ax[2].plot(np.arange(1,len(x)+1)/len(x),x,label=label,color=col,lw=2,ls='--' if label.startswith('Gap') else '-')
    ax[2].set_yscale('log');ax[2].set_xlabel('Fraction of answered cases');ax[2].set_ylabel('Normalized derivative energy');ax[2].set_title('(c) Retained cost tail');ax[2].legend(fontsize=8);ax[2].grid(alpha=.15)
    figstyle.save(fig,HERE/'fig_recovery_novelty')
    plt.close(fig)


def main():
    render(write=True);plot()

if __name__=='__main__':main()
