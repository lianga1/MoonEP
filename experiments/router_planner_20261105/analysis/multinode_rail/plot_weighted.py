"""Plot weight-inclusive replay results with visible measurement boundaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/multinode_rail'


def main():
    old=json.loads((OUT/'weighted_training/summary.json').read_text())['rows']
    new=json.loads((OUT/'weighted_main/summary.json').read_text())['rows']
    lookup={(r['scenario'],r['method']):r for r in old+new}
    scenarios=['fanout_skew_1p5','fanout_skew_2p0','trace_lift_layer00','trace_lift_layer07']
    labels=['Synthetic rail 1.5','Synthetic rail 2.0','Trace-derived L00','Trace-derived L07']
    methods=['seed','rail_b8_r1','weight_aware_simple']
    fig,axes=plt.subplots(1,2,figsize=(14.2,5.8),gridspec_kw={'width_ratios':[1.65,1]})
    ax=axes[0]
    colors=['#3676a8','#bcc7d0','#d99240','#8e5fa9']
    for j,m in enumerate(methods):
        rows=[lookup[s,m] for s in scenarios]
        values=np.array([[r['weight_us'],r['training'][0]['token_training_segment_us_per_microbatch'],
                         r['gradient_us'],r['training'][0]['incremental_planner_us_per_microbatch']] for r in rows])/1000
        x=np.arange(4)*4+j
        bottom=np.zeros(4)
        for k,part in enumerate(['Weight refresh','Activation + GEMM','Replica gradient','Planner allowance']):
            ax.bar(x,values[:,k],bottom=bottom,color=colors[k],width=.78,label=part if j==0 else None)
            bottom+=values[:,k]
        for xx,total in zip(x,bottom):
            ax.text(xx,total+1.3,f'{total:.1f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(np.arange(4)*4+1,labels,fontsize=9)
    ax.set_ylabel('Modeled MoE training segment (ms)')
    ax.set_ylim(0,184)
    ax.set_title('One microbatch per weight update',loc='left',fontsize=12,fontweight='bold')
    ax.text(.02,.96,'Within each group: Seed / Previous B8 / New',transform=ax.transAxes,fontsize=9,va='top')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.16),ncols=2,frameon=False,fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axes[1]
    fields=['aggregate_rdma_bytes','aggregate_max_nic_port_bytes','aggregate_nvlink_bytes']
    metric_labels=['Total RDMA','Busiest NIC port','Total NVLink']
    for j,s in enumerate(scenarios[-2:]):
        a,b=lookup[s,'seed'],lookup[s,'weight_aware_simple']
        pct=np.array([100*(b[f]/a[f]-1) for f in fields])
        x=np.arange(3)+(j-.5)*.34
        ax.bar(x,pct,width=.32,color=['#3676a8','#d99240'][j],label=labels[j+2])
        for xx,pct_i in zip(x,pct):
            ax.text(xx,pct_i+(.4+j*.55 if pct_i>=0 else -.4-j*.55),f'{pct_i:+.2f}%',ha='center',
                    va='bottom' if pct_i>=0 else 'top',fontsize=8)
    ax.axhline(0,color='#555',linewidth=.8)
    ax.set_xticks(np.arange(3),metric_labels,fontsize=9)
    ax.set_ylabel('Change vs seed (%)')
    ax.set_title('All payloads: weights + gradients + tokens',loc='left',fontsize=11,fontweight='bold')
    ax.set_ylim(-12,3);ax.legend(frameon=False,loc='lower right',fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for a in axes:
        a.spines[['top','right']].set_visible(False)
    fig.suptitle('Weight cost reverses the planner decision',x=.055,y=.98,ha='left',fontsize=17,fontweight='bold')
    fig.text(.055,.015,'CPU model, not measured GPU latency. 4 x 8 GPUs; 128K tokens; H=7168, I=2048; BF16 weights/gradients.\n'
             'No overlap; backward transport assumed equal to forward. New planner allowance: 128 us, or 2 us for early exit. Common seed planner omitted.',fontsize=9,color='#555')
    fig.subplots_adjust(left=.065,right=.98,top=.86,bottom=.25,wspace=.32)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'weight_aware_comparison.{ext}',dpi=170,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
