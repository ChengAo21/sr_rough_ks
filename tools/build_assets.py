"""Redraw every showcase figure from the public data and model. No PDF crops."""
from pathlib import Path
import io
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from roughlab.models import published_predictions,metrics,BOUNDS,PAPER_DISTRIBUTIONS,PAPER_PARAMETERS
from roughlab.pce import fit_pce
from roughlab.charts import pce_surface_data


def main():
    assets=ROOT/"assets"; assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"figure.facecolor":"#f7f7f2","axes.facecolor":"#f7f7f2","axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":"#152c39","text.color":"#152c39","xtick.color":"#53696c","ytick.color":"#53696c","savefig.facecolor":"#f7f7f2"})
    df=published_predictions()
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained",gridspec_kw={"width_ratios":[1,1.08]})
    colors=["#087f8c","#e58b53","#7e78a8","#718e57","#b75b70"]
    rows=[]
    for color,(label,g) in zip(colors,df.groupby("label")):
        axes[0].scatter(g.target,g.prediction,label=f"{label} (n={len(g)})",s=34,c=color,marker="D" if label=="DNSLC" else "o",edgecolors="white",linewidth=.5)
        rows.append({"label":label,"n":len(g),**metrics(g.target,g.prediction)})
    axes[0].plot([0,25],[0,25],"--",color="#617475",lw=1)
    axes[0].set(xlabel="Reference ks / krms",ylabel="Predicted ks / krms",title="PUBLIC MODEL / RECOMPUTED")
    axes[0].legend(frameon=False,fontsize=8)
    axes[0].grid(alpha=.12)
    axes[1].barh([r["label"] for r in rows],[r["MARE (%)"] for r in rows],color=colors,height=.55)
    for i,r in enumerate(rows): axes[1].text(r["MARE (%)"]+.2,i,f"{r['MARE (%)']:.2f}%",va="center")
    axes[1].set(xlabel="Mean absolute relative error (%)",xlim=[0,19],title="GENERALIZATION / FIVE DATA FAMILIES")
    axes[1].grid(axis="x",alpha=.12)
    fig.savefig(assets/"model_results.png",dpi=180); plt.close(fig)
    pce=fit_pce("Roughness drag",6,2500,42,PAPER_DISTRIBUTIONS,PAPER_PARAMETERS)
    plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans"})
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    axes[0].scatter(pce["truth"],pce["prediction"],s=7,alpha=.35,c="#087f8c")
    lo,hi=np.min(pce["truth"]),np.max(pce["truth"])
    axes[0].plot([lo,hi],[lo,hi],"--",color="#617475",lw=1)
    axes[0].set(xlabel="Symbolic model response",ylabel="PCE response",title="INDEPENDENT VALIDATION / 1,200 LHS POINTS")
    x=np.arange(5)
    axes[1].bar(x-.18,pce["first"],width=.34,color="#087f8c",label="First order")
    axes[1].bar(x+.18,pce["total"],width=.34,color="#e58b53",label="Total order")
    axes[1].set(xticks=x,xticklabels=["Po","Ex","Sk","kavg","Sx"],ylabel="Variance fraction",title="SOBOL DECOMPOSITION / TABLE 5 INPUTS")
    axes[1].legend(frameon=False)
    fig.savefig(assets/"pce_results.png",dpi=180);plt.close(fig)
    xx,yy,truth,nx,ny,surfaces=pce_surface_data(pce,resolution=44)
    frames=[]
    zmin,zmax=float(truth.min()),float(truth.max())
    for degree,zz in enumerate(surfaces):
        fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
        for ax,z,title in [(axes[0],truth,"Subfunction product g(Ex) x f(Sk)"),(axes[1],zz,f"Projected PCE / degree <= {degree}")]:
            im=ax.pcolormesh(xx,yy,z,vmin=zmin,vmax=zmax,cmap="YlGnBu",shading="auto")
            ax.set(xlabel="Effective slope Ex",ylabel="Skewness Sk",title=title)
        fig.colorbar(im,ax=axes,label="g(Ex) x f(Sk)",shrink=.8)
        fig.suptitle("A polynomial surrogate, assembled degree by degree",fontsize=14)
        buf=io.BytesIO();fig.savefig(buf,format="png",dpi=100);plt.close(fig);buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    frames[0].save(assets/"pce_expansion.gif",save_all=True,append_images=frames[1:]+[frames[-1]]*2,duration=850,loop=0)
    report={"model_metrics":metrics(df.target,df.prediction),"groups":rows,
        "pce":{k:pce[k] for k in ["case","order","samples","seed","distrib","parameters","relative_l2","first","total","seconds"]},
        "animation":{"target":"g(Ex) * f(Sk)","other_subfunctions":"unit contribution", "surrogate":"Marginal projection of the full-model PCE, divided by the other factors' mean product."}}
    (ROOT/"docs/verification-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
