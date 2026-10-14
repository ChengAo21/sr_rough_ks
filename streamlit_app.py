"""Roughness Lab: interactive companion to Cheng et al., JFE (2026)."""
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from roughlab.models import (ROOT, FEATURES, FACTOR_NAMES, LABELS, BOUNDS, PAPER_DISTRIBUTIONS,
    PAPER_PARAMETERS, load_data, published_predictions, drag_model, factors, metrics)
from roughlab.pce import fit_pce, coefficient_table
from roughlab.charts import (TEAL, ORANGE, NAVY, COLORS, style, parity, pareto,
    response_surface, pce_animation)
from roughlab.sr_config import UNARY, BINARY, NESTING, PAPER_TEMPLATE, preset_config, normalize_config

st.set_page_config(page_title="Roughness Lab | JFE 2026",page_icon="〰",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Libre+Caslon+Display&display=swap');
html, body, [class*="css"], .stApp {font-family:'DM Sans',Arial,sans-serif;}
.block-container{max-width:1400px;padding-top:2.4rem;padding-bottom:3rem;}
[data-testid="stSidebar"]{background:#eeefe8;border-right:1px solid #dce2d8;}
[data-testid="stSidebar"] .block-container{padding-top:2rem;}
h1,h2,h3{color:#152c39;letter-spacing:-.04em;}
h1{font-family:'Libre Caslon Display',Georgia,serif!important;font-size:3.7rem!important;font-weight:400!important;line-height:1.05!important;}
h2{font-family:'Libre Caslon Display',Georgia,serif!important;font-size:2.45rem!important;font-weight:400!important;}
h3{font-size:1.2rem!important;letter-spacing:-.02em;}
p,li{line-height:1.65;} .eyebrow{font-size:.72rem;letter-spacing:.18em;font-weight:700;color:#087f8c;text-transform:uppercase;margin-bottom:1rem;}
.brand{font-family:Georgia,serif;font-size:1.8rem;color:#152c39;letter-spacing:-.07em;}
.brand span{color:#087f8c;} .muted{color:#6e7e80;font-size:.87rem;}
.lede{font-size:1.04rem;color:#53686c;max-width:660px;line-height:1.85;margin-top:.9rem;}
.rule{height:1px;background:#dbe1d9;margin:1.2rem 0;}
.note{border-left:3px solid #087f8c;padding:10px 18px;background:#edf2e9;border-radius:0 8px 8px 0;color:#496468;font-size:.85rem;}
.tag{display:inline-block;border:1px solid #cbd9cd;border-radius:30px;padding:5px 11px;color:#4b6868;font-size:.73rem;margin:0 5px 8px 0;}
.step{font-size:.8rem;color:#087f8c;font-weight:700;letter-spacing:.1em;}
.paper{background:#152c39;color:#f7f7f2;padding:28px;border-radius:14px;margin:12px 0;}
.paper a{color:#9cd8cb;} .paper h3{color:white!important;font-family:Georgia,serif!important;font-size:1.4rem!important;line-height:1.5;}
[data-testid="stMetric"]{padding:15px 18px;border-top:2px solid #a7c8bf;background:#f0f2eb;border-radius:0 0 7px 7px;}
[data-testid="stMetricValue"]{font-family:Georgia,serif;font-size:2.1rem;}
[data-testid="stMetricLabel"]{color:#5a7375;}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:#d9e1d7!important;border-radius:12px!important;}
.stButton>button{border-radius:6px;font-weight:600;} .stDownloadButton>button{border-radius:6px;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:9px;}
footer{visibility:hidden;} @media(max-width:800px){h1{font-size:2.7rem!important;}.block-container{padding-top:1.5rem;}}
</style>""",unsafe_allow_html=True)


@st.cache_data
def data(): return load_data()


@st.cache_data
def predictions(): return published_predictions()


def eyebrow(text): st.markdown(f'<div class="eyebrow">{html.escape(text)}</div>',unsafe_allow_html=True)


def caption(text): st.markdown(f'<div class="muted">{html.escape(text)}</div>',unsafe_allow_html=True)


def chart(fig, key=None): st.plotly_chart(fig,use_container_width=True,key=key,config={"displaylogo":False,"scrollZoom":False})


def section(number,title,description=None):
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    eyebrow(f"{number} / {title}")
    if description: st.write(description)


with st.sidebar:
    st.markdown('<div class="brand">roughness<span> / lab</span></div>',unsafe_allow_html=True)
    caption("INTERACTIVE RESEARCH COMPANION")
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    page = st.radio("Explore",["Overview","Symbolic Lab","PCE Studio","Model Explorer","Research & Cite"],label_visibility="collapsed",key="page")
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    st.markdown("**Cheng et al. · JFE 2026**")
    caption("Feature importance-informed symbolic drag modelling.")
    st.link_button("Read the paper ↗","https://doi.org/10.1115/1.4070838",use_container_width=True)
    st.link_button("GitHub repository ↗","https://github.com/ChengAo21/sr_rough_ks",use_container_width=True)
    st.markdown(" ")
    caption("Original research code and reproducible experiments. Published, archived and newly computed results are labelled separately.")
    st.caption("Interface & tooling assisted by OpenAI Codex.")


def overview():
    a,b = st.columns([1.08,1],gap="large")
    with a:
        eyebrow("JOURNAL OF FLUIDS ENGINEERING / 2026")
        st.markdown("<h1>Rough-surface<br>drag modelling</h1>",unsafe_allow_html=True)
        st.markdown('<div class="lede">Explore the equivalent sandgrain roughness model, examine interactions between geometric statistics, and run symbolic regression and PCE sensitivity analysis.</div>',unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown('<span class="tag">Feature importance</span><span class="tag">Symbolic regression</span><span class="tag">Polynomial chaos</span>',unsafe_allow_html=True)
        st.markdown("**Rough-surface drag modelling · From data discovery to physical insight**")
        caption("Drag to rotate g(Ex) × f(Sk), the product of two learned subfunctions. All other subfunctions have unit contribution.")
    with b:
        chart(response_surface(height=420),"hero-surface")
    cols=st.columns(4)
    for col,label,value,helptext in zip(cols,["Training surfaces","Unseen test surfaces","Mean relative error","Model agreement R²"],["96","8","9.60%","0.94"],["Training set used in the paper","Unseen extrapolation test set used in the paper","Paper, Table 4: all 104 cases","Paper, Table 4: all 104 cases"]):
        col.metric(label,value,help=helptext)
    caption("Summary metrics are from Table 4 of the paper. Predictions below are evaluated from the repository data and analytical model.")
    section("01","Feature importance-informed symbolic regression")
    st.image(str(ROOT/"assets/workflow.svg"),use_container_width=True)
    cols=st.columns(3,gap="large")
    for col,num,title,body in zip(cols,["01","02","03"],["Find what matters","Search with structure","Quantify the uncertainty"],["Reduce the geometric statistics to five influential inputs while preserving interpretability.","Constrain symbolic search with a product of univariate factors to obtain a portable analytical expression.","Use the original PolyChaos implementation to fit a surrogate and compute Sobol sensitivity indices."]):
        with col:
            st.markdown(f'<div class="step">{num}</div>',unsafe_allow_html=True)
            st.subheader(title)
            st.write(body)
    section("02","Evidence, recomputed","Recomputed from the 104 rough-surface cases in the repository. Diamonds mark DNSLC cases excluded from training.")
    a,b=st.columns([1.6,1],gap="large")
    df=predictions()
    with a: chart(parity(df),"overview-parity")
    with b:
        st.subheader("Across five data families")
        rows=[]
        for name,group in df.groupby("label"):
            rows.append({"Dataset":name,"N":len(group),"MARE (%)":metrics(group.target,group.prediction)["MARE (%)"]})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,column_config={"MARE (%)":st.column_config.NumberColumn(format="%.2f")})
        st.markdown('<div class="note">Each point evaluates the public expression. Live searches in this interface are new experiments; their performance is separate from the published model.</div>',unsafe_allow_html=True)
        st.download_button("Download predictions · CSV",df.to_csv(index=False),"published_model_predictions.csv","text/csv")
    section("03","Related publication")
    st.markdown('<div class="paper"><div class="eyebrow" style="color:#91c9bd">RELATED PUBLICATION / JCP 2026</div><h3>A physics-embedded symbolic wall model for large-eddy simulation of separated flows</h3><p>A related publication on interpretable, physics-embedded symbolic wall modelling for separated-flow large-eddy simulation.</p><a href="https://doi.org/10.1016/j.jcp.2026.115311" target="_blank">Read the related paper ↗</a></div>',unsafe_allow_html=True)


def start_search(config):
    job=ROOT/".runtime/jobs"/uuid.uuid4().hex
    job.mkdir(parents=True)
    (job/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    env=os.environ.copy()
    env["PYTHONUNBUFFERED"]="1"
    flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    with (job/"worker.log").open("w",encoding="utf-8") as log:
        proc=subprocess.Popen([sys.executable,"-u",str(ROOT/"roughlab/sr_worker.py"),str(job)],cwd=ROOT,
            env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,start_new_session=(os.name!="nt"))
    st.session_state.sr_job={"path":str(job),"proc":proc,"start":time.time(),"config":config}
    st.session_state.pop("sr_result",None)


def cancel_job(job_info):
    proc=job_info["proc"]
    if proc.poll() is None:
        if os.name=="nt":
            subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(proc.pid,signal.SIGTERM)
    job_info["cancelled"]=True


@st.fragment(run_every=2)
def search_monitor():
    job_info=st.session_state.get("sr_job")
    if not job_info: return
    job,proc=Path(job_info["path"]),job_info["proc"]
    elapsed=time.time()-job_info["start"]
    if (job/"result.json").exists():
        result=json.loads((job/"result.json").read_text(encoding="utf-8"))
        st.session_state.sr_result=result
        if not job_info.get("done"):
            job_info["done"]=True
            st.rerun()
        st.success(f"Completed · {result['engine']} · {result['seconds']:.1f} s")
    elif job_info.get("cancelled"):
        st.warning("This search has been cancelled.")
    elif (job/"error.json").exists() or proc.poll() is not None:
        st.error("The search did not complete. Check the process log; PySR initializes its Julia packages on the first run. Python GP is available for immediate exploration.")
    else:
        budget=job_info["config"]["timeout"]
        if budget and elapsed > budget+600:
            cancel_job(job_info)
            st.error("The combined initialization and search time limit was reached. The process has been stopped.")
        else:
            progress={}
            try: progress=json.loads((job/"progress.json").read_text(encoding="utf-8"))
            except (FileNotFoundError,json.JSONDecodeError): pass
            fraction=progress.get("generation",0)/max(1,progress.get("total",1))
            st.progress(fraction,text=f"{job_info['config']['engine']} · {progress.get('stage','starting')} · {elapsed:.0f} s")
            if "loss" in progress: st.caption(f"Generation {progress['generation']} / {progress['total']} · Best training MAE {progress['loss']:.5f}")
            if st.button("Cancel search",key="cancel-search"):
                cancel_job(job_info)
                job_info["done"]=True
                st.warning("This search has been cancelled.")
    if (job/"worker.log").exists():
        with st.expander("Live process log",expanded=False):
            st.code((job/"worker.log").read_text(encoding="utf-8",errors="replace")[-14000:],language="text")


def symbolic_lab():
    eyebrow("EXPERIMENT 01 / SYMBOLIC REGRESSION")
    st.title("Symbolic regression of rough-surface drag")
    st.write("Configure a symbolic-regression search on 96 training cases. The eight DNSLC cases remain reserved for independent testing.")
    with st.expander("How the two search engines differ"):
        st.write("**PySR** runs the research configuration from `results/SR_train.py`, including the expression template, nested-operator constraints, weighted complexity and L1 objective.\n\n**Python GP** is a lightweight alternative for immediate exploration. It enforces the selected nesting limits and constant-complexity weight, with protected numerical operators. PySR population counts, cycles and execution mode apply to the PySR backend.")
    a,b=st.columns([1.5,1.8],gap="large")
    with a:
        with st.container(border=True):
            st.subheader("Search controls")
            engine=st.selectbox("Search engine",["PySR","Python GP"],index=0 if importlib.util.find_spec("pysr") else 1,key="sr-engine")
            profile=st.selectbox("Configuration preset",["Quick run","Paper settings"],key="sr-profile")
            defaults=preset_config(profile)
            st.caption("Paper settings are read from the original training script. Quick run reduces the search budget and uses serial execution.")
            with st.form(f"sr-controls-{profile}"):
                template=st.checkbox("Product template · Separated variables",value=True)
                tabs=st.tabs(["Search budget","Expression constraints"])
                with tabs[0]:
                    iterations=st.number_input("Generations / iterations",5,100000,defaults["iterations"],5)
                    population=st.number_input("Population size",10,1000,defaults["population"],10)
                    populations=st.number_input("PySR populations",1,1000,defaults["populations"],disabled=engine!="PySR")
                    cycles=st.number_input("PySR cycles per iteration",10,10000,defaults["cycles"],10,disabled=engine!="PySR")
                    seed=st.number_input("Random seed",0,100000,42)
                    timeout=st.number_input("Search time limit (s)",0,86400,defaults["timeout"],10,help="0 disables the time limit, as in the original script.")
                    parallelism=st.selectbox("PySR execution mode",["serial","multithreading"],index=0 if defaults["parallelism"]=="serial" else 1,disabled=engine!="PySR",help="The paper uses multithreading. Serial execution with a fixed seed is reproducible.")
                with tabs[1]:
                    unary=st.multiselect("Unary operators",UNARY,default=defaults["unary"])
                    binary=st.multiselect("Binary operators",BINARY,default=defaults["binary"])
                    maxsize=st.number_input("Maximum weighted complexity",10,200,defaults["maxsize"],help="PySR maxsize. The original script sets this to 60.")
                    constant_cost=st.number_input("Constant complexity weight",1,10,defaults["constant_complexity"],help="complexity_of_constants = 3 in the original script. Variables and operators each have weight 1.")
                    st.markdown("**Nested-operator constraints**")
                    st.caption("Rows are outer operators; columns are inner operators. 0 forbids nesting, 1 permits it once per branch, and −1 removes that limit.")
                    inner_ops=list(next(iter(NESTING.values())))
                    matrix=pd.DataFrame([{ "Outer":outer,**{inner:limits.get(inner,-1) for inner in inner_ops}} for outer,limits in NESTING.items()])
                    edited=st.data_editor(matrix,hide_index=True,use_container_width=True,disabled=["Outer"],
                        column_config={"Outer":st.column_config.TextColumn(width="small"),**{inner:st.column_config.NumberColumn(min_value=-1,max_value=10,step=1,required=True,width="small") for inner in inner_ops}},key=f"sr-nesting-{profile}")
                    nested={row["Outer"]:{inner:int(row[inner]) for inner in inner_ops} for row in edited.to_dict("records")}
                    st.caption("Limits apply to the selected operators. The default table is the original nested_constraints configuration.")
                with st.expander("Objective and selection"):
                    st.caption("L1DistLoss() · precision 64 · PySR model selection: best · variable/operator complexity: 1")
                    st.code(PAPER_TEMPLATE["combine"],language="text")
                    parsimony=st.number_input("Python GP complexity penalty",0.,1.,.003,.001,format="%.3f",disabled=engine!="Python GP")
                submitted=st.form_submit_button("Run symbolic search →",type="primary",use_container_width=True)
            if submitted:
                existing=st.session_state.get("sr_job")
                if existing and existing["proc"].poll() is None:
                    st.warning("A search is already running. Wait for it to finish or cancel it before starting another.")
                elif not unary and not binary:
                    st.error("Select at least one operator.")
                elif engine=="PySR" and importlib.util.find_spec("pysr") is None:
                    st.error("PySR is not installed in this environment. Run python -m pip install -r requirements-pysr.txt, then restart the app.")
                else:
                    config=dict(profile=profile,engine=engine,template=template,iterations=iterations,population=population,
                        maxsize=maxsize,unary=unary,binary=binary,seed=int(seed),populations=int(populations),
                        cycles=int(cycles),timeout=int(timeout),parsimony=parsimony,parallelism=parallelism,
                        constant_complexity=int(constant_cost),nested_constraints=nested)
                    try: start_search(normalize_config(config))
                    except ValueError as exc: st.error(str(exc))
            st.caption("Paper settings: 1500 iterations / 90 populations / population size 80 / 1200 cycles / maxsize 60 / constant complexity 3.")
    with b:
        st.subheader("Search workspace")
        saved_run=ROOT/"docs/runs/pysr_seed42.json"
        if saved_run.exists() and not st.session_state.get("sr_job"):
            if st.button("View verified PySR demo"):
                saved=json.loads(saved_run.read_text(encoding="utf-8"))
                saved["saved_demo"]=True
                st.session_state.sr_result=saved
        search_monitor()
        result=st.session_state.get("sr_result")
        if not result:
            st.markdown('<div class="note">Select Run symbolic search to start a new experiment. The chart below shows archived complexity–loss fronts from the repository, separate from live search results.</div>',unsafe_allow_html=True)
            template_df=pd.read_csv(ROOT/"results/csvFiles/hall_of_fame_withTemplate.csv")
            free_df=pd.read_csv(ROOT/"results/csvFiles/hall_of_fame_noTemplate.csv")
            chart(pareto(template_df,free_df),"archive-pareto")
        else:
            origin="SAVED VERIFIED RUN" if result.get("saved_demo") else "LIVE RESULT"
            st.caption(f"{origin} · {result['engine']} · {result['config'].get('profile','Saved configuration')} · seed {result['config']['seed']} · {result['seconds']:.2f} s")
            c1,c2,c3=st.columns(3)
            c1.metric("Training MAE",f"{result['train_metrics']['MAE']:.4f}")
            c2.metric("Held-out MARE",f"{result['test_metrics']['MARE (%)']:.2f}%")
            c3.metric("Held-out R²",f"{result['test_metrics']['R²']:.3f}")
            st.code(result["equation"],language="text",wrap_lines=True)
            if result["config"].get("template") and result["config"]["engine"]=="PySR":
                st.caption("Template: f(Po) × g(Ex) × h(Sk) × ksi(Sx) × eta(kavg). #1 denotes the input variable of each individual subfunction.")
            tabs=st.tabs(["Prediction","Search trajectory","Equations & export"])
            with tabs[0]:
                df=data().copy()
                df["prediction"]=result["predictions"]
                chart(parity(df),"live-sr-parity")
            with tabs[1]:
                fig=go.Figure()
                if result["history"]:
                    fig.add_trace(go.Scatter(x=[r["generation"] for r in result["history"]],y=[r["loss"] for r in result["history"]],mode="lines+markers",line=dict(color=TEAL)))
                    fig.update_layout(xaxis_title="Generation",yaxis_title="Best training MAE")
                else:
                    fig.add_trace(go.Scatter(x=[r["complexity"] for r in result["frontier"]],y=[r["loss"] for r in result["frontier"]],mode="lines+markers",line=dict(color=TEAL)))
                    fig.update_layout(xaxis_title="Expression complexity",yaxis_title="Training MAE")
                chart(style(fig,350),"live-sr-history")
            with tabs[2]:
                st.dataframe(pd.DataFrame(result["frontier"]),hide_index=True,use_container_width=True)
                with st.expander("Effective search configuration"):
                    st.json(result["config"])
                st.download_button("Download reproducible run · JSON",json.dumps(result,indent=2,ensure_ascii=False),"symbolic_run.json","application/json")
                df=data().copy(); df["prediction"]=result["predictions"]
                st.download_button("Download predictions · CSV",df.to_csv(index=False),"symbolic_predictions.csv","text/csv")
                if result.get("saved_demo"):
                    log=(ROOT/"docs/runs/pysr_process.txt").read_text(encoding="utf-8")
                    st.download_button("Download saved process log",log,"pysr_process.txt","text/plain")
            caption("Short searches illustrate the discovery process. Their performance depends on the budget, seed and operators and is reported separately from the extensively optimized published model.")


def pce_studio():
    eyebrow("EXPERIMENT 02 / POLYNOMIAL CHAOS EXPANSION")
    st.title("Watch uncertainty take shape.")
    st.write("Fit a surrogate with the original PolyChaos implementation, compute statistical moments and Sobol indices, and animate the polynomial terms as they assemble the response surface.")
    controls,workspace=st.columns([1,1.85],gap="large")
    with controls:
        with st.container(border=True):
            st.subheader("Build a surrogate")
            case=st.selectbox("Case study",["Roughness drag","Ishigami benchmark"],key="pce-case")
            presets=["Observed uniform ranges","Paper · Table 5 mixed"] if case=="Roughness drag" else ["Independent U(−π, π)"]
            preset=st.selectbox("Input distribution preset",presets,key="pce-preset")
            dim=5 if case=="Roughness drag" else 3
            paper=preset=="Paper · Table 5 mixed"
            order=st.slider("Total polynomial degree",1,6,6 if paper or dim==3 else 4,key=f"degree-{case}-{paper}")
            nt=math.comb(dim+order,order)
            samples=st.number_input("LHS training samples",nt,6000,max(nt*3,2500 if paper else 800),100,key=f"samples-{case}-{order}-{paper}")
            seed=st.number_input("Random seed · PCE",0,100000,42,key="pce-seed")
            names=FEATURES if dim==5 else ["x1","x2","x3"]
            distrib=PAPER_DISTRIBUTIONS.copy() if paper else ["U"]*dim
            params=[p.copy() for p in (PAPER_PARAMETERS if paper else BOUNDS)] if dim==5 else [[-np.pi,np.pi] for _ in range(3)]
            st.caption("U: a = lower bound, b = upper bound. N: a = mean, b = standard deviation.")
            settings=pd.DataFrame({"Feature":names,"Distribution":distrib,"a":np.array(params)[:,0],"b":np.array(params)[:,1]})
            edited=st.data_editor(settings,hide_index=True,use_container_width=True,disabled=["Feature"],
                column_config={"Distribution":st.column_config.SelectboxColumn(options=["U","N"],required=True),"a":st.column_config.NumberColumn(format="%.3f",required=True),"b":st.column_config.NumberColumn(format="%.3f",required=True)},key=f"dist-{case}-{preset}")
            st.caption(f"{dim} inputs · {nt} basis terms · {int(samples)/nt:.1f} samples per term")
            if samples<2*nt: st.warning("The sample count is less than twice the number of basis terms. Consider increasing it for a more stable fit.")
            if st.button("Fit PCE & analyze →",type="primary",use_container_width=True):
                try:
                    with st.status("Building the polynomial surrogate…",expanded=True) as status:
                        progress=st.progress(0.)
                        def update(fraction,text): progress.progress(fraction,text=text)
                        result=fit_pce(case,order,int(samples),int(seed),edited.Distribution.tolist(),edited[["a","b"]].to_numpy().tolist(),update)
                        st.session_state.pce_result=result
                        status.update(label=f"Surrogate ready · {result['seconds']:.2f} s",state="complete",expanded=False)
                except Exception as exc: st.error(str(exc))
            st.caption("Sobol decomposition assumes independent inputs. The mixed preset matches Table 5 of the paper. Normal distributions are untruncated and may extend beyond the training-data range.")
    with workspace:
        result=st.session_state.get("pce_result")
        if result is None:
            with st.spinner("Preparing a freshly computed example…"):
                result=fit_pce("Roughness drag",4,800,42,["U"]*5,BOUNDS)
                st.session_state.pce_result=result
        st.caption(f"DISPLAYED RUN · {result['case']} · degree {result['order']} · {result['samples']} LHS · seed {result['seed']} · {result['seconds']:.2f} s")
        current=(case,order,int(samples),int(seed),edited.Distribution.tolist(),edited[["a","b"]].to_numpy().tolist())
        previous=(result["case"],result["order"],result["samples"],result["seed"],result["distrib"],result["parameters"])
        if current!=previous: st.info("The controls have changed. Select Fit PCE to update the results; the completed experiment identified above is still displayed.")
        c1,c2,c3=st.columns(3)
        c1.metric("Mean μ",f"{result['model'].mu:.3f}")
        c2.metric("Standard deviation σ",f"{result['model'].sigma:.3f}")
        c3.metric("Validation relative L₂",f"{100*result['relative_l2']:.2f}%",help="Computed on 1200 independent LHS validation points, separate from the training error.")
        tabs=st.tabs(["Animated expansion","Sensitivity","Validation","Coefficients & log"])
        with tabs[0]:
            fig=pce_animation(result)
            # An embedded Plotly document preserves native Play/Pause frame controls.
            components.html(fig.to_html(full_html=False,include_plotlyjs=True,config={"displaylogo":False}),height=555,scrolling=False)
            caption("Select Play to retain progressively higher polynomial degrees. For roughness, the plot compares g(Ex) × f(Sk) with its projected PCE approximation; all other subfunctions have unit contribution. The orange surface is the analytical target. The Ishigami plot shows an x1–x2 slice at x3=0.")
        with tabs[1]:
            names=FEATURES if result["model"].dim==5 else ["x1","x2","x3"]
            fig=go.Figure([go.Bar(x=names,y=result["first"],name="First-order Si",marker_color=TEAL),go.Bar(x=names,y=result["total"],name="Total-order STi",marker_color=ORANGE)])
            fig.update_layout(barmode="group",yaxis_title="Fraction of output variance")
            chart(style(fig,350),"sobol")
            st.metric("Variance from interactions",f"{result['interaction']*100:.2f}%")
            if result["case"]=="Ishigami benchmark":
                st.caption("Standard Ishigami reference values: S1≈0.3139, S2≈0.4424, S3=0; ST1≈0.5576, ST2≈0.4424, ST3≈0.2437. Increase the polynomial degree to inspect convergence.")
            caption("These indices describe model sensitivity under the selected independent input distributions; they do not establish causal contributions across rough surfaces.")
        with tabs[2]:
            lo,hi=float(result["truth"].min()),float(result["truth"].max())
            fig=go.Figure([go.Scatter(x=result["truth"],y=result["prediction"],mode="markers",marker=dict(color=TEAL,size=4,opacity=.5),name="Independent LHS"),go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",line=dict(color=NAVY,dash="dash"),name="Exact")])
            fig.update_layout(xaxis_title="Target model",yaxis_title="PCE surrogate")
            chart(style(fig,330),"pce-parity")
            h=pd.DataFrame(result["history"])
            fig=go.Figure(go.Scatter(x=h.degree,y=h.relative_l2,mode="lines+markers",line=dict(color=ORANGE)))
            fig.update_layout(xaxis_title="Retained degree of fitted expansion",yaxis_title="Relative L₂ error",yaxis_type="log")
            chart(style(fig,250),"pce-convergence")
        with tabs[3]:
            coeff=coefficient_table(result)
            st.dataframe(coeff,hide_index=True,use_container_width=True,height=220)
            st.code(result["log"],language="text")
            st.download_button("Download PCE coefficients · CSV",coeff.to_csv(index=False),"pce_coefficients.csv","text/csv")
            summary={key:result[key] for key in ["case","order","samples","seed","distrib","parameters","relative_l2","first","total","interaction","seconds","history"]}
            st.download_button("Download settings & sensitivity · JSON",json.dumps(summary,indent=2),"pce_run.json","application/json")


def explorer():
    eyebrow("MODEL EXPLORER / ROUGH-SURFACE DRAG MODEL")
    st.title("Subfunctions and pairwise interactions")
    st.latex(r"\frac{k_s}{k_{\mathrm{rms}}}=h(P_o)\,g(E_x)\,f(S_k)\,\psi(k_{\mathrm{avg}})\,\phi(S_x)")
    left,right=st.columns([1,2],gap="large")
    with left:
        st.subheader("Geometric statistics")
        anchor=[]
        defaults=[.634,.517,.367,.067,.159]
        for label,bound,value in zip(LABELS,BOUNDS,defaults): anchor.append(st.slider(label,float(bound[0]),float(bound[1]),value,.001))
        st.metric("Predicted ks / krms",f"{drag_model(np.array([anchor]))[0]:.3f}")
        st.caption("The sliders vary geometric statistics independently to probe the analytical model. A selected combination may not correspond to a physically realizable rough surface.")
        axis0=st.selectbox("Surface x-axis",FEATURES,index=1)
        axis1=st.selectbox("Surface y-axis",[x for x in FEATURES if x!=axis0],index=1)
    with right:
        first,second=FEATURES.index(axis0),FEATURES.index(axis1)
        chart(response_surface(axis0=first,axis1=second,selected_inputs=anchor,height=490),"explorer-surface")
        st.caption(f"{FACTOR_NAMES[first]} × {FACTOR_NAMES[second]}. All other subfunctions have unit contribution. The orange marker shows the selected values of these two inputs.")
    section("01","Univariate subfunctions","Each curve is evaluated from the analytical model. The dot marks the current slider setting.")
    cols=st.columns(5)
    for j,col in enumerate(cols):
        points=np.tile(anchor,(160,1)); points[:,j]=np.linspace(*BOUNDS[j],160)
        fig=go.Figure([go.Scatter(x=points[:,j],y=factors(points)[:,j],mode="lines",line=dict(color=COLORS[j],width=3)),go.Scatter(x=[anchor[j]],y=[factors(np.array([anchor]))[0,j]],mode="markers",marker=dict(color=NAVY,size=9))])
        fig.update_layout(showlegend=False,xaxis_title=FEATURES[j],yaxis_title="Factor")
        with col: chart(style(fig,230),f"factor-{j}")
    with st.expander("Exact public expression"):
        for formula in [r"h(P_o)=(\cos P_o-P_o e^{P_o})^2+8.331",r"g(E_x)=\tanh((E_x-0.084)/0.294)",r"f(S_k)=\tanh S_k+S_k/\tanh S_k",r"\psi(k)=2k-0.692-\tanh(-0.024/k^2)e^k",r"\phi(S_x)=e^{2S_x}"]:
            st.latex(formula)
        st.caption("Continuous limits are used at Sk=0, where f(0)=1, and kavg=0, where ψ(0)=0.308.")
    section("02","Inspect the source data")
    df=predictions()
    labels=st.multiselect("Data families",sorted(df.label.unique()),default=sorted(df.label.unique()))
    st.dataframe(df.loc[df.label.isin(labels),["Case name","label","split"]+FEATURES+["target","prediction","ARE (%)"]],hide_index=True,use_container_width=True)


def research():
    eyebrow("RESEARCH / CITATION & REPRODUCIBILITY")
    st.title("Keep the provenance visible.")
    st.subheader("Primary publication")
    st.write("**A Drag Model for Rough Surfaces Learned Using Feature Importance-Informed Symbolic Regression**")
    st.write("Ao Cheng, Zhideng Zhou, Xiaolei Yang, Guowei He · Journal of Fluids Engineering · 148(4), 041302 · April 2026")
    st.link_button("DOI · 10.1115/1.4070838","https://doi.org/10.1115/1.4070838")
    bib=(ROOT/"CITATION.bib").read_text(encoding="utf-8")
    st.code(bib,language="bibtex")
    st.download_button("Download verified BibTeX",bib,"CITATION.bib","text/plain")
    st.subheader("Related publication")
    st.write("**A physics-embedded symbolic wall model for large-eddy simulation of separated flows**")
    st.write("Ao Cheng, Xin-Lei Zhang, Zhideng Zhou, Xiaolei Yang, Guowei He · Journal of Computational Physics · 567, 115311 · 2026")
    st.link_button("DOI · 10.1016/j.jcp.2026.115311","https://doi.org/10.1016/j.jcp.2026.115311")
    st.subheader("Learning resources")
    st.markdown("[PySR examples](https://pysr.ai/examples) · [PySR API](https://ai.damtp.cam.ac.uk/pysr/api) · [UQpy PCE gallery](https://uqpyproject.readthedocs.io/en/latest/auto_examples/surrogates/pce/) · [Chaospy polynomial expansions](https://chaospy.readthedocs.io/en/master/user_guide/polynomial/introduction.html)")
    st.write("These resources provide learning examples. This app calls the repository's own PolyChaos implementation; it does not reuse the linked projects' implementations or figures.")
    st.subheader("Reproducibility notes")
    st.markdown("- The target is the normalized equivalent sandgrain roughness height, `ks/krms`.\n- All non-DNSLC cases form the training set. DNSLC cases remain held out for testing.\n- Pairwise response surfaces show the product of two subfunctions, with all other subfunctions contributing one.\n- The Paper settings preset reads the template, operators, nesting limits and complexity settings from the original training script.")
    st.subheader("Contributors & acknowledgments")
    st.write("Research and original code: Ao Cheng and coauthors. The interactive interface, code integration, documentation and verification were assisted by **OpenAI Codex**.")


if page.startswith("Overview"): overview()
elif page.startswith("Symbolic"): symbolic_lab()
elif page.startswith("PCE"): pce_studio()
elif page.startswith("Model"): explorer()
else: research()

st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
caption("ROUGHNESS / LAB     ·     Cheng et al., JFE 2026     ·     Built around the original research code")
