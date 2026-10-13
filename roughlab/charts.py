import numpy as np
import plotly.graph_objects as go
from .models import FEATURES, BOUNDS, drag_model, ishigami

TEAL, ORANGE, NAVY = "#087f8c", "#e58b53", "#152c39"
COLORS = [TEAL, ORANGE, "#7e78a8", "#718e57", "#b75b70"]
SCALE = [[0, "#e8f0df"], [.35, "#8bc3ba"], [.7, "#248d94"], [1, "#174959"]]


def style(fig, height=420, title=None):
    fig.update_layout(height=height, margin=dict(l=30,r=24,t=45 if title else 20,b=35),
        title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif",size=13,color=NAVY),
        legend=dict(orientation="h",y=1.12,x=0), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor="#e1e6df",zeroline=False)
    fig.update_yaxes(gridcolor="#e1e6df",zeroline=False)
    return fig


def parity(df, pred_column="prediction", target_column="target", height=430):
    fig = go.Figure()
    for i, (label, group) in enumerate(df.groupby("label",sort=True)):
        fig.add_trace(go.Scatter(x=group[target_column], y=group[pred_column], mode="markers",name=label,
            text=group["Case name"], marker=dict(color=COLORS[i%5],size=9,
                symbol="diamond" if label == "DNSLC" else "circle",line=dict(width=1,color="white")),
            hovertemplate="%{text}<br>Reference: %{x:.3f}<br>Prediction: %{y:.3f}<extra>%{fullData.name}</extra>"))
    low = min(df[target_column].min(),df[pred_column].min())
    high = max(df[target_column].max(),df[pred_column].max())
    fig.add_trace(go.Scatter(x=[low,high],y=[low,high],mode="lines",name="Exact agreement",line=dict(color=NAVY,width=1,dash="dash")))
    fig.update_layout(xaxis_title="Reference ks / krms",yaxis_title="Predicted ks / krms")
    return style(fig,height)


def pareto(template, free):
    fig = go.Figure()
    for data, label, color in [(template,"With template",TEAL),(free,"Without template",ORANGE)]:
        fig.add_trace(go.Scatter(x=data.Complexity,y=data.Loss,mode="lines+markers",name=label,
            line=dict(color=color,width=2),marker=dict(size=5)))
    fig.update_layout(xaxis_title="Expression complexity",yaxis_title="Training MAE",yaxis_type="log")
    return style(fig,380)


def surface_grid(case="Roughness drag", resolution=40, axis0=None, axis1=None, anchor=None):
    if case == "Roughness drag":
        axis0, axis1 = (1 if axis0 is None else axis0), (2 if axis1 is None else axis1)
        bounds = BOUNDS
        anchor = np.array([.634,.517,.367,.067,.159]) if anchor is None else np.asarray(anchor)
        names, fun = FEATURES, drag_model
    else:
        axis0, axis1 = (0 if axis0 is None else axis0), (1 if axis1 is None else axis1)
        bounds = [[-np.pi,np.pi]]*3
        anchor = np.zeros(3) if anchor is None else np.asarray(anchor)
        names, fun = ["x1","x2","x3"], ishigami
    u = np.linspace(*bounds[axis0],resolution)
    v = np.linspace(*bounds[axis1],resolution)
    xx, yy = np.meshgrid(u,v)
    points = np.tile(anchor,(resolution*resolution,1))
    points[:,axis0], points[:,axis1] = xx.ravel(),yy.ravel()
    return xx,yy,points,fun(points).reshape(xx.shape),names[axis0],names[axis1]


def response_surface(case="Roughness drag", axis0=None,axis1=None,anchor=None,height=430):
    xx,yy,points,z,nx,ny = surface_grid(case,axis0=axis0,axis1=axis1,anchor=anchor)
    fig = go.Figure(go.Surface(x=xx,y=yy,z=z,colorscale=SCALE,showscale=False,
        contours=dict(z=dict(show=True,usecolormap=True,project_z=True)),
        hovertemplate=f"{nx}: %{{x:.3f}}<br>{ny}: %{{y:.3f}}<br>Response: %{{z:.3f}}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title=nx,yaxis_title=ny,zaxis_title="ks / krms" if case=="Roughness drag" else "f(x)",
        camera=dict(eye=dict(x=1.5,y=-1.6,z=1.15)),bgcolor="rgba(0,0,0,0)",
        xaxis=dict(backgroundcolor="#f2f4ee",gridcolor="#d5e1d9"),
        yaxis=dict(backgroundcolor="#f2f4ee",gridcolor="#d5e1d9"),
        zaxis=dict(backgroundcolor="#f2f4ee",gridcolor="#d5e1d9")))
    return style(fig,height)


def pce_animation(result):
    """Plotly frames really evaluate each degree truncation of fitted coefficients."""
    model = result["model"]
    xx,yy,points,truth,nx,ny = surface_grid(result["case"],resolution=32)
    old = model.coeff.copy()
    frames = []
    try:
        for degree in range(result["order"]+1):
            model.coeff = old*(model.multi_index.sum(axis=1)<=degree)
            zz = model.evaluate(points).reshape(xx.shape)
            frames.append(go.Frame(name=str(degree),data=[go.Surface(x=xx,y=yy,z=zz,colorscale=SCALE,
                showscale=False,cmin=truth.min(),cmax=truth.max())],traces=[0],
                layout=go.Layout(title_text=f"Degree ≤ {degree} · fitted expansion truncation")))
    finally:
        model.coeff = old
    fig=go.Figure(data=[frames[-1].data[0],go.Surface(x=xx,y=yy,z=truth,opacity=.15,showscale=False,
        colorscale=[[0,ORANGE],[1,ORANGE]],name="Target",hoverinfo="skip")],frames=frames)
    zmin=min(min(np.min(f.data[0].z) for f in frames),truth.min())
    zmax=max(max(np.max(f.data[0].z) for f in frames),truth.max())
    fig.update_layout(scene=dict(xaxis_title=nx,yaxis_title=ny,zaxis_title="Response",zaxis=dict(range=[zmin,zmax]),
        camera=dict(eye=dict(x=1.5,y=-1.7,z=1.2))),
        updatemenus=[dict(type="buttons",direction="left",x=0,y=-.1,buttons=[
            dict(label="▶ Play",method="animate",args=[None,dict(frame=dict(duration=850,redraw=True),transition=dict(duration=0),fromcurrent=True)]),
            dict(label="Ⅱ Pause",method="animate",args=[[None],dict(mode="immediate",frame=dict(duration=0,redraw=False))])])],
        sliders=[dict(active=result["order"],x=.25,len=.75,y=-.08,currentvalue=dict(prefix="Degree ≤ "),
            steps=[dict(label=str(d),method="animate",args=[[str(d)],dict(mode="immediate",frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for d in range(result["order"]+1)])])
    return style(fig,500)
