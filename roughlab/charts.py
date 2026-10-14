import numpy as np
import plotly.graph_objects as go
from .models import FEATURES, FACTOR_NAMES, BOUNDS, pairwise_response, ishigami
from .pce import pairwise_projection_coefficients

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


def surface_grid(case="Roughness drag", resolution=40, axis0=None, axis1=None):
    if case == "Roughness drag":
        axis0, axis1 = (1 if axis0 is None else axis0), (2 if axis1 is None else axis1)
        bounds = BOUNDS
        anchor = np.zeros(5)
        names = FEATURES
        fun = lambda points:pairwise_response(points,axis0,axis1)
    else:
        axis0, axis1 = (0 if axis0 is None else axis0), (1 if axis1 is None else axis1)
        bounds = [[-np.pi,np.pi]]*3
        anchor = np.zeros(3)
        names, fun = ["x1","x2","x3"], ishigami
    u = np.linspace(*bounds[axis0],resolution)
    v = np.linspace(*bounds[axis1],resolution)
    xx, yy = np.meshgrid(u,v)
    points = np.tile(anchor,(resolution*resolution,1))
    points[:,axis0], points[:,axis1] = xx.ravel(),yy.ravel()
    return xx,yy,points,fun(points).reshape(xx.shape),names[axis0],names[axis1]


def response_surface(case="Roughness drag", axis0=None,axis1=None,selected_inputs=None,height=430):
    axis0=1 if axis0 is None else axis0
    axis1=2 if axis1 is None else axis1
    xx,yy,points,z,nx,ny = surface_grid(case,axis0=axis0,axis1=axis1)
    product=f"{FACTOR_NAMES[axis0]} × {FACTOR_NAMES[axis1]}" if case=="Roughness drag" else "f(x)"
    fig = go.Figure(go.Surface(x=xx,y=yy,z=z,colorscale=SCALE,showscale=False,
        contours=dict(z=dict(show=True,usecolormap=True,project_z=True)),
        hovertemplate=f"{nx}: %{{x:.3f}}<br>{ny}: %{{y:.3f}}<br>{product}: %{{z:.3f}}<extra></extra>"))
    if selected_inputs is not None and case=="Roughness drag":
        selected=np.asarray([selected_inputs],dtype=float)
        fig.add_trace(go.Scatter3d(x=selected[:,axis0],y=selected[:,axis1],z=pairwise_response(selected,axis0,axis1),
            mode="markers",marker=dict(size=5,color=ORANGE),name="Selected values",showlegend=False))
    fig.update_layout(scene=dict(xaxis_title=nx,yaxis_title=ny,zaxis_title=product,
        camera=dict(eye=dict(x=2.1,y=-2.2,z=1.6)),bgcolor="rgba(0,0,0,0)",
        xaxis=dict(backgroundcolor="#f2f4ee",gridcolor="#d5e1d9"),
        yaxis=dict(backgroundcolor="#f2f4ee",gridcolor="#d5e1d9"),
        zaxis=dict(backgroundcolor="#f2f4ee",gridcolor="#d5e1d9")))
    style(fig,height)
    fig.update_layout(margin=dict(l=30,r=24,t=20,b=35))
    return fig


def pce_surface_data(result,resolution=32):
    """Computed degree truncations, projected onto two subfunctions for roughness."""
    model = result["model"]
    xx,yy,points,truth,nx,ny = surface_grid(result["case"],resolution=resolution)
    old = model.coeff.copy()
    projected=pairwise_projection_coefficients(model) if result["case"]=="Roughness drag" else old
    surfaces=[]
    try:
        for degree in range(result["order"]+1):
            model.coeff = projected*(model.multi_index.sum(axis=1)<=degree)
            surfaces.append(model.evaluate(points).reshape(xx.shape))
    finally:
        model.coeff = old
    return xx,yy,truth,nx,ny,surfaces


def pce_animation(result):
    xx,yy,truth,nx,ny,surfaces=pce_surface_data(result)
    frames=[]
    for degree,zz in enumerate(surfaces):
        frames.append(go.Frame(name=str(degree),data=[go.Surface(x=xx,y=yy,z=zz,colorscale=SCALE,
            showscale=False,cmin=truth.min(),cmax=truth.max())],traces=[0],
            layout=go.Layout(title_text=f"Degree ≤ {degree} · polynomial approximation")))
    fig=go.Figure(data=[frames[-1].data[0],go.Surface(x=xx,y=yy,z=truth,opacity=.15,showscale=False,
        colorscale=[[0,ORANGE],[1,ORANGE]],name="Target",hoverinfo="skip")],frames=frames)
    zmin=min(min(np.min(f.data[0].z) for f in frames),truth.min())
    zmax=max(max(np.max(f.data[0].z) for f in frames),truth.max())
    fig.update_layout(scene=dict(xaxis_title=nx,yaxis_title=ny,zaxis_title="g(Ex) × f(Sk)" if result["case"]=="Roughness drag" else "f(x)",zaxis=dict(range=[zmin,zmax]),
        camera=dict(eye=dict(x=1.5,y=-1.7,z=1.2))),
        updatemenus=[dict(type="buttons",direction="left",x=0,y=-.1,buttons=[
            dict(label="▶ Play",method="animate",args=[None,dict(frame=dict(duration=850,redraw=True),transition=dict(duration=0),fromcurrent=True)]),
            dict(label="Ⅱ Pause",method="animate",args=[[None],dict(mode="immediate",frame=dict(duration=0,redraw=False))])])],
        sliders=[dict(active=result["order"],x=.25,len=.75,y=-.08,currentvalue=dict(prefix="Degree ≤ "),
            steps=[dict(label=str(d),method="animate",args=[[str(d)],dict(mode="immediate",frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for d in range(result["order"]+1)])])
    return style(fig,500,title="Pairwise subfunction product" if result["case"]=="Roughness drag" else "Ishigami polynomial approximation")
