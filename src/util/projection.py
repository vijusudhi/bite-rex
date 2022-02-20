from umap import UMAP
import pandas as pd
import plotly.express as px
import textwrap
    
def get_projection_2d(vectors):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(vectors)
    return proj_2d


def plot_proj(docs, proj, colors=None, plot_3d=False):
    df = pd.DataFrame(docs)
    df.columns = ['text']
    
    scatter_fn = px.scatter_3d if plot_3d else px.scatter
    
    if plot_3d:
        fig = px.scatter_3d(proj, x=0, y=1, z=2,
                            hover_name=df['text'].apply(
                                lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))
                                 ),
                            width=800, height=1500
                             )
    else:
        fig = px.scatter(proj, x=0, y=1,
                            hover_name=df['text'].apply(
                                lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))
                                 )
                             )
        
    fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')
    
    if not colors:
        fig.update_traces(hoverlabel=dict(align="left"))
    else:
        fig.update_traces(hoverlabel=dict(align="left"), marker_color=colors)
    
    return fig


def get_projection_2d(vectors):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(vectors)
    return proj_2d


def plot_umap_2d(docs, proj_2d, colors=None, sizes=None):
    df = pd.DataFrame(docs)
    df.columns = ['text']
    
    fig_2d = px.scatter(
                        proj_2d, 
                        x=0, 
                        y=1,
                        size=sizes,
                        hover_name=df['text'].apply(
                            lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))
                            )
                        )


    fig_2d.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')
    # fig_2d.update_xaxes(visible=False)
    # fig_2d.update_yaxes(visible=False)
    fig_2d.update_layout(
        # title="Plot Title",
        xaxis_title="x",
        yaxis_title="y",
    )

    if not colors:
        fig_2d.update_traces(hoverlabel=dict(align="left"))
    else:
        fig_2d.update_traces(hoverlabel=dict(align="left"), marker_color=colors)
        
    
    return fig_2d