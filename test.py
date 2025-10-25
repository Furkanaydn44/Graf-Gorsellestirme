import dash
from dash import dcc, html, Input, Output , State
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import joblib

istel_isim =["1. İster: A ile B yazarı arasındaki en kısa yolun bulunması ve grafiksel gösterimi",
"2. İster: A yazarı ve işbirliği yaptığı yazarlar için düğüm ağırlıklarına göre kuyruk oluşturma",
"3. İster: Kuyruktaki yazarlardan bir BST (Binary Search Tree) oluşturma",
"4. İster: A yazarı ve işbirlikçi yazarlar arasında kısa yolların hesaplanması",
"5. İster: A yazarının işbirliği yaptığı yazar sayısının hesaplanması",
"6. İster: En çok işbirliği yapan yazarın belirlenmesi",
"7. İster: Kullanıcıdan alınan yazar ID’sinden gidebileceği en uzun yolun bulunması"]


excel_file = "PROLAB 3 - GÜNCEL DATASET.xlsx"
data = pd.read_excel(excel_file)
joblib.dump(data, 'data.pkl')

nodes = {}
edge_weights = {}


for index, row in data.iterrows():
    author_name = row['author_name']
    orcid = row['orcid']
    co_authors = eval(row['coauthors'])
    paper_title = row['paper_title']

    if orcid not in nodes:
        nodes[orcid] = {
            "label": author_name,
            "papers": [paper_title]
        }
    else:
        nodes[orcid]["papers"].append(paper_title)

    co_authors = [coauthor.strip()for coauthor in co_authors if coauthor.strip() != author_name]
    for coauthor in co_authors:
        coauthor_orcid = coauthor.strip()
        if coauthor_orcid not in nodes:
            nodes[coauthor_orcid] = {
                "label": coauthor_orcid,
                "papers": [paper_title]
            }
        else:
            nodes[coauthor_orcid]["papers"].append(paper_title)

        edge = tuple(sorted((orcid, coauthor_orcid)))
        if edge not in edge_weights:
            edge_weights[edge] = 1
        else:
            edge_weights[edge] += 1



G = nx.Graph()


for edge, weight in edge_weights.items():
    G.add_edge(edge[0], edge[1], weight=weight)


for node_id, data in nodes.items():
    G.add_node(node_id,label=data['label'],papers=list(set(data['papers'])))

joblib.dump(G,'graph.pkl')

joblib.dump(nodes,'nodes.pkl')
joblib.dump(edge_weights,'edge_weights.pkl')

pos = nx.spring_layout(G, k=0.4, iterations=50)

joblib.dump(pos,'pos.pkl')


edge_traces = []
scale_factor = 0.04
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = edge[2]['weight']

    edge_traces.append(go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(width=1 + weight * scale_factor, color="gray" if weight == 1 else f"rgba({min(weight*50, 255)}, 50, 50, {0.5 + weight * 0.1})"),
        hoverinfo='text',
        mode='lines',
        text=f"Ağırlık: {weight}"
    ))

node_x = []
node_y = []
hover_text = []

MAX_PAPERS = 30

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

    papers = G.nodes[node]['papers']
    if len(papers) > MAX_PAPERS:
        papers_text = "<br>".join(papers[:MAX_PAPERS]) + "<br>...and more"
    else:
        papers_text = "<br>".join(papers)

    hover_text.append(
        f"Name: {G.nodes[node]['label']}<br>ORCID: {node}<br>Papers:<br>{papers_text}"
    )

average_degree = sum(dict(G.degree()).values()) / len(G.nodes())

node_sizes = []
for node in G.nodes():
    degree = len(list(G.neighbors(node)))
    if degree > average_degree * 1.2:
        node_sizes.append(12)
    else:
        node_sizes.append(8)


node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    hovertext=hover_text,
    marker=dict(
        showscale=True,
        symbol='circle',
        colorscale='Plasma',
        opacity=1.0,
        size=node_sizes,
        color=[len(list(G.neighbors(node))) for node in G.nodes()],
        colorbar=dict(
            thickness=20,
            title='Node Connections',
            titlefont=dict(size=14, color='black'),  # Başlık rengi ve boyutu
            tickfont=dict(size=12, color='black'),  # Sayıların rengi ve boyutu
            xanchor='left',
            titleside='right'
        )
    )
)


base_figure = go.Figure(
    data=edge_traces + [node_trace],
    layout=go.Layout(
        title=dict(
            text="Ağırlıklı Yazarlar Grafiği",
            font=dict(size=16),
        ),

        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    ),
)

author_degrees = {node: len(list(G.neighbors(node))) for node in G.nodes()}
max_collaboration_author = max(author_degrees, key=author_degrees.get)
maxc_author_name = nodes[max_collaboration_author]['label']
maxc_author_orcid = max_collaboration_author
maxc_author_collaboration_count = author_degrees[max_collaboration_author]

buttons = [f"İstel {i+1}" for i in range(7)]

app = dash.Dash(__name__)
app.layout = html.Div([

    html.Div(

        dcc.Graph(
            id="graph",
            figure=base_figure,
            style={
                "width": "65%",
                "height": "910px",
                "margin": "0 auto",
            }
        ),
        style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
            "marginRight": "20px"
        }
    ),
    html.Div([

        html.Div([

            html.Div(
                [
                    dcc.Input(
                        id="input-1-1",
                        type="text",
                        placeholder="1. Panel - Yazar Orcid giriniz (1)",
                        style={"width": "200px", "padding": "10px", "margin-top": "10px", "box-sizing": "border-box"}
                    ),
                    dcc.Input(
                        id="input-1-2",
                        type="text",
                        placeholder="1. Panel - Yazar Orcid giriniz (2)",
                        style={"width": "200px", "padding": "10px", "margin-top": "10px", "box-sizing": "border-box"}
                    ),
                    html.Button(
                        'İster-1',
                        id="submit-1",
                        n_clicks=0,
                        style={"width": "200px", "margin-top": "10px", "padding": "10px 20px"}
                    )
                ],
                style={
                    "display": "block",
                    "margin": "10px 0",
                    "backgroundColor": "#f1f1f1",
                    "padding": "20px",
                    "borderRadius": "5px"
                }
            ),

            html.Div(
                [
                    dcc.Input(
                        id="input-2",
                        type="text",
                        placeholder="2. Panel - Yazar Orcid giriniz.",
                        style={"width": "200px", "padding": "10px", "margin-top": "10px", "box-sizing": "border-box"}
                    ),
                    html.Button(
                        'İster-2',
                        id="submit-2",
                        n_clicks=0,
                        style={"width": "200px", "margin-top": "10px", "padding": "10px 20px"}
                    )
                ],
                style={
                    "display": "block",
                    "margin": "10px 0",
                    "backgroundColor": "#f1f1f1",
                    "padding": "20px",
                    "borderRadius": "5px"
                }
            ),

            html.Div(
                [
                    dcc.Input(
                        id="input-3",
                        type="text",
                        placeholder="3. Panel - Yazar Orcid giriniz.",
                        style={"width": "200px", "padding": "10px", "margin-top": "10px", "box-sizing": "border-box"}
                    ),
                    html.Button(
                        'İster-3',
                        id="submit-3",
                        n_clicks=0,
                        style={"width": "200px", "margin-top": "10px", "padding": "10px 20px"}
                    )
                ],
                style={
                    "display": "block",
                    "margin": "10px 0",
                    "backgroundColor": "#f1f1f1",
                    "padding": "20px",
                    "borderRadius": "5px"
                }
            ),

            html.Div(
                [
                    dcc.Input(
                        id="input-4",
                        type="text",
                        placeholder="4. Panel - Yazar Orcid giriniz.",
                        style={"width": "200px", "padding": "10px", "margin-top": "10px", "box-sizing": "border-box"}
                    ),
                    html.Button(
                        'İster-4',
                        id="submit-4",
                        n_clicks=0,
                        style={"width": "200px", "margin-top": "10px", "padding": "10px 20px"}
                    )
                ],
                style={
                    "display": "block",
                    "margin": "10px 0",
                    "backgroundColor": "#f1f1f1",
                    "padding": "20px",
                    "borderRadius": "5px"
                }
            ),

            html.Div(
                [
                    dcc.Input(
                        id="input-5",
                        type="text",
                        placeholder="5. Panel - Yazar Orcid giriniz.",
                        style={"width": "200px", "padding": "10px", "margin-top": "10px", "box-sizing": "border-box"}
                    ),
                    html.Button(
                        'İster-5',
                        id="submit-5",
                        n_clicks=0,
                        style={"width": "200px", "margin-top": "10px", "padding": "10px 20px"}
                    )
                ],
                style={
                    "display": "block",
                    "margin": "10px 0",
                    "backgroundColor": "#f1f1f1",
                    "padding": "20px",
                    "borderRadius": "5px"
                }
            ),

            html.Div(
                [
                    html.Button(
                        'İster-6',
                        id="submit-6",
                        n_clicks=0,
                        style={"width": "200px", "margin-top": "10px", "padding": "10px 20px"}
                    )
                ],
                style={
                    "display": "block",
                    "margin": "10px 0",
                    "backgroundColor": "#f1f1f1",
                    "padding": "20px",
                    "borderRadius": "5px"
                }
            ),

            html.Div(
                [
                    dcc.Input(
                        id="input-7",
                        type="text",
                        placeholder="7. Panel - Yazar Orcid giriniz.",
                        style={"width": "200px", "padding": "10px", "margin-top": "10px", "box-sizing": "border-box"}
                    ),
                    html.Button(
                        'İster-7',
                        id="submit-7",
                        n_clicks=0,
                        style={"width": "200px", "margin-top": "10px", "padding": "10px 20px"}
                    )
                ],
                style={
                    "display": "block",
                    "margin": "10px 0",
                    "backgroundColor": "#f1f1f1",
                    "padding": "20px",
                    "borderRadius": "5px"
                }
            ),

        ], style={
            "position": "fixed",
            "top": "12%",
            "right": "20px",
            "width": "230px",
            "backgroundColor": "#f8f9fa",
            "padding": "10px",
            "boxShadow": "0px 0px 0px rgba(0, 0, 0, 0.2)",
            "borderRadius": "10px",
            "height": "800px",
            "overflowY": "auto"
        }),


        html.Div([
            html.H4("Çıktı Paneli"),

            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=[
                    html.Div(id="panel-output"),
                ]
            ),
        ], style={
            "width": "300px",
            "height": "750px",
            "padding": "20px",
            "backgroundColor": "#C0C0C0",
            "boxShadow": "0px 4px 10px rgba(0, 0, 0, 0.1)",
            "borderRadius": "10px",
            "position": "absolute",
            "top": "5%",
            "left": "20px",
            "overflowY": "auto"
        })
    ])])

def find_coauthors(node_id):

    df = joblib.load('data.pkl')

    graph = joblib.load('graph.pkl')

    if node_id in graph:
        neighbors_set = set()

        neighbors_list = graph[node_id]

        for neighbor in neighbors_list:
            neighbor = neighbor.strip()
            if neighbor != node_id:
                neighbors_set.add(neighbor)

        if neighbors_set:
            return list(neighbors_set)
        else:
            return f"Düğüm: {node_id} için komşu yok"
    else:
        return f"Düğüm {node_id} grafta bulunamadı"


def find_author(orcidx):

    df = joblib.load('data.pkl')

    author_row = df[df['orcid'] == orcidx]

    if not author_row.empty:
        a_name = author_row['author_name'].values[0]
        return f"{a_name}"
    else:
        return  {orcidx}

def longest_path_dfs(graph, start):
    def dfs(node, visited, path):
        visited.add(node)
        path.append(node)
        is_dead_end = True

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                is_dead_end = False
                dfs(neighbor, visited, path)

        if is_dead_end:
            # En uzun yolu güncelle
            nonlocal longest_path
            if len(path) > len(longest_path):
                longest_path = list(path)

        path.pop()
        visited.remove(node)

    longest_path = []
    dfs(start, set(), [])
    return longest_path

def create_author_graph(start_node):      # 7. ister

    data = joblib.load('data.pkl')
    nodes = joblib.load('nodes.pkl')
    edge_weights = joblib.load('edge_weights.pkl')
    G = joblib.load('graph.pkl')
    pos = joblib.load('pos.pkl')

    if start_node in G.nodes():
        longest_path = longest_path_dfs(G, start_node)
    else:
        longest_path = []

    edge_traces = []
    node_x = []
    node_y = []
    hover_text = []

    highlight_edges = []
    if longest_path:
        highlight_edges = [(longest_path[i], longest_path[i + 1]) for i in range(len(longest_path) - 1)]

    scale_factor = 0.04
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        if (edge[0], edge[1]) in highlight_edges or (edge[1], edge[0]) in highlight_edges:
            color = "black"
            width = 5
        else:
            color = "gray" if weight == 1 else f"rgba({min(weight * 50, 255)}, 50, 50, {0.5 + weight * 0.1})"
            width = 1 + weight * scale_factor

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='text',
            mode='lines',
            text=f"Ağırlık: {weight}"
        ))

    MAX_PAPERS = 40
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        papers = G.nodes[node]['papers']
        if len(papers) > MAX_PAPERS:
            papers_text = "<br>".join(papers[:MAX_PAPERS]) + "<br>...and more"
        else:
            papers_text = "<br>".join(papers)

        hover_text.append(
            f"Name: {G.nodes[node]['label']}<br>ORCID: {node}<br>Papers:<br>{papers_text}"
        )

    average_degree = sum(dict(G.degree()).values()) / len(G.nodes())

    node_sizes = []
    for node in G.nodes():
        degree = len(list(G.neighbors(node)))
        if degree > average_degree * 1.5:
            node_sizes.append(12)
        else:
            node_sizes.append(8)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=hover_text,
        marker=dict(
            showscale=True,
            symbol='circle',
            colorscale='Plasma',
            opacity=1.0,
            size=node_sizes,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            colorbar=dict(
                thickness=20,
                title='Node Connections',
                titlefont=dict(size=14, color='black'),  # Başlık rengi ve boyutu
                tickfont=dict(size=12, color='black'),  # Sayıların rengi ve boyutu
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title="En Uzun Yolu Bulunmuş Ağırlıklı Yazarlar Grafiği",
                        titlefont=dict(size=20, color='black'),
                        paper_bgcolor='#2c3e50',  # Grafik arka plan rengi
                        plot_bgcolor='#d3d3d3',  # Çizim alanı arka plan rengi
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False,
                                   zeroline=False,
                                   tickfont=dict(size=12, color='black')),
                        yaxis=dict(showgrid=False,
                                   zeroline=False,
                                   tickfont=dict(size=12, color='black'))
                    ))
    fig.show()
    panel_content = html.Div([
        html.H3(f"{istel_isim[6]}"),
        html.P(f"En uzun yol: {longest_path}"),
        html.P(f"Ortalama Derece: {average_degree}"),
        html.P(f"Grafikteki Düğüm Sayısı: {len(G.nodes())}"),
        html.P(f"Grafikteki Kenar Sayısı: {len(G.edges())}")
    ])

    return panel_content


def two_author_distance_graph(A,B): #                 1.ister

    data = joblib.load('data.pkl')
    nodes = joblib.load('nodes.pkl')
    edge_weights = joblib.load('edge_weights.pkl')
    G = joblib.load('graph.pkl')
    pos = joblib.load('pos.pkl')

    if A not in G and B not in G:
        panel_content=html.Div([
            html.H3(f"{istel_isim[0]}"),
            html.P(f"ID ({A}) ve ID ({B}) bu grafta bulunmuyor")
        ])
        return panel_content
    elif A not in G:
        panel_content=html.Div([
            html.H3(f"{istel_isim[0]}"),
            html.P(f"ID ({A}) bu grafta bulunmuyor")
        ])
        return panel_content
    elif B not in G:
        panel_content=html.Div([
            html.H3(f"{istel_isim[0]}"),
            html.P(f"ID ({B}) bu grafta bulunmuyor")
        ])
        return panel_content

    if A==B:
        panel_content = html.Div([
            html.H3(f"{istel_isim[0]}"),
            html.P(f"İki ID de aynı. {A}-{B}")
        ])
        return panel_content

    def shortest_path_bfs(graph, start, goal):
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            current_node, path = queue.popleft()

            if current_node == goal:
                return path

            if current_node not in visited:
                visited.add(current_node)

                for neighbor in graph.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        return None


    shortest_path = shortest_path_bfs(G, A, B)

    edge_traces = []
    node_x = []
    node_y = []
    hover_text = []

    highlight_edges = []
    if shortest_path:
        highlight_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        color = "red" if (edge[0], edge[1]) in highlight_edges or (edge[1], edge[0]) in highlight_edges else "gray"
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=color),
            hoverinfo='none',
            mode='lines'
        ))


    edge_traces = []
    scale_factor = 0.04

    highlight_edges = []
    if shortest_path:
        highlight_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        if (edge[0], edge[1]) in highlight_edges or (edge[1], edge[0]) in highlight_edges:
            color = "black"
            width = 5
        else:
            color = "gray" if weight == 1 else f"rgba({min(weight * 50, 255)}, 50, 50, {0.5 + weight * 0.1})"
            width = 1 + weight * scale_factor

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='text',
            mode='lines',
            text=f"Ağırlık: {weight}"
        ))

    MAX_PAPERS = 40

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        papers = G.nodes[node]['papers']
        if len(papers) > MAX_PAPERS:
            papers_text = "<br>".join(papers[:MAX_PAPERS]) + "<br>...and more"
        else:
            papers_text = "<br>".join(papers)

        hover_text.append(
            f"Name: {G.nodes[node]['label']}<br>ORCID: {node}<br>Papers:<br>{papers_text}"
        )

    average_degree = sum(dict(G.degree()).values()) / len(G.nodes())

    node_sizes = []
    for node in G.nodes():
        degree = len(list(G.neighbors(node)))
        if degree > average_degree * 1.5:
            node_sizes.append(12)
        else:
            node_sizes.append(8)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=hover_text,
        marker=dict(
            showscale=True,
            symbol='circle',
            colorscale='Plasma',
            opacity=1.0,
            size=node_sizes,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            colorbar=dict(
                thickness=20,
                title='Node Connections',
                titlefont=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black'),
                xanchor='left',
                titleside='right'
            )
        )
    )

    joblib.dump(shortest_path, 'bst_list.pkl')
    print(shortest_path)

    if shortest_path:
        panel_content = html.Div([
            html.H3(f"{istel_isim[0]}"),
            html.P(f"En kısa yol üzerindeki yazarlar: {shortest_path}")
        ])
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(
                            title="Ağırlıklı Yazarlar Grafiği",
                            titlefont=dict(size=20, color='black'),
                            paper_bgcolor='#2c3e50',
                            plot_bgcolor='#d3d3d3',
                            showlegend=False,
                            hovermode='closest',
                            xaxis=dict(showgrid=False,
                                       zeroline=False,
                                       tickfont=dict(size=12, color='black')),
                            yaxis=dict(showgrid=False,
                                       zeroline=False,
                                       tickfont=dict(size=12, color='black'))
                        ))

        fig.show()
    else:
        panel_content = html.Div([
            html.H3(f"{istel_isim[0]}"),
            html.P(f"A ve B arasında ortak yol yok")
        ])
    return panel_content


def bst_visual(shortest_path):  #        3.ister

    class Node:
        def __init__(self, key):
            self.left = None
            self.right = None
            self.value = key
            self.height = 1

    class AVLTree:
        def __init__(self):
            self.root = None

        def insert(self, root, key):
            if not root:
                return Node(key)
            elif key < root.value:
                root.left = self.insert(root.left, key)
            else:
                root.right = self.insert(root.right, key)

            root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))

            balance = self.getBalance(root)

            if balance > 1 and key < root.left.value:
                return self.rightRotate(root)

            if balance < -1 and key > root.right.value:
                return self.leftRotate(root)

            if balance > 1 and key > root.left.value:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)

            if balance < -1 and key < root.right.value:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

            return root

        def leftRotate(self, z):
            y = z.right
            T2 = y.left

            y.left = z
            z.right = T2

            z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
            y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

            return y

        def rightRotate(self, z):
            y = z.left
            T3 = y.right

            y.right = z
            z.left = T3

            z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
            y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

            return y

        def getHeight(self, root):
            if not root:
                return 0
            return root.height

        def getBalance(self, root):
            if not root:
                return 0
            return self.getHeight(root.left) - self.getHeight(root.right)

        def visualize(self, root):
            G = nx.DiGraph()
            self._add_edges(root, G)

            pos = self._get_node_positions(root)

            nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold")
            plt.show()
            plt.close()

        def _add_edges(self, root, G):
            if root is None:
                return

            G.add_node(root.value)
            if root.left:
                G.add_edge(root.value, root.left.value)
                self._add_edges(root.left, G)
            if root.right:
                G.add_edge(root.value, root.right.value)
                self._add_edges(root.right, G)

        def _get_node_positions(self, root, pos=None, level=0, width=0.1, x=0.5):
            if root is None:
                return pos
            if pos is None:
                pos = {}
            pos[root.value] = (x, 1 - level * 0.2)
            if root.left:
                l = x - width / 2
                self._get_node_positions(root.left, pos, level + 1, width / 2, l)
            if root.right:
                r = x + width / 2
                self._get_node_positions(root.right, pos, level + 1, width / 2, r)
            return pos

    avl_tree = AVLTree()
    root = None
    for orcid in shortest_path:
        root = avl_tree.insert(root, orcid)

    avl_tree.visualize(root)

def tail_graph_visual(selected_orcid):       #  4. ister
    data = joblib.load('data.pkl')
    nodes = joblib.load('nodes.pkl')
    edge_weights = joblib.load('edge_weights.pkl')
    G = joblib.load('graph.pkl')
    pos = joblib.load('pos.pkl')

    edge_traces = []
    scale_factor = 0.04
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1 + weight * scale_factor,
                      color="gray" if weight == 1 else f"rgba({min(weight * 50, 255)}, 50, 50, {0.5 + weight * 0.1})"),
            hoverinfo='text',
            mode='lines',
            text=f"Ağırlık: {weight}"
        ))


    node_x = []
    node_y = []
    hover_text = []

    MAX_PAPERS = 60

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        papers = G.nodes[node]['papers']
        if len(papers) > MAX_PAPERS:
            papers_text = "<br>".join(papers[:MAX_PAPERS]) + "<br>...and more"
        else:
            papers_text = "<br>".join(papers)

        hover_text.append(
            f"Name: {G.nodes[node]['label']}<br>ORCID: {node}<br>Papers:<br>{papers_text}"
        )

    average_degree = sum(dict(G.degree()).values()) / len(G.nodes())

    node_sizes = []
    for node in G.nodes():
        degree = len(list(G.neighbors(node)))
        if degree > average_degree * 1.2:
            node_sizes.append(12)
        else:
            node_sizes.append(8)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=hover_text,
        marker=dict(
            showscale=True,
            symbol='circle',
            colorscale='Plasma',
            opacity=1.0,
            size=node_sizes,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            colorbar=dict(
                thickness=20,
                title='Node Connections',
                titlefont=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black'),
                xanchor='left',
                titleside='right'
            )
        )
    )

    def shortest_path_bfs(graph, start, goal):
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            current_node, path = queue.popleft()

            if current_node == goal:
                return path

            if current_node not in visited:
                visited.add(current_node)

                for neighbor in graph.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        return None

    def find_neighbors(graph, node, degree=3):
        visited = set()
        current_level = {node}
        all_neighbors = set(current_level)

        for _ in range(degree):
            next_level = set()
            for current_node in current_level:
                if current_node not in visited:
                    next_level.update(graph.neighbors(current_node))
            visited.update(current_level)
            current_level = next_level - visited
            all_neighbors.update(current_level)

        return all_neighbors

    if selected_orcid not in G.nodes:
        panel_content = html.Div([
            html.H3(f"{istel_isim[3]}"),
            html.P(f"ID Bu grafikte mevcut değil.")
        ])
    else:
        subgraph_nodes = find_neighbors(G, selected_orcid, degree=2)

        subgraph_edges = []
        for edge in G.edges(data=True):
            if edge[0] in subgraph_nodes and edge[1] in subgraph_nodes:
                subgraph_edges.append(edge)

        H = nx.Graph()
        H.add_nodes_from(subgraph_nodes)
        H.add_edges_from((edge[0], edge[1], edge[2]) for edge in subgraph_edges)

        panel_content = html.Div([
            html.H3(f"{istel_isim[3]}"),

        ])

        shortest_paths = {}
        for node in H.nodes():
            if node != selected_orcid:
                path = shortest_path_bfs(H, selected_orcid, node)
                if path:
                    shortest_paths[node] = len(path) - 1

        shortest_paths_str = []
        for node, dist in shortest_paths.items():
            panel_content.children.append(
                html.P(f"ORCID: {node}, En Kısa Yol: {dist}")
            )

        sub_pos = nx.spring_layout(H, k=0.4, iterations=50)

        edge_traces = []
        for edge in H.edges(data=True):
            x0, y0 = sub_pos[edge[0]]
            x1, y1 = sub_pos[edge[1]]
            weight = edge[2]['weight']

            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1 + weight * scale_factor, color="gray"),
                hoverinfo='text',
                mode='lines',
                text=f"Ağırlık: {weight}"
            ))

        node_x = []
        node_y = []
        hover_text = []
        node_sizes = []

        for node in H.nodes():
            x, y = sub_pos[node]
            node_x.append(x)
            node_y.append(y)

            papers = G.nodes[node]['papers']
            papers_text = "<br>".join(papers[:MAX_PAPERS]) + ("<br>...and more" if len(papers) > MAX_PAPERS else "")

            hover_text.append(
                f"Name: {G.nodes[node]['label']}<br>ORCID: {node}<br>Papers:<br>{papers_text}"
            )
            degree = len(list(H.neighbors(node)))
            node_sizes.append(12 if degree > average_degree * 1.2 else 8)

        # Düğüm izleri
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                showscale=True,
                symbol='circle',
                colorscale='Plasma',
                opacity=1.0,
                size=node_sizes,
                color=[len(list(H.neighbors(node))) for node in H.nodes()],
                colorbar=dict(
                    thickness=20,
                    title='Node Connections',
                    titlefont=dict(size=14, color='black'),
                    tickfont=dict(size=12, color='black'),
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        # Alt ağ grafiği
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(
                            title=f"{selected_orcid} için Alt Ağ",
                            titlefont=dict(size=20, color='black'),
                            paper_bgcolor='#2c3e50',
                            plot_bgcolor='#d3d3d3',
                            showlegend=False,
                            hovermode='closest',
                            xaxis=dict(showgrid=False,
                                       zeroline=False,
                                       tickfont=dict(size=12, color='black')),
                            yaxis=dict(showgrid=False,
                                       zeroline=False,
                                       tickfont=dict(size=12, color='black'))
                        ))
        fig.show()
    if selected_orcid  in G.nodes:
        panel_content.children.append(html.P(f"Alt ağ düğüm sayısı:, {len(H.nodes())}"))
        panel_content.children.append(html.P(f"Alt ağ kenar sayısı:, {len(H.edges())}"))



    return panel_content

def node_weight_tail(author_id):       # 2. ister


    data = joblib.load('data.pkl')
    nodes = joblib.load('nodes.pkl')
    edge_weights = joblib.load('edge_weights.pkl')
    G = joblib.load('graph.pkl')
    pos = joblib.load('pos.pkl')

    edge_traces = []
    scale_factor = 0.04
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1 + weight * scale_factor,
                      color="gray" if weight == 1 else f"rgba({min(weight * 50, 255)}, 50, 50, {0.5 + weight * 0.1})"),
            hoverinfo='text',
            mode='lines',
            text=f"Ağırlık: {weight}"
        ))

    author_id = author_id.strip()
    if author_id in G.nodes:
        neighbors = list(G.neighbors(author_id))
        coauthors_with_weights = []

        for neighbor in neighbors:
            coauthor_paper_count = len(G.nodes[neighbor]['papers'])
            coauthors_with_weights.append((neighbor, coauthor_paper_count))

        sorted_coauthors = sorted(coauthors_with_weights, key=lambda x: x[1], reverse=True)
        queue = deque(sorted_coauthors)

        panel_content = html.Div([
            html.H3(f"{istel_isim[1]}"),
            html.P(f"\n{G.nodes[author_id]['label']} adlı yazar ile işbirliği yapan yazarlar (düğüm ağırlıklarına göre sıralı):")
        ])

        for coauthor, weight in queue:
            panel_content.children.append(html.P(f"- {G.nodes[coauthor]['label']} (ORCID: {coauthor}) - Makale Sayısı: {weight}"))
    else:
        panel_content = html.Div([
            html.H3(f"{istel_isim[1]}"),
            html.P(f"Girilen Orcid geçerli değil veya Orcidin düğümü geçerli değil.")
        ])

    return panel_content

@app.callback(
    [Output(f"panel-output","children"),
     Output("graph","figure")],
    [Input(f"submit-{i+1}","n_clicks")for i in range(7)]  ,
    [State( f"input-1-1","value")]+
    [State( f"input-1-2","value")]+
    [State( f"input-{i+2}","value") for i in range (4)]+
    [State(f"input-7","value")]
)
def functions(*args):

    ctx = dash.callback_context

    if not ctx.triggered:
        return ["Bir butona tıklanmadı", dash.no_update]

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_index = int(button_id.split("-")[1])



    panel_content = dash.no_update
    graph_figure = dash.no_update

    if button_index == 1:
        input_1_1_value = args[7]
        input_1_2_value = args[8]
        if input_1_1_value and input_1_2_value:
            panel_content = two_author_distance_graph(input_1_1_value,input_1_2_value)

        return [panel_content, dash.no_update]
    elif button_index == 2:
        input_2_value = args[9]
        if input_2_value:
            panel_content = node_weight_tail(input_2_value)

        return [panel_content , dash.no_update]

    elif button_index == 3:
        bst_list = joblib.load('bst_list.pkl')
        input_3_value = args[10]
        print(bst_list)
        if input_3_value:
            if input_3_value in bst_list:
                bst_list.remove(input_3_value)
                print(bst_list)
                bst_visual(bst_list)
                panel_content = html.Div([
                    html.H3(f"{istel_isim[2]}"),
                    html.P(f"Çıkarılan orcid: {input_3_value}")
                ])
            else:
                panel_content = html.Div([
                    html.H3("Hata"),
                    html.P(f"Verilen orcid {input_3_value} listede bulunamadı!")
                ])

        return [panel_content, dash.no_update]

    elif button_index == 4:
        input_4_value = args[11]
        if input_4_value:
            panel_content = tail_graph_visual(input_4_value)
        return [panel_content,dash.no_update]

    elif button_index == 5:
        input_5_value = args[12]
        if input_5_value:

            coauthors = find_coauthors(input_5_value)
            author = find_author(input_5_value)
            if isinstance(coauthors, list):
                coauthor_count = len(coauthors)
                panel_content = html.Div([
                    html.H3(f"{istel_isim[4]}"),
                    html.P(f"Author: {author}"),
                    html.P(f"Orcid: {input_5_value}"),
                    html.P(f"Co-author sayısı : {coauthor_count}"),
                    html.P("Co-authorlar:"),
                    html.Ul([html.Li(coauthor) for coauthor in coauthors])
                ])
            else:
                panel_content = html.Div([html.P(coauthors)])
        else:
            panel_content = html.Div([html.P("Lütfen geçerli bir ORCID girin.")])
        return [panel_content, dash.no_update]

    elif button_index == 6:
        panel_content = html.Div([
            html.H3(f"{istel_isim[5]}"),
            html.P(f"En fazla işbirliği yapan yazar : {maxc_author_name}"),
            html.P(f"Orcid : {maxc_author_orcid}"),
            html.P(f"İşbirliği Sayısı: {maxc_author_collaboration_count}")
        ])
        return [panel_content, dash.no_update]


    elif button_index == 7:

        input_7_value = args[13]
        panel_content = create_author_graph(input_7_value)

        return [panel_content,dash.no_update]

    return [f"Buton {button_index} tıklandı"]

if __name__ == "__main__":
    app.run_server(debug=True)


