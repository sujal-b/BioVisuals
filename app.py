import os
import plotly.graph_objects as go
from openai import OpenAI
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
from io import BytesIO

# --- CONFIG & SECURE API HANDLING ---
# Hardcoded for local testing to avoid 'OPENROUTER_API_KEY' system variable overrides
API_KEY = "sk-or-v1-95437b44bbdc522c2e201a4"
MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

FALLBACK_MOLECULES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "vanillin": "O=Cc1cc(OC)c(O)cc1",
}

# --- THE APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # CRITICAL: This allows Vercel to find the Flask instance

app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="bi bi-hexagon-fill me-2"),
                "BIOTECH NEURAL EXPLORER"
            ], className="nav-brand-glow")
        ]),
        color="dark",
        dark=True,
        className="mb-4 shadow-sm border-bottom border-info"
    ),

    dbc.Row([
        dbc.Col([
            html.H2("⚡ Bridging Chemical Graphs and AI Insight", className="text-center neon-text-info mt-2"),
            html.P("Explore molecular structures, properties, and generated insights in real-time.", className="text-center text-muted mb-5")
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(
                        [html.I(className="bi bi-cpu me-2"), "Compound Synthesis Target"], 
                        className="card-title neon-text-info mb-4"
                    ),
                    dbc.Input(id="drug-input", placeholder="e.g. Aspirin, Caffeine, Ibuprofen...", type="text", className="mb-4 custom-input"),
                    dcc.Loading(
                        id="loading-btn",
                        type="circle",
                        color="#00ffff",
                        children=dbc.Button("INITIALIZE NEURAL RENDER", id="submit-btn", className="w-100 neon-button mb-3")
                    ),
                    dcc.Loading(
                        id="loading-desc",
                        type="default",
                        color="#00ffff",
                        children=html.Div(id="molecule-desc", className="mt-3 text-light", style={"minHeight": "60px"})
                    )
                ])
            ], className="glass-card mb-4"),

            dbc.Card([
                dbc.CardBody([
                    html.H5("2D Molecular Graph", className="card-title neon-text-success"),
                    dcc.Loading(
                        id="loading-2d",
                        type="dot",
                        color="#00ffaa",
                        children=html.Div(
                            html.Img(id="molecule-2d", style={"maxWidth": "100%", "maxHeight": "250px", "objectFit": "contain"}),
                            className="image-container mt-3"
                        )
                    )
                ])
            ], className="glass-card mb-4")
        ], width=12, lg=4),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P(id="metric-mw", className="metric-value"),
                        html.P("Molecular Wt", className="metric-label")
                    ], className="metric-card")
                ], width=6, md=3, className="mb-3"),
                dbc.Col([
                    html.Div([
                        html.P(id="metric-logp", className="metric-value"),
                        html.P("LogP", className="metric-label")
                    ], className="metric-card")
                ], width=6, md=3, className="mb-3"),
                dbc.Col([
                    html.Div([
                        html.P(id="metric-hba", className="metric-value"),
                        html.P("H-Acceptors", className="metric-label")
                    ], className="metric-card")
                ], width=6, md=3, className="mb-3"),
                dbc.Col([
                    html.Div([
                        html.P(id="metric-hbd", className="metric-value"),
                        html.P("H-Donors", className="metric-label")
                    ], className="metric-card")
                ], width=6, md=3, className="mb-3"),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Drug-Likeness Radar", className="card-title neon-text-success mb-3"),
                            dcc.Loading(
                                id="loading-radar",
                                type="circle",
                                color="#00ffaa",
                                children=dcc.Graph(id="radar-chart", config={'displayModeBar': False})
                            )
                        ], width=12, md=5),
                        dbc.Col([
                            html.Div(className="d-flex justify-content-between align-items-center mb-2", children=[
                                html.H5("3D Conformation Render", className="card-title neon-text-info m-0"),
                                dbc.Badge("INTERACTIVE", color="info", className="ms-2")
                            ]),
                            dcc.Loading(
                                id="loading-3d",
                                type="circle",
                                color="#00ffff",
                                children=html.Iframe(id="3d-viewer", style={"width": "100%", "height": "400px", "border": "none", "borderRadius": "12px", "backgroundColor": "#060606"})
                            )
                        ], width=12, md=7)
                    ])
                ])
            ], className="glass-card")
        ], width=12, lg=8)
    ])
], fluid=True, className="p-4")

# --- CREATIVE ANALYTICS ---
def create_radar(mol):
    """Generates a Lipinski Rule radar chart."""
    mw = min(Descriptors.MolWt(mol) / 500, 1.2)
    logp = min(abs(Descriptors.MolLogP(mol)) / 5, 1.2)
    hbd = min(Descriptors.NumHDonors(mol) / 5, 1.2)
    hba = min(Descriptors.NumHAcceptors(mol) / 10, 1.2)

    fig = go.Figure(data=go.Scatterpolar(
        r=[mw, logp, hbd, hba, mw],
        theta=['MolWt','LogP','H-Donors','H-Acceptors', 'MolWt'],
        fill='toself',
        line_color='#00bc8c'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color="white", margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

def generate_3d_html(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "<html><body style='background:#060606;color:white;text-align:center;padding-top:40%;'>Invalid molecule</body></html>"
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)

    return f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; }}
        html, body {{ width: 100%; height: 100%; overflow: hidden; background-color: #060606; }}
        #container {{ width: 100%; height: 100%; position: absolute; top: 0; left: 0; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <script>
        let v = $3Dmol.createViewer("container", {{backgroundColor: "0x060606"}});
        v.addModel(`{mblock}`, "mol");
        v.setStyle({{stick: {{radius: 0.15, colorscheme: "Jmol"}}, sphere: {{scale: 0.25, colorscheme: "Jmol"}}}});
        v.zoomTo();
        v.render();
        let anim = () => {{ v.rotate(0.5, "y"); v.render(); requestAnimationFrame(anim); }};
        anim();
    </script>
</body>
</html>"""

def generate_2d_img(mol):
    """Generates base64 encoded 2D structure image."""
    img = Draw.MolToImage(mol, size=(400, 300))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.callback(
    [
        Output("molecule-desc", "children"), 
        Output("3d-viewer", "srcDoc"), 
        Output("radar-chart", "figure"),
        Output("molecule-2d", "src"),
        Output("metric-mw", "children"),
        Output("metric-logp", "children"),
        Output("metric-hba", "children"),
        Output("metric-hbd", "children")
    ],
    [Input("submit-btn", "n_clicks")],
    [State("drug-input", "value")]
)
def update_creative_app(n, drug_name):
    empty_fig = go.Figure().update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if not n or not drug_name: return "Awaiting input...", "", empty_fig, "", "-", "-", "-", "-"

    try:
        prompt = f"Provide ONLY the SMILES string and a 1-sentence bio-use for {drug_name}. Separate them with a pipe character '|'."
        res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
        content = res.choices[0].message.content.strip()
        print(f"DEBUG LLM Raw Output: {content}")
        if "|" in content:
            smiles, desc = content.split("|", 1)
        else:
            raise ValueError(f"Invalid format from LLM. No pipe character found. Output was: {content}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"LLM Error encountered: {e}")
        smiles = FALLBACK_MOLECULES.get(drug_name.lower().strip(), "")
        desc = "Local fallback used due to AI parsing error or API limitations."
        if not smiles: return html.Div("Molecule not found. Try Aspirin, Caffeine, or Vanillin.", className="text-danger font-weight-bold"), "", empty_fig, "", "-", "-", "-", "-"

    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return html.Div("Invalid SMILES structure generated.", className="text-danger"), "", empty_fig, "", "-", "-", "-", "-"
    
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)

    return (
        desc, 
        generate_3d_html(smiles), 
        create_radar(mol), 
        generate_2d_img(mol),
        f"{mw}",
        f"{logp}",
        f"{hba}",
        f"{hbd}"
    )

# No app.run() block needed for Vercel; the server variable handles it.
if __name__ == '__main__':
    app.run(debug=True, port=8050)