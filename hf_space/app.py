import gradio as gr
import math

# Pre-computed runs — each shows the agent mapping a different part of Kendall Square
LOCATIONS = {
    "Centre  (42.3601, -71.0942)": {
        "gif":     "assets/run_centre.gif",
        "coverage": "74%",
        "score":    "68%",
        "lat": 42.3601, "lon": -71.0942,
    },
    "North   (42.3621, -71.0942)": {
        "gif":     "assets/run_north.gif",
        "coverage": "—",
        "score":    "—",
        "lat": 42.3621, "lon": -71.0942,
    },
    "East    (42.3601, -71.0915)": {
        "gif":     "assets/run_east.gif",
        "coverage": "—",
        "score":    "—",
        "lat": 42.3601, "lon": -71.0915,
    },
    "South   (42.3580, -71.0942)": {
        "gif":     "assets/run_south.gif",
        "coverage": "—",
        "score":    "—",
        "lat": 42.3580, "lon": -71.0942,
    },
}

MAP_BOUNDS = {
    "north": 42.362794, "south": 42.357405,
    "east": -71.090553, "west": -71.097847,
}

def nearest_location(lat_str, lon_str):
    """Map user GPS input to the nearest pre-computed location."""
    try:
        lat, lon = float(lat_str), float(lon_str)
    except (ValueError, TypeError):
        return None
    best, best_dist = None, float("inf")
    for name, meta in LOCATIONS.items():
        d = math.hypot(lat - meta["lat"], lon - meta["lon"])
        if d < best_dist:
            best_dist, best = d, name
    return best


def show_location(choice):
    meta = LOCATIONS[choice]
    stats = ""
    if meta["coverage"] != "—":
        stats = f"**{meta['coverage']}** region coverage · **{meta['score']}** accuracy · Kendall Square, Boston · 10 m/cell"
    else:
        stats = "Kendall Square, Boston · 10 m/cell"
    return meta["gif"], stats


def run_custom(lat_str, lon_str):
    name = nearest_location(lat_str, lon_str)
    if name is None:
        return (
            LOCATIONS["Centre  (42.3601, -71.0942)"]["gif"],
            "Invalid coordinates — showing Centre run instead.",
        )
    meta = LOCATIONS[name]
    note = f"Nearest pre-computed location: **{name.strip()}**"
    return meta["gif"], note


CSS = """
#title { text-align: center; margin-bottom: 0; }
#subtitle { text-align: center; color: #888; margin-top: 4px; }
.stat-box { font-size: 0.9em; color: #aaa; min-height: 24px; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS, title="Active SLAM") as demo:

    gr.Markdown("# Active SLAM — Mamba SSM + PPO", elem_id="title")
    gr.Markdown(
        "An agent maps an unknown city block using only local LiDAR observations. "
        "The SLAM belief map is built in real-time from scratch.",
        elem_id="subtitle",
    )

    gr.Markdown("---")

    # ── Best result banner ────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Best result")
            gr.Markdown(
                "**89.7% region coverage · 88.8% accuracy**  \n"
                "Kendall Square, Boston · 600 m² target area · 1200 steps"
            )
            gr.Markdown(
                "Left panel: SLAM belief map being built from sensor data alone.  \n"
                "Right panel: ground truth + agent trajectory (orange).",
                elem_classes="stat-box",
            )
        with gr.Column(scale=2):
            gr.Image("assets/run_centre.gif", label=None, show_label=False,
                     show_download_button=False, container=False)

    gr.Markdown("---")

    # ── Explorer ──────────────────────────────────────────────────
    gr.Markdown("### Explore the map")
    gr.Markdown(
        "Pick a target area to see the agent navigate and map it.",
        elem_classes="stat-box",
    )

    with gr.Row():
        with gr.Column(scale=1):
            location_dd = gr.Dropdown(
                choices=list(LOCATIONS.keys()),
                value=list(LOCATIONS.keys())[0],
                label="Target area",
                interactive=True,
            )
            stats_md = gr.Markdown(elem_classes="stat-box")

            gr.Markdown("#### Or enter GPS coordinates")
            gr.Markdown(
                "Coordinates are mapped to the nearest pre-computed run.",
                elem_classes="stat-box",
            )
            with gr.Row():
                lat_in  = gr.Textbox(label="Latitude",  placeholder="42.361", scale=1)
                lon_in  = gr.Textbox(label="Longitude", placeholder="-71.092", scale=1)
            gps_btn = gr.Button("Find nearest run", variant="secondary")

        with gr.Column(scale=2):
            output_gif = gr.Image(label=None, show_label=False,
                                  show_download_button=True, container=False)

    # ── Wire up ───────────────────────────────────────────────────
    location_dd.change(fn=show_location, inputs=[location_dd],
                       outputs=[output_gif, stats_md])
    gps_btn.click(fn=run_custom, inputs=[lat_in, lon_in],
                  outputs=[output_gif, stats_md])

    # Load default on startup
    demo.load(fn=show_location, inputs=[location_dd],
              outputs=[output_gif, stats_md])

    gr.Markdown("---")
    gr.Markdown(
        "<div style='text-align:center; color:#666; font-size:0.8em'>"
        "Mamba SSM (d=128, 2 layers) · PPO · Bayesian occupancy grid · "
        "Bresenham LiDAR · Kendall Square, Boston (real OSM data)  "
        "</div>"
    )

if __name__ == "__main__":
    demo.launch()
