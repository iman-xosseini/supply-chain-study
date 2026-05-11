# Standard library imports
import json
import os
import re
from datetime import datetime, timedelta
from typing import Literal, Optional, Union

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from openai import OpenAI
from pyvis.network import Network



def visualize_important_features(
    importance_df,
    top_n=20,
    effect_type="both",
    figsize=(14, 10),
    fontsize=11,
    show_values=True,
    min_threshold=1e-10,
):

    # Filter out features with absolute importance below threshold (effectively 0)
    filtered_df = importance_df[
        importance_df["feature_importance"].abs() > min_threshold
    ].copy()

    # Filter by effect type
    if effect_type == "positive":
        filtered_df = filtered_df[filtered_df["feature_importance"] > 0.0].copy()
        title_suffix = "(Positive Effects Only)"
        color_scheme = "green"
    elif effect_type == "negative":
        filtered_df = filtered_df[filtered_df["feature_importance"] < 0.0].copy()
        title_suffix = "(Negative Effects Only)"
        color_scheme = "red"
    elif effect_type == "both":
        title_suffix = "(All Effects)"
        color_scheme = None  # Will use conditional coloring
    else:
        print(
            f"❌ Error: effect_type must be 'positive', 'negative', or 'both'. Got: {effect_type}"
        )
        return None

    # Check if we have data
    if len(filtered_df) == 0:
        print(
            f"❌ No features found with {effect_type} effects (excluding near-zero effects)"
        )
        return pd.DataFrame()

    # Sort by absolute importance and get top N
    filtered_df["abs_importance"] = filtered_df["feature_importance"].abs()
    top_features = filtered_df.nlargest(top_n, "abs_importance")

    # Sort for display (ascending for horizontal bar chart - lowest at bottom)
    top_features = top_features.sort_values("feature_importance", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if color_scheme:
        colors = [color_scheme] * len(top_features)
    else:
        colors = [
            "green" if x > 0 else "red" for x in top_features["feature_importance"]
        ]

    # Create horizontal bar chart
    bars = ax.barh(
        range(len(top_features)),
        top_features["feature_importance"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
    )

    # Set y-axis labels (feature names)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature_name"], fontsize=fontsize)

    # Set x-axis label
    ax.set_xlabel(
        "Feature Importance (SHAP Value)", fontsize=fontsize + 2, fontweight="bold"
    )

    # Set title
    actual_count = len(top_features)
    title = f"Top {actual_count} Most Important Features {title_suffix}"
    ax.set_title(title, fontsize=fontsize + 4, fontweight="bold", pad=20)

    # Add zero line
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.8)

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Add value labels on bars if requested
    if show_values:
        for i, (bar, val) in enumerate(zip(bars, top_features["feature_importance"])):
            # Position text at the end of the bar
            if val >= 0:
                x_pos = val
                ha = "left"
                offset = max(top_features["feature_importance"]) * 0.01
            else:
                x_pos = val
                ha = "right"
                offset = -max(top_features["feature_importance"].abs()) * 0.01

            ax.text(
                x_pos + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                ha=ha,
                va="center",
                fontsize=fontsize - 2,
                fontweight="bold",
            )

    # Add statistics box
    stats_text = (
        f"Total features shown: {len(top_features)}\n"
        f"Positive: {(top_features['feature_importance'] > 0).sum()}\n"
        f"Negative: {(top_features['feature_importance'] < 0).sum()}\n"
        f"Max: {top_features['feature_importance'].max():.4f}\n"
        f"Min: {top_features['feature_importance'].min():.4f}"
    )

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=fontsize - 1,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor="lightyellow", alpha=0.8, edgecolor="black"
        ),
    )

    plt.tight_layout()
    plt.show()

    # Print summary
    zero_count = len(importance_df) - len(filtered_df)
    print("\n" + "=" * 80)
    print(f"FEATURE IMPORTANCE VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"Effect type filter: {effect_type}")
    print(f"Features in original dataset: {len(importance_df)}")
    print(f"Features excluded (near-zero): {zero_count}")
    print(f"Features with non-zero effects: {len(filtered_df)}")
    print(f"Features displayed (top {top_n}): {len(top_features)}")
    print(f"  - Positive effects: {(top_features['feature_importance'] > 0).sum()}")
    print(f"  - Negative effects: {(top_features['feature_importance'] < 0).sum()}")
    print("=" * 80)

    # Return the displayed features
    return top_features.sort_values("abs_importance", ascending=False)[
        ["feature_name", "feature_importance", "abs_importance"]
    ]


# ============================================================================
# INTERACTIVE KNOWLEDGE GRAPH VISUALIZATION (refined_triplet)
# ============================================================================

# ── Pre-built themes ────────────────────────────────────────────────────────
KG_THEMES = {
    "light": {
        "bg_color": "#ffffff",
        "node_font_color": "#1a1a1a",
        "edge_font_color": "#333333",
        "node_border": "#555555",
        "default_edge": "#888888",
        "legend_bg": "rgba(255,255,255,0.94)",
        "legend_text": "#1a1a1a",
        "legend_border": "#cccccc",
        "legend_hr": "#dddddd",
        "legend_muted": "#666666",
        "node_colors": [  # category palette
            "#0077b6",
            "#e63946",
            "#2a9d8f",
            "#e76f51",
            "#6a4c93",
            "#1d3557",
            "#f4a261",
            "#264653",
            "#a8dadc",
            "#457b9d",
            "#d62828",
            "#118ab2",
            "#06d6a0",
            "#ef476f",
            "#ffd166",
            "#073b4c",
            "#8338ec",
        ],
        "edge_relation_colors": {
            "decrease": "#d62828",
            "depreciate": "#e76f51",
            "increase": "#2a9d8f",
            "deteriorate": "#c1121f",
            "improve": "#0077b6",
            "disrupt": "#e63946",
            "stabilize": "#06d6a0",
        },
    },
    "dark": {
        "bg_color": "#1a1a2e",
        "node_font_color": "#eaeaea",
        "edge_font_color": "#cccccc",
        "node_border": "#aaaaaa",
        "default_edge": "#888888",
        "legend_bg": "rgba(30,30,60,0.92)",
        "legend_text": "#eaeaea",
        "legend_border": "#444444",
        "legend_hr": "#555555",
        "legend_muted": "#aaaaaa",
        "node_colors": [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabed4",
            "#469990",
            "#dcbeff",
            "#9A6324",
            "#800000",
            "#aaffc3",
            "#808000",
            "#000075",
            "#a9a9a9",
        ],
        "edge_relation_colors": {
            "decrease": "#ff6b6b",
            "depreciate": "#ffa502",
            "increase": "#2ed573",
            "deteriorate": "#ff4757",
            "improve": "#1e90ff",
            "disrupt": "#ff6348",
            "stabilize": "#7bed9f",
        },
    },
    "pastel": {
        "bg_color": "#fdf6ec",
        "node_font_color": "#3d3d3d",
        "edge_font_color": "#555555",
        "node_border": "#8d8d8d",
        "default_edge": "#aaaaaa",
        "legend_bg": "rgba(253,246,236,0.95)",
        "legend_text": "#3d3d3d",
        "legend_border": "#d4c5a9",
        "legend_hr": "#e0d5c1",
        "legend_muted": "#888888",
        "node_colors": [
            "#a8d8ea",
            "#aa96da",
            "#fcbad3",
            "#ffffd2",
            "#b5eaea",
            "#f6c6a8",
            "#c3aed6",
            "#ffb6b9",
            "#fae3d9",
            "#bbded6",
            "#8ac6d1",
            "#e8d5b7",
            "#f0e6ef",
            "#d5ecc2",
            "#f7d794",
            "#cf6a87",
            "#786fa6",
        ],
        "edge_relation_colors": {
            "decrease": "#cf6a87",
            "depreciate": "#e77f67",
            "increase": "#78e08f",
            "deteriorate": "#e55039",
            "improve": "#4a69bd",
            "disrupt": "#eb4d4b",
            "stabilize": "#b8e994",
        },
    },
    "highcontrast": {
        "bg_color": "#ffffff",
        "node_font_color": "#000000",
        "edge_font_color": "#000000",
        "node_border": "#000000",
        "default_edge": "#333333",
        "legend_bg": "rgba(245,245,245,0.96)",
        "legend_text": "#000000",
        "legend_border": "#000000",
        "legend_hr": "#000000",
        "legend_muted": "#333333",
        "node_colors": [
            "#d32f2f",
            "#1976d2",
            "#388e3c",
            "#f57c00",
            "#7b1fa2",
            "#0097a7",
            "#c2185b",
            "#455a64",
            "#e64a19",
            "#00796b",
            "#5d4037",
            "#303f9f",
            "#689f38",
            "#fbc02d",
            "#8e24aa",
            "#00838f",
            "#ad1457",
        ],
        "edge_relation_colors": {
            "decrease": "#d32f2f",
            "depreciate": "#e64a19",
            "increase": "#2e7d32",
            "deteriorate": "#b71c1c",
            "improve": "#1565c0",
            "disrupt": "#c62828",
            "stabilize": "#00695c",
        },
    },
}


def visualize_knowledge_graph(
    df: pd.DataFrame,
    output_file: str = "refined_triplet_knowledge_graph.html",
    triplet_col: str = "refined_triplet",
    attribute_col: str = "attribute",
    # ── Theme ────────────────────────────────────────────────────────────
    theme: Literal["light", "dark", "pastel", "highcontrast"] = "light",
    # ── Sizing ───────────────────────────────────────────────────────────
    node_size: int = 25,
    node_font_size: int = 14,
    edge_font_size: int = 12,
    # ── Edge-label style ─────────────────────────────────────────────────
    edge_label_style: Literal[
        "text", "rectangle", "roundrect", "ellipse", "circle", "diamond"
    ] = "text",
    edge_label_padding: int = 5,
) -> str:
    """
    Build an interactive, full-screen knowledge graph and save as HTML.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `triplet_col` and `attribute_col` columns.
    output_file : str
        Destination HTML file path.
    triplet_col / attribute_col : str
        Column names for triplet strings and category attributes.
    theme : str
        One of  "light" | "dark" | "pastel" | "highcontrast".
    node_size : int
        Uniform radius for every node (default 25).
    node_font_size : int
        Font size (px) for node labels (default 14).
    edge_font_size : int
        Font size (px) for edge / relation labels (default 12).
    edge_label_style : str
        "text"  – plain floating text (no background).
        "rectangle" | "roundrect" | "ellipse" | "circle" | "diamond"
                – label is drawn inside the chosen shape with a
                  semi-transparent background.
    edge_label_padding : int
        Extra padding (px) around the edge label when a shape is used
        (ignored when edge_label_style="text").

    Returns
    -------
    str   Path to the saved HTML file.
    """

    th = KG_THEMES[theme]

    # ── 1. Parse triplets ────────────────────────────────────────────────
    pattern = r"\((.+?)\)\s*──\s*(.+?)\s*──>\s*\((.+?)\)"
    triplets = []
    for t in df[triplet_col].dropna():
        m = re.match(pattern, str(t).strip())
        if m:
            triplets.append(
                (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
            )

    print(f"Parsed {len(triplets)} triplets from {len(df)} rows")

    # ── 2. Build directed graph ──────────────────────────────────────────
    G = nx.DiGraph()
    for subj, rel, obj in triplets:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=rel)

    degree = dict(G.degree())

    # ── 3. Category colours ──────────────────────────────────────────────
    node_category = {}
    for _, row in df.dropna(subset=[triplet_col, attribute_col]).iterrows():
        m = re.match(pattern, str(row[triplet_col]).strip())
        if m:
            subj = m.group(1).strip()
            obj = m.group(3).strip()
            cats = re.findall(r':\s*([^",]+)', str(row[attribute_col]))
            if cats:
                cat = cats[0].strip()
                node_category[subj] = cat
                node_category[obj] = node_category.get(obj, cat)

    unique_cats = sorted(set(node_category.values()))
    cat_color = {
        cat: th["node_colors"][i % len(th["node_colors"])]
        for i, cat in enumerate(unique_cats)
    }

    # ── 4. Build Pyvis network ───────────────────────────────────────────
    net = Network(
        height="100%",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor=th["bg_color"],
        font_color=th["node_font_color"],
        cdn_resources="remote",
    )

    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01,
        damping=0.09,
    )

    # ── Nodes (uniform size) ─────────────────────────────────────────────
    for node in G.nodes():
        cat = node_category.get(node, "Other")
        color = cat_color.get(cat, th["node_colors"][-1])
        net.add_node(
            node,
            label=node,
            title=(
                f"<b>{node}</b><br>"
                f"Category: {cat}<br>"
                f"Connections: {degree.get(node, 0)}"
            ),
            color={
                "background": color,
                "border": th["node_border"],
                "highlight": {"background": color, "border": "#000000"},
            },
            size=node_size,
            font={
                "size": node_font_size,
                "color": th["node_font_color"],
                "face": "Arial",
            },
            borderWidth=2,
            borderWidthSelected=4,
        )

    # ── Edge-label font configuration ────────────────────────────────────
    use_background = edge_label_style != "text"

    def _edge_font(color: str) -> dict:
        """Return the vis.js font dict for an edge label."""
        font = {
            "size": edge_font_size,
            "color": color if not use_background else th["edge_font_color"],
            "face": "Arial",
            "strokeWidth": 0 if use_background else 3,
            "strokeColor": th["bg_color"],  # halo matches background
            "align": "top",
        }
        if use_background:
            # vis.js draws a filled shape behind the label text
            font["background"] = th["bg_color"]
        return font

    edge_rel_colors = th["edge_relation_colors"]

    # ── Edges ────────────────────────────────────────────────────────────
    for subj, rel, obj in triplets:
        e_color = edge_rel_colors.get(rel.lower(), th["default_edge"])
        edge_opts = dict(
            label=rel,
            title=f"{subj}  ──  {rel}  ──>  {obj}",
            color={"color": e_color, "highlight": e_color, "opacity": 0.9},
            arrows="to",
            width=2,
            font=_edge_font(e_color),
            smooth={"type": "curvedCW", "roundness": 0.15},
        )
        net.add_edge(subj, obj, **edge_opts)

    # ── 5. Legend HTML ───────────────────────────────────────────────────
    legend_items = "".join(
        f'<div style="margin:3px 0;"><span style="display:inline-block;'
        f"width:14px;height:14px;background:{cat_color[c]};border-radius:50%;"
        f'margin-right:8px;vertical-align:middle;"></span>'
        f'<span style="vertical-align:middle;">{c}</span></div>'
        for c in unique_cats
    )
    edge_legend = "".join(
        f'<div style="margin:3px 0;"><span style="display:inline-block;'
        f"width:22px;height:3px;background:{col};margin-right:8px;"
        f'vertical-align:middle;border-radius:2px;"></span>'
        f'<span style="vertical-align:middle;">{rel}</span></div>'
        for rel, col in sorted(edge_rel_colors.items())
    )

    legend_html = f"""
    <div id="legend" style="position:fixed;top:12px;right:12px;
      background:{th['legend_bg']};padding:16px 20px;border-radius:10px;
      color:{th['legend_text']};font-family:Arial,sans-serif;font-size:13px;
      max-height:90vh;overflow-y:auto;z-index:9999;
      border:1px solid {th['legend_border']};
      box-shadow:0 2px 12px rgba(0,0,0,0.08);">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Node Categories</div>
      {legend_items}
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Edge Relations</div>
      {edge_legend}
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-size:11px;color:{th['legend_muted']};">
        Nodes: {G.number_of_nodes()} &nbsp;|&nbsp; Edges: {G.number_of_edges()}
        &nbsp;|&nbsp; Theme: {theme}
      </div>
    </div>
    """

    # ── 6. Full-screen CSS + edge-label shape override ───────────────────
    # vis.js does not expose "edge label background shape" via its options
    # API; the `font.background` key draws a filled rect behind the text.
    # To achieve rounded-rect, ellipse, etc. we inject a tiny post-render
    # JS patch that overrides the edge-label drawing routine.
    shape_js = ""
    if use_background and edge_label_style != "rectangle":
        # Map our friendly names → canvas drawing snippets
        shape_draw = {
            "roundrect": f"""
                var r = 6;
                ctx.beginPath();
                ctx.moveTo(x - w/2 + r, y - h/2);
                ctx.arcTo(x + w/2, y - h/2, x + w/2, y + h/2, r);
                ctx.arcTo(x + w/2, y + h/2, x - w/2, y + h/2, r);
                ctx.arcTo(x - w/2, y + h/2, x - w/2, y - h/2, r);
                ctx.arcTo(x - w/2, y - h/2, x + w/2, y - h/2, r);
                ctx.closePath(); ctx.fill(); ctx.stroke();""",
            "ellipse": """
                ctx.beginPath();
                ctx.ellipse(x, y, w/2, h/2, 0, 0, 2*Math.PI);
                ctx.fill(); ctx.stroke();""",
            "circle": """
                var rad = Math.max(w, h) / 2;
                ctx.beginPath();
                ctx.arc(x, y, rad, 0, 2*Math.PI);
                ctx.fill(); ctx.stroke();""",
            "diamond": """
                ctx.beginPath();
                ctx.moveTo(x, y - h/2);
                ctx.lineTo(x + w/2, y);
                ctx.lineTo(x, y + h/2);
                ctx.lineTo(x - w/2, y);
                ctx.closePath(); ctx.fill(); ctx.stroke();""",
        }
        draw_code = shape_draw.get(edge_label_style, "")
        if draw_code:
            shape_js = f"""
            <script>
            // Override edge-label background drawing after vis.js renders
            (function() {{
              var origDraw = null;
              var net = document.querySelector('#mynetwork');
              var observer = new MutationObserver(function() {{
                var canvas = net.querySelector('canvas');
                if (!canvas) return;
                observer.disconnect();
                // Patch the 2d context fillRect used for label backgrounds
                var ctx = canvas.getContext('2d');
                var _fillRect = ctx.fillRect.bind(ctx);
                var pad = {edge_label_padding};
                ctx.fillRect = function(rx, ry, rw, rh) {{
                  // Only intercept small label-sized rects (not the full canvas)
                  if (rw < 300 && rh < 60) {{
                    var x = rx + rw/2, y = ry + rh/2;
                    var w = rw + pad*2, h = rh + pad*2;
                    ctx.save();
                    ctx.fillStyle = ctx.fillStyle;
                    ctx.strokeStyle = '{th["legend_border"]}';
                    ctx.lineWidth = 1;
                    {draw_code}
                    ctx.restore();
                  }} else {{
                    _fillRect(rx, ry, rw, rh);
                  }}
                }};
              }});
              observer.observe(net, {{childList: true, subtree: true}});
            }})();
            </script>
            """

    fullscreen_css = f"""
    <style>
      html, body {{ margin:0; padding:0; width:100%; height:100%;
                    overflow:hidden; background:{th['bg_color']}; }}
      #mynetwork {{ width:100vw !important; height:100vh !important;
                    border:none !important; margin:0 !important;
                    padding:0 !important; }}
    </style>
    """

    # ── 7. Save & inject ─────────────────────────────────────────────────
    net.save_graph(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("<head>", "<head>" + fullscreen_css, 1)
    html = html.replace("</body>", legend_html + shape_js + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Interactive graph saved to: {output_file}")
    print(
        f"  Theme: {theme}  |  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}"
    )
    print(
        f"  Node size: {node_size}  |  Node font: {node_font_size}px  |  Edge font: {edge_font_size}px"
    )
    print(
        f"  Edge label style: {edge_label_style}"
        + (f"  |  Label padding: {edge_label_padding}px" if use_background else "")
    )
    print(f"  Categories: {', '.join(unique_cats)}")
    return output_file


# ============================================================================
# UNIFIED CAUSAL + EVENT GRAPH VISUALIZATION
# ============================================================================

_TRIPLET_PATTERN = r"\((.+?)\)\s*──\s*(.+?)\s*──>\s*\((.+?)\)"


def _parse_refined_triplet(triplet_str: str):
    """Parse '(Source) ── relation ──> (Target)' into (source, relation, target)."""
    m = re.match(_TRIPLET_PATTERN, str(triplet_str).strip())
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return None, None, None


def _extract_attribute_categories(attribute_str) -> list:
    """
    Extract category labels from an attribute string.

    Each item has the form ``"Entity: Category"`` and multiple items are
    comma-separated.  Only the part **after** the colon is returned.

    Examples
    --------
    >>> _extract_attribute_categories('"New SPM Facility: Infrastructure"')
    ['Infrastructure']
    >>> _extract_attribute_categories(
    ...     '"New wave: Infrastructure", "whatever: political event"'
    ... )
    ['Infrastructure', 'political event']
    """
    if pd.isna(attribute_str):
        return []
    cats = re.findall(r':\s*([^",]+)', str(attribute_str))
    return [c.strip() for c in cats if c.strip()]


def build_unified_graph(filtered_causal_df: pd.DataFrame, enriched_kg_df: pd.DataFrame, triplet_col: str = "refined_triplet", attribute_col: str = "attribute",) -> pd.DataFrame:
    """
    Integrate a causal graph with an event-based knowledge graph.

    For each *source* in ``filtered_causal_df``:

    1. Pull matching event triplets from ``enriched_kg_df``
       (rows where ``feature == source``).
    2. Create **event edges**:  event_entity  → relation → causal_source
    3. Create **attribute edges**: event_entity → has_attribute → category
       (categories are extracted from the ``attribute`` column, taking only
       the part after the colon, e.g. ``"Infrastructure"`` from
       ``"New SPM Facility: Infrastructure"``).
    4. Keep **causal edges**:   causal_source → causal_effect → causal_target

    Returns
    -------
    pd.DataFrame
        Unified edge list with columns:
        source, relation, target, edge_type, weight, date, title, attribute
    """
    unified_edges = []

    for src in filtered_causal_df["source"].unique():
        # ── Event edges ──────────────────────────────────────────────────
        event_rows = enriched_kg_df.loc[enriched_kg_df["feature"] == src]
        for _, row in event_rows.iterrows():
            ev_src, relation, ev_tgt = _parse_refined_triplet(row[triplet_col])
            if ev_src is None:
                continue
            unified_edges.append({
                "source": ev_src,
                "relation": relation,
                "target": ev_tgt,
                "edge_type": "event",
                "weight": None,
                "date": row.get("date"),
                "title": row.get("title"),
                "attribute": row.get(attribute_col),
            })

            # ── Attribute edges: feature ──> category ────────────────────
            categories = _extract_attribute_categories(row.get(attribute_col))
            for cat in categories:
                unified_edges.append({
                    "source": src,
                    "relation": "has_attribute",
                    "target": cat,
                    "edge_type": "attribute",
                    "weight": None,
                    "date": row.get("date"),
                    "title": row.get("title"),
                    "attribute": row.get(attribute_col),
                })

        # ── Causal edges ─────────────────────────────────────────────────
        for _, row in filtered_causal_df.loc[
            filtered_causal_df["source"] == src
        ].iterrows():
            unified_edges.append({
                "source": row["source"],
                "relation": "causal_effect",
                "target": row["target"],
                "edge_type": "causal",
                "weight": row["weight"],
                "date": None,
                "title": None,
                "attribute": None,
            })

    return pd.DataFrame(unified_edges)


def visualize_unified_graph(
    unified_df: pd.DataFrame,
    output_file: str = "unified_causal_event_graph.html",
    attribute_col: str = "attribute",
    # ── Display mode ─────────────────────────────────────────────────────
    display_mode: Literal["all", "event", "causal"] = "all",
    # ── Edge labels ──────────────────────────────────────────────────────
    show_edge_labels: bool = True,
    # ── Theme ────────────────────────────────────────────────────────────
    theme: Literal["light", "dark", "pastel", "highcontrast"] = "light",
    # ── Sizing ───────────────────────────────────────────────────────────
    node_size: int = 25,
    node_font_size: int = 14,
    edge_font_size: int = 12,
    # ── Edge-label style ─────────────────────────────────────────────────
    edge_label_style: Literal[
        "text", "rectangle", "roundrect", "ellipse", "circle", "diamond"
    ] = "text",
    edge_label_padding: int = 5,
) -> str:
    """
    Build an interactive, full-screen unified causal + event graph.

    Parameters
    ----------
    unified_df : pd.DataFrame
        Output of :func:`build_unified_graph`.  Must contain columns
        ``source``, ``relation``, ``target``, ``edge_type``, ``weight``,
        ``attribute``.
    output_file : str
        Destination HTML file path.
    display_mode : str
        ``"all"``    – show causal, event **and** attribute edges.
        ``"event"``  – show only event + attribute edges (no causal).
        ``"causal"`` – show only causal edges (no event / attribute).
    theme : str
        One of  "light" | "dark" | "pastel" | "highcontrast".
    node_size, node_font_size, edge_font_size : int
        Visual sizing parameters.
    edge_label_style : str
        Shape behind edge labels (same options as
        :func:`visualize_knowledge_graph`).
    edge_label_padding : int
        Extra padding (px) when a label shape is used.

    Returns
    -------
    str   Path to the saved HTML file.
    """

    th = KG_THEMES[theme]

    # ── 0. Filter by display_mode ────────────────────────────────────────
    if display_mode == "event":
        work_df = unified_df[unified_df["edge_type"].isin(["event", "attribute"])].copy()
    elif display_mode == "causal":
        work_df = unified_df[unified_df["edge_type"] == "causal"].copy()
    else:
        work_df = unified_df.copy()

    # ── 1. Build directed graph ──────────────────────────────────────────
    G = nx.DiGraph()
    causal_nodes: set = set()
    event_nodes: set = set()
    attribute_nodes: set = set()

    # Collect category info from event attribute column
    node_category: dict = {}

    for _, row in work_df.iterrows():
        src, tgt = row["source"], row["target"]

        if row["edge_type"] == "causal":
            causal_nodes.update([src, tgt])
            w = row["weight"]
            label = f"{w:+.4f}"
            G.add_edge(src, tgt, label=label, edge_type="causal", weight=w)
        elif row["edge_type"] == "attribute":
            event_nodes.add(src)
            attribute_nodes.add(tgt)
            G.add_edge(
                src, tgt,
                label=row["relation"],
                edge_type="attribute",
                weight=None,
            )
            node_category[tgt] = tgt  # category node IS the category
        else:  # event
            event_nodes.add(src)
            causal_nodes.add(tgt)
            G.add_edge(
                src, tgt,
                label=row["relation"],
                edge_type="event",
                weight=None,
                title=f"{row.get('title', '')}\n{row.get('date', '')}",
            )
            # Extract categories from attribute for colouring event nodes
            if pd.notna(row.get(attribute_col)):
                cats = re.findall(r':\s*([^",]+)', str(row[attribute_col]))
                if cats:
                    node_category[src] = cats[0].strip()

    # Label causal-only nodes
    for n in causal_nodes - event_nodes - attribute_nodes:
        node_category.setdefault(n, "Causal Feature")
    # Label attribute nodes
    for n in attribute_nodes:
        node_category.setdefault(n, n)

    degree = dict(G.degree())
    unique_cats = sorted(set(node_category.values()))
    cat_color = {
        cat: th["node_colors"][i % len(th["node_colors"])]
        for i, cat in enumerate(unique_cats)
    }

    # ── 2. Build Pyvis network ───────────────────────────────────────────
    net = Network(
        height="100%",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor=th["bg_color"],
        font_color=th["node_font_color"],
        cdn_resources="remote",
    )

    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01,
        damping=0.09,
    )

    # ── Node-type visual profiles ──────────────────────────────────────
    # Each node type gets a distinct border color + width + size offset
    # so causal vs event vs attribute nodes are immediately recognizable.
    _node_profiles = {
        "Causal Feature": {
            "shape": "box",
            "border": "#2e7d32",          # green border
            "border_width": 3,
            "size_offset": 4,             # slightly larger
            "shadow_color": "rgba(46,125,50,0.45)",
        },
        "Event Entity": {
            "shape": "dot",               # circle (filled dot) — very different from box
            "border": "#e67e22",          # orange border
            "border_width": 3,
            "size_offset": 0,
            "shadow_color": "rgba(230,126,34,0.45)",
        },
        "Attribute": {
            "shape": "diamond",
            "border": "#8e44ad",          # purple border
            "border_width": 2,
            "size_offset": -4,            # slightly smaller
            "shadow_color": "rgba(142,68,173,0.35)",
        },
    }

    # ── Nodes ────────────────────────────────────────────────────────────
    for node in G.nodes():
        cat = node_category.get(node, "Other")
        color = cat_color.get(cat, th["node_colors"][-1])
        if node in attribute_nodes:
            node_type = "Attribute"
        elif node in causal_nodes and node not in event_nodes:
            node_type = "Causal Feature"
        else:
            node_type = "Event Entity"

        prof = _node_profiles[node_type]
        net.add_node(
            node,
            label=node,
            shape=prof["shape"],
            title=(
                f"<b>{node}</b><br>"
                f"Category: {cat}<br>"
                f"Type: {node_type}<br>"
                f"Connections: {degree.get(node, 0)}"
            ),
            color={
                "background": color,
                "border": prof["border"],
                "highlight": {"background": color, "border": prof["border"]},
            },
            size=node_size + prof["size_offset"],
            font={
                "size": node_font_size,
                "color": th["node_font_color"],
                "face": "Arial",
            },
            borderWidth=prof["border_width"],
            borderWidthSelected=prof["border_width"] + 2,
            shadow={
                "enabled": True,
                "color": prof["shadow_color"],
                "size": 10,
                "x": 0,
                "y": 0,
            },
        )

    # ── Edge-label font helper ───────────────────────────────────────────
    use_background = edge_label_style != "text"
    edge_rel_colors = th["edge_relation_colors"]

    def _edge_font(color: str) -> dict:
        font = {
            "size": edge_font_size,
            "color": color if not use_background else th["edge_font_color"],
            "face": "Arial",
            "strokeWidth": 0 if use_background else 3,
            "strokeColor": th["bg_color"],
            "align": "top",
        }
        if use_background:
            font["background"] = th["bg_color"]
        return font

    # ── Edges ────────────────────────────────────────────────────────────
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type")
        if etype == "causal":
            w = data["weight"]
            e_color = "#2e7d32" if w >= 0 else "#d32f2f"
            edge_opts = dict(
                label=data["label"],
                title=f"{u}  ── causal ({w:+.4f}) ──>  {v}",
                color={"color": e_color, "highlight": e_color, "opacity": 0.9},
                arrows="to",
                width=3,
                font=_edge_font(e_color),
                smooth={"type": "curvedCW", "roundness": 0.15},
                dashes=False,
            )
        elif etype == "attribute":
            e_color = "#8e44ad"  # purple for attribute edges
            edge_opts = dict(
                label=data["label"],
                title=f"{u}  ── has_attribute ──>  {v}",
                color={"color": e_color, "highlight": e_color, "opacity": 0.9},
                arrows="to",
                width=2,
                font=_edge_font(e_color),
                smooth={"type": "curvedCW", "roundness": 0.15},
                dashes=[2, 4],
            )
        else:  # event
            rel = data["label"]
            e_color = edge_rel_colors.get(rel.lower(), th["default_edge"])
            edge_opts = dict(
                label=rel,
                title=data.get("title", f"{u}  ──  {rel}  ──>  {v}"),
                color={"color": e_color, "highlight": e_color, "opacity": 0.9},
                arrows="to",
                width=2,
                font=_edge_font(e_color),
                smooth={"type": "curvedCW", "roundness": 0.15},
                dashes=[5, 5],
            )
        if not show_edge_labels:
            edge_opts.pop("label", None)
            edge_opts.pop("font", None)
        net.add_edge(u, v, **edge_opts)

    # ── 3. Legend HTML ───────────────────────────────────────────────────
    cat_items = "".join(
        f'<div style="margin:3px 0;"><span style="display:inline-block;'
        f"width:14px;height:14px;background:{cat_color[c]};border-radius:50%;"
        f'margin-right:8px;vertical-align:middle;"></span>'
        f'<span style="vertical-align:middle;">{c}</span></div>'
        for c in unique_cats
    )

    n_event = (work_df["edge_type"] == "event").sum()
    n_causal = (work_df["edge_type"] == "causal").sum()
    n_attr = (work_df["edge_type"] == "attribute").sum()

    type_legend = ""
    if display_mode in ("all", "causal"):
        type_legend += (
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:22px;height:3px;'
            'background:#2e7d32;margin-right:8px;vertical-align:middle;"></span>'
            f'<span style="vertical-align:middle;">Causal + ({n_causal})</span></div>'
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:22px;height:3px;'
            'background:#d32f2f;margin-right:8px;vertical-align:middle;"></span>'
            f'<span style="vertical-align:middle;">Causal \u2212 ({n_causal})</span></div>'
        )
    if display_mode in ("all", "event"):
        type_legend += (
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:22px;height:3px;'
            'background:#3498db;margin-right:8px;vertical-align:middle;'
            'border-top:2px dashed #3498db;height:0;"></span>'
            f'<span style="vertical-align:middle;">Event ({n_event})</span></div>'
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:22px;height:3px;'
            'background:#8e44ad;margin-right:8px;vertical-align:middle;'
            'border-top:2px dotted #8e44ad;height:0;"></span>'
            f'<span style="vertical-align:middle;">Attribute ({n_attr})</span></div>'
        )

    shape_legend = ""
    if display_mode in ("all", "causal"):
        shape_legend += (
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:14px;height:14px;'
            'border:3px solid #2e7d32;margin-right:8px;vertical-align:middle;'
            'box-shadow:0 0 6px rgba(46,125,50,0.5);"></span>'
            '<span style="vertical-align:middle;">Causal Feature (box, green border)</span></div>'
        )
    if display_mode in ("all", "event"):
        shape_legend += (
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:14px;height:14px;'
            'border:3px solid #e67e22;border-radius:50%;margin-right:8px;'
            'vertical-align:middle;box-shadow:0 0 6px rgba(230,126,34,0.5);"></span>'
            '<span style="vertical-align:middle;">Event Entity (circle, orange border)</span></div>'
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:12px;height:12px;'
            'border:2px solid #8e44ad;transform:rotate(45deg);margin-right:8px;'
            'vertical-align:middle;box-shadow:0 0 6px rgba(142,68,173,0.4);"></span>'
            '<span style="vertical-align:middle;">Attribute (diamond, purple border)</span></div>'
        )

    legend_html = f"""
    <div id="legend" style="position:fixed;top:12px;right:12px;
      background:{th['legend_bg']};padding:16px 20px;border-radius:10px;
      color:{th['legend_text']};font-family:Arial,sans-serif;font-size:13px;
      max-height:90vh;overflow-y:auto;z-index:9999;
      border:1px solid {th['legend_border']};
      box-shadow:0 2px 12px rgba(0,0,0,0.08);">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Node Categories</div>
      {cat_items}
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Node Shapes</div>
      {shape_legend}
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Edge Types</div>
      {type_legend}
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-size:11px;color:{th['legend_muted']};">
        Nodes: {G.number_of_nodes()} &nbsp;|&nbsp; Edges: {G.number_of_edges()}
        &nbsp;|&nbsp; Theme: {theme}
      </div>
    </div>
    """

    # ── 4. Full-screen CSS + edge-label shape override ───────────────────
    shape_js = ""
    if use_background and edge_label_style != "rectangle":
        shape_draw = {
            "roundrect": f"""
                var r = 6;
                ctx.beginPath();
                ctx.moveTo(x - w/2 + r, y - h/2);
                ctx.arcTo(x + w/2, y - h/2, x + w/2, y + h/2, r);
                ctx.arcTo(x + w/2, y + h/2, x - w/2, y + h/2, r);
                ctx.arcTo(x - w/2, y + h/2, x - w/2, y - h/2, r);
                ctx.arcTo(x - w/2, y - h/2, x + w/2, y - h/2, r);
                ctx.closePath(); ctx.fill(); ctx.stroke();""",
            "ellipse": """
                ctx.beginPath();
                ctx.ellipse(x, y, w/2, h/2, 0, 0, 2*Math.PI);
                ctx.fill(); ctx.stroke();""",
            "circle": """
                var rad = Math.max(w, h) / 2;
                ctx.beginPath();
                ctx.arc(x, y, rad, 0, 2*Math.PI);
                ctx.fill(); ctx.stroke();""",
            "diamond": """
                ctx.beginPath();
                ctx.moveTo(x, y - h/2);
                ctx.lineTo(x + w/2, y);
                ctx.lineTo(x, y + h/2);
                ctx.lineTo(x - w/2, y);
                ctx.closePath(); ctx.fill(); ctx.stroke();""",
        }
        draw_code = shape_draw.get(edge_label_style, "")
        if draw_code:
            shape_js = f"""
            <script>
            (function() {{
              var net = document.querySelector('#mynetwork');
              var observer = new MutationObserver(function() {{
                var canvas = net.querySelector('canvas');
                if (!canvas) return;
                observer.disconnect();
                var ctx = canvas.getContext('2d');
                var _fillRect = ctx.fillRect.bind(ctx);
                var pad = {edge_label_padding};
                ctx.fillRect = function(rx, ry, rw, rh) {{
                  if (rw < 300 && rh < 60) {{
                    var x = rx + rw/2, y = ry + rh/2;
                    var w = rw + pad*2, h = rh + pad*2;
                    ctx.save();
                    ctx.fillStyle = ctx.fillStyle;
                    ctx.strokeStyle = '{th["legend_border"]}';
                    ctx.lineWidth = 1;
                    {draw_code}
                    ctx.restore();
                  }} else {{
                    _fillRect(rx, ry, rw, rh);
                  }}
                }};
              }});
              observer.observe(net, {{childList: true, subtree: true}});
            }})();
            </script>
            """

    fullscreen_css = f"""
    <style>
      html, body {{ margin:0; padding:0; width:100%; height:100%;
                    overflow:hidden; background:{th['bg_color']}; }}
      #mynetwork {{ width:100vw !important; height:100vh !important;
                    border:none !important; margin:0 !important;
                    padding:0 !important; }}
    </style>
    """

    # ── 5. Save & inject ─────────────────────────────────────────────────
    net.save_graph(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("<head>", "<head>" + fullscreen_css, 1)
    html = html.replace("</body>", legend_html + shape_js + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Unified causal+event graph saved to: {output_file}")
    print(
        f"  Display mode: {display_mode}  |  Theme: {theme}"
        f"  |  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}"
    )
    print(f"  Event edges: {n_event}  |  Causal edges: {n_causal}  |  Attribute edges: {n_attr}")
    print(
        f"  Node size: {node_size}  |  Node font: {node_font_size}px  |  Edge font: {edge_font_size}px"
    )
    print(
        f"  Edge label style: {edge_label_style}"
        + (f"  |  Label padding: {edge_label_padding}px" if use_background else "")
    )
    print(f"  Categories: {', '.join(unique_cats)}")
    return output_file


# ═══════════════════════════════════════════════════════════════════════════
#  visualize_events_in_time_window
# ═══════════════════════════════════════════════════════════════════════════

def visualize_events_in_time_window(
    unified_df: pd.DataFrame,
    reference_date: Union[str, pd.Timestamp],
    days_back: int,
    output_file: str = "event_time_window.html",
    attribute_col: str = "attribute",
    # ── Visual options ──────────────────────────────────────────────────
    show_edge_labels: bool = True,
    theme: Literal["light", "dark", "pastel", "highcontrast"] = "light",
    node_size: int = 25,
    node_font_size: int = 14,
    edge_font_size: int = 12,
    edge_label_style: Literal[
        "text", "rectangle", "roundrect", "ellipse", "circle", "diamond"
    ] = "text",
    edge_label_padding: int = 5,
) -> tuple:
    """
    Filter the unified graph to event edges whose date falls within
    [reference_date - days_back, reference_date] and visualise them.

    Parameters
    ----------
    unified_df : pd.DataFrame
        Output of :func:`build_unified_graph`.
    reference_date : str or pd.Timestamp
        The anchor date (e.g. ``matched_row["Date"]``).  Strings are
        parsed automatically.
    days_back : int
        Number of days before ``reference_date`` to include.
    output_file : str
        Destination HTML path.
    theme, node_size, ... : same as :func:`visualize_unified_graph`.

    Returns
    -------
    tuple[str, pd.DataFrame]
        Path to the saved HTML file and the filtered DataFrame with a
        ``days_before`` column showing how many days each row is before
        the reference date.
    """

    # ── 0. Resolve dates ────────────────────────────────────────────────
    if isinstance(reference_date, pd.Series):
        reference_date = reference_date.iloc[0]
    ref_dt = pd.to_datetime(reference_date)
    start_dt = ref_dt - timedelta(days=days_back)

    # ── 1. Filter to event + attribute rows inside the window ───────────
    work_df = unified_df[unified_df["edge_type"].isin(["event", "attribute"])].copy()
    work_df["_parsed_date"] = pd.to_datetime(work_df["date"], errors="coerce")
    work_df = work_df[
        work_df["_parsed_date"].notna()
        & (work_df["_parsed_date"] >= start_dt)
        & (work_df["_parsed_date"] <= ref_dt)
    ].copy()

    if work_df.empty:
        print(
            f"No event edges found between {start_dt.date()} and {ref_dt.date()}. "
            "Try a larger days_back value."
        )
        return "", pd.DataFrame()

    # ── 2. Compute days-before-reference for colour grading ─────────────
    work_df["_days_before"] = (ref_dt - work_df["_parsed_date"]).dt.days

    th = KG_THEMES[theme]

    # ── 3. Build graph ──────────────────────────────────────────────────
    G = nx.DiGraph()
    event_nodes: set = set()
    attribute_nodes: set = set()
    node_category: dict = {}
    node_earliest_day: dict = {}   # track closest-to-ref date per node

    for _, row in work_df.iterrows():
        src, tgt = row["source"], row["target"]
        days_b = row["_days_before"]

        if row["edge_type"] == "attribute":
            event_nodes.add(src)
            attribute_nodes.add(tgt)
            G.add_edge(
                src, tgt,
                label=row["relation"],
                edge_type="attribute",
                days_before=days_b,
            )
            node_category[tgt] = tgt
        else:  # event
            event_nodes.add(src)
            G.add_edge(
                src, tgt,
                label=row["relation"],
                edge_type="event",
                days_before=days_b,
                date_str=str(row["date"]),
                title_str=str(row.get("title", "")),
            )
            if pd.notna(row.get(attribute_col)):
                cats = re.findall(r':\s*([^",]+)', str(row[attribute_col]))
                if cats:
                    node_category[src] = cats[0].strip()

        # Keep the most recent appearance per node
        for n in (src, tgt):
            node_earliest_day[n] = min(node_earliest_day.get(n, days_b), days_b)

    for n in attribute_nodes:
        node_category.setdefault(n, n)

    degree = dict(G.degree())
    unique_cats = sorted(set(node_category.values()))
    cat_color = {
        cat: th["node_colors"][i % len(th["node_colors"])]
        for i, cat in enumerate(unique_cats)
    }

    # ── 4. Colour-gradient helper (recent=vivid, older=faded) ───────────
    def _time_opacity(days_before: int) -> float:
        """Return 0.35 ... 1.0  (older -> more transparent)."""
        if days_back == 0:
            return 1.0
        return 1.0 - 0.65 * (days_before / days_back)

    def _rgba(hex_color: str, opacity: float) -> str:
        """Convert #RRGGBB + opacity -> rgba string."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{opacity:.2f})"

    # ── 5. Build Pyvis network ──────────────────────────────────────────
    net = Network(
        height="100%",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor=th["bg_color"],
        font_color=th["node_font_color"],
        cdn_resources="remote",
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01,
        damping=0.09,
    )

    _node_profiles = {
        "Event Entity": {
            "shape": "dot",
            "border": "#e67e22",
            "border_width": 3,
            "shadow_color": "rgba(230,126,34,0.45)",
        },
        "Attribute": {
            "shape": "diamond",
            "border": "#8e44ad",
            "border_width": 2,
            "shadow_color": "rgba(142,68,173,0.35)",
        },
        "Target": {
            "shape": "box",
            "border": "#2e7d32",
            "border_width": 3,
            "shadow_color": "rgba(46,125,50,0.45)",
        },
    }

    for node in G.nodes():
        cat = node_category.get(node, "Other")
        base_color = cat_color.get(cat, th["node_colors"][-1])
        days_b = node_earliest_day.get(node, 0)
        opacity = _time_opacity(days_b)

        if node in attribute_nodes:
            node_type = "Attribute"
        elif node in event_nodes:
            node_type = "Event Entity"
        else:
            node_type = "Target"

        prof = _node_profiles[node_type]
        net.add_node(
            node,
            label=node,
            shape=prof["shape"],
            title=(
                f"<b>{node}</b><br>"
                f"Category: {cat}<br>"
                f"Type: {node_type}<br>"
                f"Closest event: {days_b} day(s) before reference<br>"
                f"Connections: {degree.get(node, 0)}"
            ),
            color={
                "background": _rgba(base_color, opacity),
                "border": prof["border"],
                "highlight": {"background": base_color, "border": prof["border"]},
            },
            size=node_size,
            font={
                "size": node_font_size,
                "color": th["node_font_color"],
                "face": "Arial",
            },
            borderWidth=prof["border_width"],
            borderWidthSelected=prof["border_width"] + 2,
            shadow={
                "enabled": True,
                "color": prof["shadow_color"],
                "size": 10,
                "x": 0,
                "y": 0,
            },
        )

    # ── Edge-label font helper ──────────────────────────────────────────
    use_background = edge_label_style != "text"
    edge_rel_colors = th["edge_relation_colors"]

    def _edge_font(color: str) -> dict:
        font = {
            "size": edge_font_size,
            "color": color if not use_background else th["edge_font_color"],
            "face": "Arial",
            "strokeWidth": 0 if use_background else 3,
            "strokeColor": th["bg_color"],
            "align": "top",
        }
        if use_background:
            font["background"] = th["bg_color"]
        return font

    # ── Edges ───────────────────────────────────────────────────────────
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type")
        days_b = data.get("days_before", 0)
        opacity = _time_opacity(days_b)

        if etype == "attribute":
            e_color = _rgba("#8e44ad", opacity)
            edge_opts = dict(
                label=data["label"],
                title=f"{u}  -- has_attribute -->  {v}",
                color={"color": e_color, "highlight": "#8e44ad", "opacity": 0.9},
                arrows="to",
                width=2,
                font=_edge_font("#8e44ad"),
                smooth={"type": "curvedCW", "roundness": 0.15},
                dashes=[2, 4],
            )
        else:  # event
            rel = data["label"]
            date_str = data.get("date_str", "")
            title_str = data.get("title_str", "")
            base_e_color = edge_rel_colors.get(rel.lower(), th["default_edge"])
            e_color = _rgba(base_e_color, opacity)
            edge_opts = dict(
                label=f"{rel}  ({date_str}, -{days_b}d)",
                title=(
                    f"<b>{u} -- {rel} --> {v}</b><br>"
                    f"Date: {date_str}<br>"
                    f"Days before reference: {days_b}<br>"
                    f"{title_str}"
                ),
                color={"color": e_color, "highlight": base_e_color, "opacity": 0.9},
                arrows="to",
                width=max(1, 3 - int(2 * days_b / max(days_back, 1))),
                font=_edge_font(base_e_color),
                smooth={"type": "curvedCW", "roundness": 0.15},
                dashes=[5, 5],
            )

        if not show_edge_labels:
            edge_opts.pop("label", None)
            edge_opts.pop("font", None)
        net.add_edge(u, v, **edge_opts)

    # ── 6. Legend ───────────────────────────────────────────────────────
    n_event = sum(1 for _, _, d in G.edges(data=True) if d["edge_type"] == "event")
    n_attr = sum(1 for _, _, d in G.edges(data=True) if d["edge_type"] == "attribute")

    cat_items = "".join(
        f'<div style="margin:3px 0;"><span style="display:inline-block;'
        f"width:14px;height:14px;background:{cat_color[c]};border-radius:50%;"
        f'margin-right:8px;vertical-align:middle;"></span>'
        f'<span style="vertical-align:middle;">{c}</span></div>'
        for c in unique_cats
    )

    legend_html = f"""
    <div id="legend" style="position:fixed;top:12px;right:12px;
      background:{th['legend_bg']};padding:16px 20px;border-radius:10px;
      color:{th['legend_text']};font-family:Arial,sans-serif;font-size:13px;
      max-height:90vh;overflow-y:auto;z-index:9999;
      border:1px solid {th['legend_border']};
      box-shadow:0 2px 12px rgba(0,0,0,0.08);">
      <div style="font-weight:bold;font-size:15px;margin-bottom:6px;">
        Time Window</div>
      <div style="margin:3px 0;">
        Reference: <b>{ref_dt.date()}</b></div>
      <div style="margin:3px 0;">
        From: <b>{start_dt.date()}</b>  ({days_back} days back)</div>
      <div style="margin:3px 0;font-size:11px;color:{th['legend_muted']};">
        Vivid = recent &nbsp;|&nbsp; Faded = older</div>
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Node Categories</div>
      {cat_items}
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Node Shapes</div>
      <div style="margin:3px 0;">
        <span style="display:inline-block;width:14px;height:14px;
        border:3px solid #e67e22;border-radius:50%;margin-right:8px;
        vertical-align:middle;"></span>
        <span style="vertical-align:middle;">Event Entity</span></div>
      <div style="margin:3px 0;">
        <span style="display:inline-block;width:12px;height:12px;
        border:2px solid #8e44ad;transform:rotate(45deg);margin-right:8px;
        vertical-align:middle;"></span>
        <span style="vertical-align:middle;">Attribute</span></div>
      <div style="margin:3px 0;">
        <span style="display:inline-block;width:14px;height:14px;
        border:3px solid #2e7d32;margin-right:8px;
        vertical-align:middle;"></span>
        <span style="vertical-align:middle;">Target Feature</span></div>
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-weight:bold;font-size:15px;margin-bottom:10px;">
        Edges</div>
      <div style="margin:3px 0;">Event edges: {n_event}</div>
      <div style="margin:3px 0;">Attribute edges: {n_attr}</div>
      <hr style="border-color:{th['legend_hr']};margin:10px 0;">
      <div style="font-size:11px;color:{th['legend_muted']};">
        Nodes: {G.number_of_nodes()} &nbsp;|&nbsp; Edges: {G.number_of_edges()}
        &nbsp;|&nbsp; Theme: {theme}
      </div>
    </div>
    """

    # ── 7. Full-screen CSS ──────────────────────────────────────────────
    fullscreen_css = f"""
    <style>
      html, body {{ margin:0; padding:0; width:100%; height:100%;
                    overflow:hidden; background:{th['bg_color']}; }}
      #mynetwork {{ width:100vw !important; height:100vh !important;
                    border:none !important; margin:0 !important;
                    padding:0 !important; }}
    </style>
    """

    # ── 8. Save & inject ────────────────────────────────────────────────
    net.save_graph(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("<head>", "<head>" + fullscreen_css, 1)
    html = html.replace("</body>", legend_html + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Event time-window graph saved to: {output_file}")
    print(
        f"  Reference date: {ref_dt.date()}  |  Window: {days_back} days back  "
        f"(from {start_dt.date()})"
    )
    print(
        f"  Event edges: {n_event}  |  Attribute edges: {n_attr}  "
        f"|  Nodes: {G.number_of_nodes()}"
    )
    print(f"  Theme: {theme}")

    # ── 9. Build result DataFrame ───────────────────────────────────────
    result_df = work_df.drop(columns=["_parsed_date"]).rename(
        columns={"_days_before": "days_before"}
    ).sort_values("days_before").reset_index(drop=True)

    return output_file, result_df


# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC (interactive days_back) version
# ═════════════════════════════════════════════════════════════════════════════

def visualize_events_in_time_window_dynamic(
    unified_df: pd.DataFrame,
    reference_date: Union[str, pd.Timestamp],
    days_back: int = 35,
    output_file: str = "event_time_window_dynamic.html",
    attribute_col: str = "attribute",
    # ── Visual options ──────────────────────────────────────────────────
    show_edge_labels: bool = True,
    theme: Literal["light", "dark", "pastel", "highcontrast"] = "light",
    node_size: int = 25,
    node_font_size: int = 14,
    edge_font_size: int = 12,
    edge_label_style: Literal[
        "text", "rectangle", "roundrect", "ellipse", "circle", "diamond"
    ] = "text",
    edge_label_padding: int = 5,
) -> str:
    """
    Like :func:`visualize_events_in_time_window` but the resulting HTML page
    contains an input box that lets the viewer change **days_back** and
    redraw the graph in the browser without re-running Python.

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    import json as _json

    # ── 0. Resolve reference date ──────────────────────────────────────
    if isinstance(reference_date, pd.Series):
        reference_date = reference_date.iloc[0]
    ref_dt = pd.to_datetime(reference_date)
    ref_date_str = ref_dt.strftime("%Y-%m-%d")

    # ── 1. Prepare all event + attribute rows ──────────────────────────
    work_df = unified_df[unified_df["edge_type"].isin(["event", "attribute"])].copy()
    work_df["_parsed_date"] = pd.to_datetime(work_df["date"], errors="coerce")
    # Keep only rows with valid dates that are <= reference_date
    work_df = work_df[
        work_df["_parsed_date"].notna()
        & (work_df["_parsed_date"] <= ref_dt)
    ].copy()

    if work_df.empty:
        print("No event edges found before the reference date.")
        return ""

    # ── 2. Extract category info per source node ───────────────────────
    node_categories = {}
    for _, row in work_df.iterrows():
        if pd.notna(row.get(attribute_col)):
            cats = re.findall(r':\s*([^",]+)', str(row[attribute_col]))
            if cats:
                node_categories[row["source"]] = cats[0].strip()

    # ── 3. Serialize edges to JSON ─────────────────────────────────────
    edges_data = []
    for _, row in work_df.iterrows():
        edges_data.append({
            "source": str(row["source"]),
            "target": str(row["target"]),
            "relation": str(row["relation"]),
            "edge_type": str(row["edge_type"]),
            "date": str(row["date"]),
            "title": str(row.get("title", "")),
            "attribute": str(row.get(attribute_col, "")),
        })

    # ── 4. Serialize theme ─────────────────────────────────────────────
    th = KG_THEMES[theme]

    # ── 5. Build the full HTML page ────────────────────────────────────
    use_background = edge_label_style != "text"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Event Time Window (Dynamic)</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%;
                overflow:hidden; background:{th['bg_color']}; font-family:Arial,sans-serif; }}
  #mynetwork {{ width:100vw; height:100vh; }}
  #controls {{
    position:fixed; top:12px; left:12px; z-index:10000;
    background:{th['legend_bg']}; padding:14px 18px; border-radius:10px;
    border:1px solid {th['legend_border']};
    box-shadow:0 2px 12px rgba(0,0,0,0.10);
    color:{th['legend_text']}; font-size:14px;
  }}
  #controls label {{ font-weight:bold; margin-right:8px; }}
  #days-input {{
    width:70px; padding:6px 10px; font-size:14px; border-radius:6px;
    border:1px solid {th['legend_border']}; background:{th['bg_color']};
    color:{th['legend_text']}; text-align:center;
  }}
  #redraw-btn {{
    padding:6px 16px; font-size:14px; border-radius:6px; cursor:pointer;
    border:1px solid {th['legend_border']}; margin-left:6px;
    background:{th['node_colors'][0]}; color:#fff; font-weight:bold;
  }}
  #redraw-btn:hover {{ opacity:0.85; }}
  #status {{
    margin-top:8px; font-size:12px; color:{th['legend_muted']};
  }}
  #legend {{
    position:fixed; top:12px; right:12px; z-index:9999;
    background:{th['legend_bg']}; padding:16px 20px; border-radius:10px;
    color:{th['legend_text']}; font-size:13px;
    max-height:90vh; overflow-y:auto;
    border:1px solid {th['legend_border']};
    box-shadow:0 2px 12px rgba(0,0,0,0.08);
  }}
</style>
</head>
<body>

<div id="controls">
  <label for="days-input">Days back:</label>
  <input id="days-input" type="number" min="1" max="9999" value="{days_back}">
  <button id="redraw-btn">Redraw</button>
  <div style="margin-top:8px;">
    <label for="show-attr" style="font-weight:bold; cursor:pointer;">
      <input id="show-attr" type="checkbox" checked style="margin-right:6px; cursor:pointer; vertical-align:middle;">
      Show attributes
    </label>
  </div>
  <div id="status">Reference: {ref_date_str} &nbsp;|&nbsp; Window: {days_back} days</div>
</div>

<div id="legend"></div>
<div id="mynetwork"></div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────
const ALL_EDGES = {_json.dumps(edges_data)};
const NODE_CATEGORIES = {_json.dumps(node_categories)};
const REF_DATE = new Date("{ref_date_str}T00:00:00");
const THEME = {_json.dumps(th)};
const CONFIG = {{
  nodeSize: {node_size},
  nodeFontSize: {node_font_size},
  edgeFontSize: {edge_font_size},
  showEdgeLabels: {str(show_edge_labels).lower()},
  useBackground: {str(use_background).lower()},
  edgeLabelStyle: "{edge_label_style}",
}};

const NODE_PROFILES = {{
  "Event Entity": {{ shape:"dot", border:"#e67e22", borderWidth:3, shadowColor:"rgba(230,126,34,0.45)" }},
  "Attribute":    {{ shape:"diamond", border:"#8e44ad", borderWidth:2, shadowColor:"rgba(142,68,173,0.35)" }},
  "Target":       {{ shape:"box", border:"#2e7d32", borderWidth:3, shadowColor:"rgba(46,125,50,0.45)" }},
}};

let network = null;

function daysBetween(d1, d2) {{
  return Math.round((d2 - d1) / (1000 * 60 * 60 * 24));
}}

function hexToRgba(hex, opacity) {{
  const h = hex.replace("#","");
  const r = parseInt(h.substring(0,2),16);
  const g = parseInt(h.substring(2,4),16);
  const b = parseInt(h.substring(4,6),16);
  return "rgba("+r+","+g+","+b+","+opacity.toFixed(2)+")";
}}

function timeOpacity(daysBefore, daysBack) {{
  if (daysBack === 0) return 1.0;
  return 1.0 - 0.65 * (daysBefore / daysBack);
}}

function buildGraph(daysBack, showAttributes) {{
  const startDate = new Date(REF_DATE);
  startDate.setDate(startDate.getDate() - daysBack);

  // Filter edges by time window, then optionally exclude attribute edges
  const filtered = ALL_EDGES.filter(e => {{
    const d = new Date(e.date + "T00:00:00");
    if (isNaN(d) || d < startDate || d > REF_DATE) return false;
    if (!showAttributes && e.edge_type === "attribute") return false;
    return true;
  }});

  if (filtered.length === 0) {{
    document.getElementById("status").innerHTML =
      "No events found in window. Try a larger value.";
    if (network) {{ network.destroy(); network = null; }}
    document.getElementById("legend").innerHTML = "";
    return;
  }}

  // Build node/edge sets
  const eventNodes = new Set();
  const attrNodes = new Set();
  const nodeCategory = {{}};
  const nodeEarliestDay = {{}};
  const visNodes = [];
  const visEdges = [];
  const addedNodes = new Set();

  // Assign categories from pre-computed + attribute edges
  Object.assign(nodeCategory, NODE_CATEGORIES);

  filtered.forEach(e => {{
    const d = new Date(e.date + "T00:00:00");
    const db = daysBetween(d, REF_DATE);

    if (e.edge_type === "attribute") {{
      eventNodes.add(e.source);
      attrNodes.add(e.target);
      nodeCategory[e.target] = e.target;
    }} else {{
      eventNodes.add(e.source);
    }}

    [e.source, e.target].forEach(n => {{
      if (!(n in nodeEarliestDay) || db < nodeEarliestDay[n])
        nodeEarliestDay[n] = db;
    }});
  }});

  // Unique categories & color map
  const allNodes = new Set([...eventNodes, ...attrNodes]);
  filtered.forEach(e => {{ allNodes.add(e.source); allNodes.add(e.target); }});

  const uniqueCats = [...new Set(Object.values(nodeCategory))].sort();
  const catColor = {{}};
  uniqueCats.forEach((c, i) => {{
    catColor[c] = THEME.node_colors[i % THEME.node_colors.length];
  }});

  // Compute degree
  const degree = {{}};
  filtered.forEach(e => {{
    degree[e.source] = (degree[e.source]||0) + 1;
    degree[e.target] = (degree[e.target]||0) + 1;
  }});

  // Add nodes
  allNodes.forEach(node => {{
    if (addedNodes.has(node)) return;
    addedNodes.add(node);

    const cat = nodeCategory[node] || "Other";
    const baseColor = catColor[cat] || THEME.node_colors[THEME.node_colors.length-1];
    const db = nodeEarliestDay[node] || 0;
    const opacity = timeOpacity(db, daysBack);

    let nodeType = "Target";
    if (attrNodes.has(node)) nodeType = "Attribute";
    else if (eventNodes.has(node)) nodeType = "Event Entity";

    const prof = NODE_PROFILES[nodeType];
    visNodes.push({{
      id: node, label: node, shape: prof.shape,
      title: "<b>"+node+"</b><br>Category: "+cat+"<br>Type: "+nodeType+
             "<br>Closest event: "+db+" day(s) before reference<br>Connections: "+(degree[node]||0),
      color: {{
        background: hexToRgba(baseColor, opacity),
        border: prof.border,
        highlight: {{ background: baseColor, border: prof.border }},
      }},
      size: CONFIG.nodeSize,
      font: {{ size: CONFIG.nodeFontSize, color: THEME.node_font_color, face: "Arial" }},
      borderWidth: prof.borderWidth,
      borderWidthSelected: prof.borderWidth + 2,
      shadow: {{ enabled:true, color:prof.shadowColor, size:10, x:0, y:0 }},
    }});
  }});

  // Add edges
  let nEvent = 0, nAttr = 0;
  filtered.forEach(e => {{
    const d = new Date(e.date + "T00:00:00");
    const db = daysBetween(d, REF_DATE);
    const opacity = timeOpacity(db, daysBack);

    let edgeOpts;
    if (e.edge_type === "attribute") {{
      nAttr++;
      const eColor = hexToRgba("#8e44ad", opacity);
      edgeOpts = {{
        from: e.source, to: e.target,
        label: CONFIG.showEdgeLabels ? e.relation : undefined,
        title: e.source + " -- has_attribute --> " + e.target,
        color: {{ color: eColor, highlight: "#8e44ad", opacity: 0.9 }},
        arrows: "to", width: 2,
        smooth: {{ type: "curvedCW", roundness: 0.15 }},
        dashes: [2, 4],
      }};
      if (CONFIG.showEdgeLabels) {{
        edgeOpts.font = {{
          size: CONFIG.edgeFontSize,
          color: CONFIG.useBackground ? THEME.edge_font_color : "#8e44ad",
          face: "Arial", strokeWidth: CONFIG.useBackground ? 0 : 3,
          strokeColor: THEME.bg_color, align: "top",
        }};
        if (CONFIG.useBackground) edgeOpts.font.background = THEME.bg_color;
      }}
    }} else {{
      nEvent++;
      const rel = e.relation;
      const relColors = THEME.edge_relation_colors || {{}};
      const baseEColor = relColors[rel.toLowerCase()] || THEME.default_edge;
      const eColor = hexToRgba(baseEColor, opacity);
      edgeOpts = {{
        from: e.source, to: e.target,
        label: CONFIG.showEdgeLabels ? rel + "  (" + e.date + ", -" + db + "d)" : undefined,
        title: "<b>"+e.source+" -- "+rel+" --> "+e.target+"</b><br>Date: "+e.date+
               "<br>Days before reference: "+db+"<br>"+e.title,
        color: {{ color: eColor, highlight: baseEColor, opacity: 0.9 }},
        arrows: "to", width: Math.max(1, 3 - Math.floor(2 * db / Math.max(daysBack, 1))),
        smooth: {{ type: "curvedCW", roundness: 0.15 }},
        dashes: [5, 5],
      }};
      if (CONFIG.showEdgeLabels) {{
        edgeOpts.font = {{
          size: CONFIG.edgeFontSize,
          color: CONFIG.useBackground ? THEME.edge_font_color : baseEColor,
          face: "Arial", strokeWidth: CONFIG.useBackground ? 0 : 3,
          strokeColor: THEME.bg_color, align: "top",
        }};
        if (CONFIG.useBackground) edgeOpts.font.background = THEME.bg_color;
      }}
    }}
    visEdges.push(edgeOpts);
  }});

  // ── Render ──────────────────────────────────────────────────────────
  const container = document.getElementById("mynetwork");
  if (network) network.destroy();

  const data = {{
    nodes: new vis.DataSet(visNodes),
    edges: new vis.DataSet(visEdges),
  }};
  const options = {{
    physics: {{
      barnesHut: {{
        gravitationalConstant: -8000,
        centralGravity: 0.3,
        springLength: 200,
        springConstant: 0.01,
        damping: 0.09,
      }},
    }},
  }};
  network = new vis.Network(container, data, options);

  // ── Update legend ───────────────────────────────────────────────────
  const sDate = new Date(REF_DATE);
  sDate.setDate(sDate.getDate() - daysBack);
  const catItems = uniqueCats.map(c =>
    '<div style="margin:3px 0;"><span style="display:inline-block;width:14px;height:14px;'+
    'background:'+catColor[c]+';border-radius:50%;margin-right:8px;vertical-align:middle;"></span>'+
    '<span style="vertical-align:middle;">'+c+'</span></div>'
  ).join("");

  document.getElementById("legend").innerHTML =
    '<div style="font-weight:bold;font-size:15px;margin-bottom:6px;">Time Window</div>'+
    '<div style="margin:3px 0;">Reference: <b>{ref_date_str}</b></div>'+
    '<div style="margin:3px 0;">From: <b>'+sDate.toISOString().slice(0,10)+'</b>  ('+daysBack+' days back)</div>'+
    '<div style="margin:3px 0;font-size:11px;color:{th["legend_muted"]};">Vivid = recent | Faded = older</div>'+
    '<hr style="border-color:{th["legend_hr"]};margin:10px 0;">'+
    '<div style="font-weight:bold;font-size:15px;margin-bottom:10px;">Node Categories</div>'+
    catItems+
    '<hr style="border-color:{th["legend_hr"]};margin:10px 0;">'+
    '<div style="font-weight:bold;font-size:15px;margin-bottom:10px;">Node Shapes</div>'+
    '<div style="margin:3px 0;"><span style="display:inline-block;width:14px;height:14px;border:3px solid #e67e22;border-radius:50%;margin-right:8px;vertical-align:middle;"></span><span style="vertical-align:middle;">Event Entity</span></div>'+
    '<div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;border:2px solid #8e44ad;transform:rotate(45deg);margin-right:8px;vertical-align:middle;"></span><span style="vertical-align:middle;">Attribute</span></div>'+
    '<div style="margin:3px 0;"><span style="display:inline-block;width:14px;height:14px;border:3px solid #2e7d32;margin-right:8px;vertical-align:middle;"></span><span style="vertical-align:middle;">Target Feature</span></div>'+
    '<hr style="border-color:{th["legend_hr"]};margin:10px 0;">'+
    '<div style="font-weight:bold;font-size:15px;margin-bottom:10px;">Edges</div>'+
    '<div style="margin:3px 0;">Event edges: '+nEvent+'</div>'+
    '<div style="margin:3px 0;">Attribute edges: '+nAttr+'</div>'+
    '<hr style="border-color:{th["legend_hr"]};margin:10px 0;">'+
    '<div style="font-size:11px;color:{th["legend_muted"]};">Nodes: '+visNodes.length+' | Edges: '+visEdges.length+' | Theme: {theme}</div>';

  document.getElementById("status").innerHTML =
    "Reference: {ref_date_str} &nbsp;|&nbsp; Window: " + daysBack +
    " days &nbsp;|&nbsp; Nodes: " + visNodes.length +
    " &nbsp;|&nbsp; Edges: " + visEdges.length;
}}

// ── Helper to read current controls ─────────────────────────────────
function redraw() {{
  const val = parseInt(document.getElementById("days-input").value, 10);
  const showAttr = document.getElementById("show-attr").checked;
  if (!isNaN(val) && val > 0) buildGraph(val, showAttr);
}}

// ── Initial draw ────────────────────────────────────────────────────
buildGraph({days_back}, true);

// ── Event listeners ─────────────────────────────────────────────────
document.getElementById("redraw-btn").addEventListener("click", redraw);
document.getElementById("days-input").addEventListener("keydown", function(e) {{
  if (e.key === "Enter") redraw();
}});
document.getElementById("show-attr").addEventListener("change", redraw);
</script>
</body>
</html>"""

    # ── 6. Write HTML ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Dynamic event time-window graph saved to: {output_file}")
    print(f"  Reference date: {ref_dt.date()}  |  Default window: {days_back} days")
    print(f"  Theme: {theme}")

    return output_file


# def generate_explanation(
#     event_triplets_df: pd.DataFrame,
#     causal_relations_df: pd.DataFrame,
#     openrouter_api_key: str,
#     llm_name: str = "gpt-5-mini",
# ) -> str:
#     """
#     Generate a plain-language explanation for supply chain users by combining
#     event knowledge-graph triplets with macroeconomic causal relationships.

#     Parameters
#     ----------
#     event_triplets_df : pd.DataFrame
#         DataFrame with columns ["source", "relation", "target"] representing
#         event triplets extracted from the knowledge graph.
#     causal_relations_df : pd.DataFrame
#         DataFrame with columns ["source", "target", "sign"] where sign is
#         "positive" or "negative", representing causal links between
#         macroeconomic features.
#     openrouter_api_key : str
#         API key for OpenRouter.
#     llm_name : str, default "gpt-5-mini"
#         Model name to use (without the "openai/" prefix).

#     Returns
#     -------
#     str
#         A plain-language explanation suitable for supply chain stakeholders.
#     """
#     from openai import OpenAI

#     client = OpenAI(
#         api_key=openrouter_api_key,
#         base_url="https://openrouter.ai/api/v1",
#     )
#     openai_model = f"openai/{llm_name}"

#     # ── Build event summary ──────────────────────────────────────────────
#     event_lines = []
#     for _, row in event_triplets_df.iterrows():
#         event_lines.append(f"- {row['source']} → {row['relation']} → {row['target']}")
#     event_block = "\n".join(event_lines) if event_lines else "No event information available."

#     # ── Build causal summary ─────────────────────────────────────────────
#     causal_lines = []
#     for _, row in causal_relations_df.iterrows():
#         causal_lines.append(f"- {row['source']} has a {row['sign']} effect on {row['target']}")
#     causal_block = "\n".join(causal_lines) if causal_lines else "No causal relationships available."

#     # ── Prompt ───────────────────────────────────────────────────────────
#     system_prompt = (
#         "You are a supply chain analyst who explains complex economic events "
#         "and data relationships in plain, easy-to-understand language. "
#         "Your audience is supply chain managers and business decision-makers "
#         "who are not data scientists. Avoid technical jargon. "
#         "Use short sentences and concrete examples where possible."
#     )

#     user_prompt = f"""Below you are given two pieces of information about recent supply chain and macroeconomic conditions.

# **Recent Events**
# These are events extracted from news and reports, shown as (subject → relationship → object):
# {event_block}

# **Macroeconomic Causal Relationships**
# These describe how key economic indicators influence each other. A "positive" effect means they move in the same direction; a "negative" effect means they move in opposite directions:
# {causal_block}

# Based on the information above, write a clear and concise explanation (a few paragraphs) for a supply chain professional that:
# 1. Summarizes what recent events have occurred and why they matter for supply chains.
# 2. Explains how the macroeconomic indicators are connected and what that means in practice.
# 3. Highlights any potential risks or opportunities a supply chain manager should pay attention to.

# Keep the language simple and actionable."""

#     response = client.chat.completions.create(
#         model=openai_model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         temperature=0.4,
#     )

#     return response.choices[0].message.content




def generate_explanation(
    event_triplets_df: pd.DataFrame, 
    causal_relations_df: pd.DataFrame, 
    openrouter_api_key: str, 
    row_number: int,
    max_words: Union[int, str] = 150,     # UPDATED: Now accepts int or str
    llm_name: str = "openai/gpt-4o",
) -> str:
    """
    Generate a plain-language explanation for supply chain users by combining
    event knowledge-graph triplets with macroeconomic causal relationships.

    Parameters
    ----------
    event_triplets_df : pd.DataFrame
        DataFrame with columns ["source", "relation", "target"] representing
        event triplets extracted from the knowledge graph.
    causal_relations_df : pd.DataFrame
        DataFrame with columns ["source", "target", "sign"] where sign is
        "positive" or "negative", representing causal links between
        macroeconomic features.
    openrouter_api_key : str
        API key for OpenRouter.
    row_number : int
        The current row number, used for saving the output JSON.
    max_words : Union[int, str], default 150
        The maximum desired word count. Pass "no_limit" for unrestricted length.
    llm_name : str, default "openai/gpt-4o"
        Model name to use.

    Returns
    -------
    str
        A plain-language explanation suitable for supply chain stakeholders.
    """

    from datetime import datetime

    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    openai_model = f"{llm_name}"

    # ── Build event summary ──────────────────────────────────────────────
    event_lines = []
    for _, row in event_triplets_df.iterrows():
        event_lines.append(f"- {row['source']} → {row['relation']} → {row['target']}")
    event_block = "\n".join(event_lines) if event_lines else "No event information available."

    # ── Build causal summary ─────────────────────────────────────────────
    causal_lines = []
    for _, row in causal_relations_df.iterrows():
        causal_lines.append(f"- {row['source']} has a {row['sign']} effect on {row['target']}")
    causal_block = "\n".join(causal_lines) if causal_lines else "No causal relationships available."

    # ── Prompt ───────────────────────────────────────────────────────────
    system_prompt = (
        "You are a supply chain analyst who explains complex economic events "
        "and data relationships in plain, easy-to-understand language. "
        "Your audience is supply chain managers and business decision-makers "
        "who are not data scientists. Avoid technical jargon. "
        "Use short sentences and concrete examples where possible."
    )

    # UPDATED: Conditional logic for the length constraint
    if max_words == "no_limit":
        length_instruction = "CRITICAL INSTRUCTION: Keep the language simple and actionable."
    else:
        length_instruction = f"CRITICAL INSTRUCTION: Keep the language simple and actionable. Strictly limit your entire response to a maximum of {max_words} words."

    user_prompt = f"""Below you are given two pieces of information about recent supply chain and macroeconomic conditions.

**Recent Events**
These are events extracted from news and reports, shown as (subject → relationship → object):
{event_block}

**Macroeconomic Causal Relationships**
These describe how key economic indicators influence each other. A "positive" effect means they move in the same direction; a "negative" effect means they move in opposite directions:
{causal_block}

Based on the information above, write a clear and concise explanation for a supply chain professional that:
1. Summarizes what recent events have occurred and why they matter for supply chains.
2. Explains how the macroeconomic indicators are connected and what that means in practice.
3. Highlights any potential risks or opportunities a supply chain manager should pay attention to.

{length_instruction}"""

    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    
    explanation = response.choices[0].message.content

    # ── Save Response to JSON ────────────────────────────────────────────
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    save_dir = "../llm_response/"
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"response_{row_number}_{date_str}_{time_str}.json"
    file_path = os.path.join(save_dir, file_name)
    
    json_data = {
        "row_number": row_number,
        "max_words_requested": max_words,
        "date_generated": date_str,
        "time_generated": time_str,
        "explanation": explanation
    }
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    return explanation



