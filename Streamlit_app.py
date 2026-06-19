import numpy as np
import scipy.stats as stats
import streamlit as st

# ==============================================================================
# TARGET LEAGUES & INTERNATIONAL ISOLATION (SECTION A LIST)
# ==============================================================================
TARGET_LEAGUES = {
    "England - Premier League": 8,
    "England - Championship": 9,
    "Spain - La Liga": 564,
    "Spain - Segunda Division": 567,
    "Germany - Bundesliga": 82,
    "Germany - 2. Bundesliga": 85,
    "Italy - Serie A": 384,
    "Italy - Serie B": 387,
    "France - Ligue 1": 301,
    "France - Ligue 2": 304,
    "USA - MLS": 242,
    "Sweden - Allsvenskan": 269,
    "Sweden - Superettan": 272,
    "Netherlands - Eredivisie": 600,
    "Netherlands - Eerste Divisie": 603,
    "Portugal - Liga Portugal": 462,
    "Portugal - Liga Portugal 2": 465,
    # --- INTERNATIONAL TOURNAMENTS ADDED ---
    "FIFA World Cup": 732,
    "UEFA European Championship": 24,
    "Copa America": 27,
}

# ==============================================================================
# WEBSITE INTERFACE DESIGN (OWLSMAN BRANDING ARCHITECTURE)
# ==============================================================================
st.set_page_config(page_title="OWLSMAN Engine", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: #ffffff; }
    .main-header { font-family: 'Arial', sans-serif; font-weight: 900; color: #10b981; text-align: center; margin-bottom: 25px; }
    .fixture-container { background-color: #1f2937; padding: 20px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 20px; }
    .market-badge { background-color: #047857; color: #ffffff; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 12px; margin-right: 5px; }
    .fade-badge { background-color: #b91c1c; color: #ffffff; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 12px; margin-right: 5px; }
    </style>
    """,
    unsafe_html=True,
)

# ==============================================================================
# THE EXACT STEP-BY-STEP FORMULA ENGINE (SECTION B RULES)
# ==============================================================================
def execute_exact_owlsman_system(home, away):
    # --- PHASE 1: INITIAL CALCULATIONS (DYNAMIC INVERSION SYSTEM) ---
    # Strictly follows your math: (Goals ÷ Shots) * 100. 
    # If over 100, do 100 ÷ Index. If under 100, do (1 ÷ Index) * 100.
    
    h_idx_score = (home["avg_goals_scored"] / home["avg_sot_for"]) * 100
    h_calc_scored = (100 / h_idx_score) if h_idx_score > 100 else (1 / h_idx_score * 100)
    
    h_idx_concede = (home["avg_goals_conceded"] / home["avg_sot_against"]) * 100
    h_calc_conceded = (100 / h_idx_concede) if h_idx_concede > 100 else (1 / h_idx_concede * 100)
    
    a_idx_score = (away["avg_goals_scored"] / away["avg_sot_for"]) * 100
    a_calc_scored = (100 / a_idx_score) if a_idx_score > 100 else (1 / a_idx_score * 100)
    
    a_idx_concede = (away["avg_goals_conceded"] / away["avg_sot_against"]) * 100
    a_calc_conceded = (100 / a_idx_concede) if a_idx_concede > 100 else (1 / a_idx_concede * 100)

    # --- PHASE 2: NET SCORES & SCORE POWER ---
    # Team A Net = Team A Scored ÷ Team B Conceded
    h_net_score = h_calc_scored / a_calc_conceded
    # Team A Score Power = (Team A Avg Goals ÷ Team B Avg Conceded) ÷ Team A Net * Team A Avg Shots
    h_score_power = ((home["avg_goals_scored"] / away["avg_goals_conceded"]) / h_net_score) * home["avg_sot_for"]
    
    # Team B Net = Team B Scored ÷ Team A Conceded
    a_net_score = a_calc_scored / h_calc_conceded
    # Team B Score Power = (Team B Avg Goals ÷ Team A Avg Conceded) ÷ Team B Net * Team B Avg Shots
    a_score_power = ((away["avg_goals_scored"] / home["avg_goals_conceded"]) / a_net_score) * away["avg_sot_for"]

    # --- PHASE 3: STRENGTH BLENDING ---
    # Team Strength = (xG + Big Chances) ÷ 2 * Score Power
    # No external baselines or unauthorized steps added.
    home_strength = ((home["avg_xg_for"] + home["avg_bc_scored"]) / 2) * h_score_power
    away_strength = ((away["avg_xg_for"] + away["avg_bc_scored"]) / 2) * a_score_power

    # --- PHASE 4: POISSON GRID GENERATION (MAX 6 CAP) ---
    h_matrix = [stats.poisson.pmf(k, home_strength) for k in range(7)]
    a_matrix = [stats.poisson.pmf(k, away_strength) for k in range(7)]
    grid = np.outer(h_matrix, a_matrix)

    p_home = float(np.sum(np.tril(grid, -1)))
    p_draw = float(np.sum(np.diag(grid)))
    p_away = float(np.sum(np.triu(grid, 1)))
    
    total_p = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total_p, p_draw / total_p, p_away / total_p

    # Extract top 2 most likely scorelines from grid matrix cells
    flat_grid = grid.flatten()
    top_two_indices = np.argsort(flat_grid)[::-1][:2]
    correct_scores = []
    for idx in top_two_indices:
        h_g, a_g = np.unravel_index(idx, grid.shape)
        correct_scores.append(f"{h_g}-{a_g}")

    # --- PHASE 5: 5-ODDS STRESS MATRIX & DOMINANCE INDEX ---
    σ = 0.036
    states = {}
    states["State 0"] = {"H": p_home, "D": p_draw, "A": p_away}

    def apply_variance_drag(target, b1, b2):
        boosted = (target + 0.50) - σ
        norm = boosted + b1 + b2
        return boosted / norm, b1 / norm, b2 / norm

    h1, d1, a1 = apply_variance_drag(p_home, p_draw, p_away)
    states["State 1"] = {"H": h1, "D": d1, "A": a1}

    a2, h2, d2 = apply_variance_drag(p_away, p_home, p_draw)
    states["State 2"] = {"H": h2, "D": d2, "A": a2}

    d3, h3, a3 = apply_variance_drag(p_draw, p_home, p_away)
    states["State 3"] = {"H": h3, "D": d3, "A": a3}

    h4_r, d4_r, a4_r = (p_home + 0.5 - σ), (p_draw + 0.5 - σ), (p_away + 0.5 - σ)
    t4 = h4_r + d4_r + a4_r
    states["State 4"] = {"H": h4_r / t4, "D": d4_r / t4, "A": a4_r / t4}

    dom_home = sum([states[s]["H"] - states["State 0"]["H"] for s in states])
    dom_away = sum([states[s]["A"] - states["State 0"]["A"] for s in states])

    blend_h = np.mean([states[s]["H"] for s in states])
    blend_d = np.mean([states[s]["D"] for s in states])
    blend_a = np.mean([states[s]["A"] for s in states])

    return {
        "baseline_odds": [1 / p_home, 1 / p_draw, 1 / p_away],
        "blended_odds": [1 / blend_h, 1 / blend_d, 1 / blend_a],
        "blended_probs": [blend_h, blend_d, blend_a],
        "dominance": [dom_home, dom_away],
        "correct_scores": correct_scores,
    }

# ==============================================================================
# MONDAY BATCH PIPELINE STREAM SIMULATION
# ==============================================================================
def get_weekly_fixtures_stream():
    return [
        {
            "id": 994812,
            "league": "FIFA World Cup",
            "home": "Argentina",
            "away": "France",
            "bookie_odds": [2.15, 3.20, 3.40],
            "home_stats": {
                "avg_goals_scored": 2.1, "avg_sot_for": 5.4,
                "avg_goals_conceded": 0.8, "avg_sot_against": 2.9,
                "avg_bc_scored": 0.62, "avg_xg_for": 1.85
            },
            "away_stats": {
                "avg_goals_scored": 1.9, "avg_sot_for": 4.8,
                "avg_goals_conceded": 1.1, "avg_sot_against": 3.2,
                "avg_bc_scored": 0.55, "avg_xg_for": 1.68
            }
        }
    ]

# ==============================================================================
# FRONTEND INTERFACE GENERATION BLOCK (EXPLICIT OWLSMAN TITLE LOCKED)
# ==============================================================================
st.markdown("<h1 class='main-header'>🦉 OWLSMAN</h1>", unsafe_html=True)
st.sidebar.markdown("### 📅 Execution Panel")
loop_mode = st.sidebar.selectbox("Loop Mode", ["7-Day Accumulator (Mon-Sun)"])
league_select = st.sidebar.selectbox("Isolated League Pool", ["All Active Targets"] + list(TARGET_LEAGUES.keys()))

fixtures = get_weekly_fixtures_stream()

for f in fixtures:
    if league_select != "All Active Targets" and f["league"] != league_select:
        continue

    res = execute_exact_owlsman_system(f["home_stats"], f["away_stats"])

    with st.container():
        st.markdown("<div class='fixture-container'>", unsafe_html=True)
        st.markdown(f"### 🏟️ {f['league']} | {f['home']} vs {f['away']} `[ID: #{f['id']}]`")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("#### 🏛️ Market Lines & Filters")
            st.write(f"**Bookie odds:** H: {f['bookie_odds'][0]} | X: {f['bookie_odds'][1]} | A: {f['bookie_odds'][2]}")
            
            st.markdown("**System Alerts:**")
            alert_html = ""
            if res["blended_probs"][0] > 0.55 and res["dominance"][0] >= 0.00:
                alert_html += "<span class='market-badge'>MAIN ASIAN HANDICAP</span>"
            if res["blended_probs"][0] > 0.55 and res["dominance"][0] < -0.25:
                alert_html += "<span class='fade-badge'>FADE FAVORITE (X2)</span>"
            if res["blended_odds"][0] < 2.10 and res["blended_odds"][2] < 2.10:
                alert_html += "<span class='market-badge'>BTTS: YES</span>"
            alert_html += "<span class='market-badge'>DOUBLE CHANCE 12</span>"
            st.markdown(alert_html, unsafe_html=True)
            
        with c2:
            st.markdown("#### 🔮 Master Blended Odds")
            st.write(f"True Home Odd: **{res['blended_odds'][0]:.2f}** ({res['blended_probs'][0]*100:.1f}%)")
            st.write(f"True Draw Odd: **{res['blended_odds'][1]:.2f}** ({res['blended_probs'][1]*100:.1f}%)")
            st.write(f"True Away Odd: **{res['blended_odds'][2]:.2f}** ({res['blended_probs'][2]*100:.1f}%)")
            st.write(f"🎯 **Correct Scores:** `{res['correct_scores'][0]}` or `{res['correct_scores'][1]}`")
            
        with c3:
            st.markdown("#### ⚡ Stress & Dominance")
            st.write(f"Odd 0 (Baseline Home): `{res['baseline_odds'][0]:.2f}`")
            h_c = "#10b981" if res["dominance"][0] >= 0 else "#ef4444"
            a_c = "#10b981" if res["dominance"][1] >= 0 else "#ef4444"
            st.markdown(f"Home Dominance: <span style='color:{h_c};font-weight:bold;'>{res['dominance'][0]:.2f}</span>", unsafe_html=True)
            st.markdown(f"Away Dominance: <span style='color:{a_c};font-weight:bold;'>{res['dominance'][1]:.2f}</span>", unsafe_html=True)

        st.markdown("</div>", unsafe_html=True)
