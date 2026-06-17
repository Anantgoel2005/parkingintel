"""
ParkingIntel — AI-Driven Parking Hotspot Intelligence
Theme 1: Poor Visibility on Parking-Induced Congestion
"""

import streamlit as st
import folium
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_clean
from src.analytics import (
    station_summary, peak_hour_profile, vehicle_fingerprint,
    weekday_vs_weekend, enforcement_gap_analysis,
)
from src.clustering import get_hotspots, grid_aggregation
from src.blindspots import combined_blind_spot_score, vehicle_diversity_blindspots, time_coverage_blindspots, junction_density_blindspots
from src.visualization import (
    create_base_map, add_violation_heatmap, add_hotspot_markers,
    plot_station_scorecard, plot_hourly_heatmap, plot_vehicle_pie,
    plot_weekend_scatter, plot_hourly_bar, plot_blindspot_ranking,
)
from src.chatbot import render_chat
from src.planner import predict_hotspots as plan_hotspots, suggest_patrol_order, format_plan

st.set_page_config(page_title="ParkingIntel", page_icon="🚗", layout="wide")

# ── Mobile-Responsive CSS ───────────────────────────────
st.markdown('''
<style>
[data-testid="stMetric"] { border-radius:12px; padding:12px; transition:transform .2s; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 8px 25px rgba(255,107,53,.15); }
.stTabs [role="tab"]:hover { color:#ff6b35!important; }
::-webkit-scrollbar{width:8px} ::-webkit-scrollbar-track{background:#1a1a2e} ::-webkit-scrollbar-thumb{background:#ff6b35;border-radius:4px}

/* Planner tab — prominent styling */
div[data-testid="stTabs"] button[data-baseweb="tab"]:first-child {
    font-weight: 700; font-size: 15px;
    border-bottom: 2px solid #ff6b35 !important;
}
div[data-baseweb="tab-panel"]:first-child h2 {
    color: #ff6b35; font-size: 1.8rem;
}
/* Primary button glow */
button[kind="primary"] {
    box-shadow: 0 0 15px rgba(255,107,53,0.4);
    transition: all 0.2s;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 25px rgba(255,107,53,0.6);
    transform: scale(1.02);
}
</style>
''', unsafe_allow_html=True)

# ── Load Data ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_data():
    df = load_clean()
    if "violation_list" in df.columns:
        df = df.drop(columns=["violation_list"])
    return df

@st.cache_data(ttl=3600)
def get_summary(df):
    return station_summary(df)

@st.cache_data(ttl=3600)
def get_hotspots_cached(df):
    return get_hotspots(df)

@st.cache_data(ttl=3600)
def get_blindspots(df):
    return combined_blind_spot_score(df)

@st.cache_data(ttl=3600)
def get_weekend_data(df):
    return weekday_vs_weekend(df)

with st.spinner("Loading data..."):
    df = get_data()

# ── Sidebar: Navigation + Filters ───────────────────────
st.sidebar.title("🚗 ParkingIntel")

# No sidebar nav — tabs will be at top

# --- Filters (collapsible) ---
with st.sidebar.expander("🔽 Filters", expanded=False):
    all_stations = sorted(df["police_station"].dropna().unique())
    selected_stations = st.multiselect(
        "Stations", options=all_stations,
        default=[s for s in all_stations if s != "No Police Station"][:5],
    )
    if "No Police Station" in all_stations:
        st.caption("No Police Station = highway / mobile units (339 records)")

    time_buckets = {
        "Late Night (12-4AM)": (0,4), "Early AM (4-8AM)": (4,8),
        "Morning (8-12PM)": (8,12), "Afternoon (12-4PM)": (12,16),
        "Evening (4-8PM)": (16,20), "Night (8-12AM)": (20,24),
    }
    selected_buckets = st.multiselect("Time", list(time_buckets.keys()), default=list(time_buckets.keys()))
    day_options = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    selected_days = st.multiselect("Days", range(7), default=list(range(7)), format_func=lambda x: day_options[x])
    all_vehicles = sorted(df["vehicle_type"].dropna().unique())
    selected_vehicles = st.multiselect("Vehicles", all_vehicles, default=all_vehicles[:6])

# Apply filters
hour_ranges = [time_buckets[b] for b in selected_buckets]
st_mask = df["police_station"].isin(selected_stations) if selected_stations else pd.Series(True, index=df.index)
if hour_ranges:
    h_mask = pd.Series(False, index=df.index)
    for lo, hi in hour_ranges:
        h_mask |= df["hour"].between(lo, hi-1) if hi > lo else df["hour"] >= lo
else:
    h_mask = pd.Series(True, index=df.index)
d_mask = df["day_of_week"].isin(selected_days) if selected_days else pd.Series(True, index=df.index)
v_mask = df["vehicle_type"].isin(selected_vehicles) if selected_vehicles else pd.Series(True, index=df.index)
filtered_df = df[st_mask & h_mask & d_mask & v_mask]
display_df = filtered_df if len(filtered_df) > 0 else df

if len(filtered_df) == 0:
    st.sidebar.warning("No matches. Showing all data.")

# ── Header (compact) ────────────────────────────────────
st.title("🚗 ParkingIntel")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Violations", f"{len(df):,}")
c2.metric("Showing", f"{len(display_df):,}")
c3.metric("Stations", f"{df['police_station'].nunique()}")
c4.metric("Period", "Nov '23 - Apr '24")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Enforcement Planner", "🗺️ Hotspot Map", "📊 Station Comparison",
    "⏰ Time Patterns", "🔍 Gap Analysis", "🚘 Vehicle Profiles",
    "📅 Weekday vs Weekend",
])

# ═══════════════════════════════════════════════════════════
# TAB 7: Enforcement Planner
# ═══════════════════════════════════════════════════════════
with tab2:
    st.header("Enforcement Planner")
    st.caption(
        "Select a day and time window. The system predicts where violations "
        "are most likely to concentrate based on historical patterns, and "
        "suggests an efficient patrol route."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        plan_day = st.selectbox("Day", options=range(7), format_func=lambda x: day_options[x], index=5)
    with col2:
        plan_start, plan_end = st.select_slider(
            "Time Window",
            options=list(range(24)),
            value=(16, 20),
            format_func=lambda x: f"{x:02d}:00",
        )
    with col3:
        n_officers = st.number_input("Available Officers", min_value=1, max_value=10, value=3)

    if st.button("Generate Patrol Plan", type="primary"):
        with st.spinner("Analyzing historical patterns..."):
            zones = plan_hotspots(df, day_of_week=plan_day, start_hour=plan_start, end_hour=plan_end, n_officers=n_officers)

        if len(zones) == 0:
            st.warning("Not enough historical data for this time window. Try a different day or broader time range.")
        else:
            ordered = suggest_patrol_order(zones)
            plan_text = format_plan(ordered, day_options[plan_day], plan_start, plan_end, n_officers)

            # Show the plan
            st.success(f"Plan ready — {len(ordered)} priority zones identified")
            st.markdown(plan_text.replace("  \n", "\n"))

            # Show zones on a mini map
            st.subheader("Patrol Route Map")
            pm = create_base_map()
            for _, zone in ordered.iterrows():
                color = "#00ff00" if zone["stop_number"] == 1 else "#ffcc00"
                folium.Marker(
                    location=[zone["latitude"], zone["longitude"]],
                    icon=folium.DivIcon(
                        html=f'<div style="background:{color};color:#000;border-radius:50%;width:28px;height:28px;text-align:center;line-height:28px;font-weight:bold;font-size:14px;border:2px solid #fff">{int(zone["stop_number"])}</div>'
                    ),
                ).add_to(pm)
            # Fit map to show all zones
            if len(ordered) > 0:
                pm.fit_bounds([[ordered["latitude"].min(), ordered["longitude"].min()],
                               [ordered["latitude"].max(), ordered["longitude"].max()]])
            st.components.v1.html(pm._repr_html_(), height=450)

            st.caption(
                "Green = starting point (highest priority). Yellow = subsequent stops. "
                "Route optimized for minimum travel distance. Numbers show recommended visit order."
            )
    else:
        st.info("Select a day, time window, and number of officers, then click **Generate Patrol Plan**.")



# ═══════════════════════════════════════════════════════════
# PAGE 2: Station Comparison
# ═══════════════════════════════════════════════════════════
with tab3:
    st.header("How do stations compare?")
    st.caption("These metrics help identify where additional support may be useful. They are not officer rankings.")

    summary = get_summary(df)
    metric = st.selectbox("View by", options=[
        "overall_score","quality_score","coverage_score",
        "responsiveness_score","balance_score",
        "rejection_rate","zone_complexity","unprocessed_rate",
    ], format_func=lambda x: {
        "overall_score":"Overall Score","quality_score":"Ticket Accuracy",
        "coverage_score":"Hours Active","responsiveness_score":"Processing Speed",
        "balance_score":"Weekday/Weekend Fit","rejection_rate":"Rejection Rate",
        "zone_complexity":"Zone Difficulty","unprocessed_rate":"Backlog",
    }[x])

    fig = plot_station_scorecard(summary, metric)
    st.plotly_chart(fig, width='stretch')

    with st.expander("How are these scores calculated?"):
        st.markdown(
            "**Quality:** Approval rate of recorded violations.\\n\\n"
            "**Coverage:** Percentage of the 24-hour day with recorded activity.\\n\\n"
            "**Responsiveness:** How quickly records are processed.\\n\\n"
            "**Balance:** How well the weekday/weekend split fits a balanced pattern.\\n\\n"
            "**Zone Difficulty:** Estimated from vehicle type diversity and number of junctions.\\n\\n"
            "A lower score may mean older devices, a more complex area, or different shift patterns "
            "— not that officers are performing poorly."
        )

    st.download_button("Download CSV", summary.to_csv(index=False), "stations.csv")

    st.divider()
    st.markdown("### Ask ParkingIntel")
    render_chat("2")
# ═══════════════════════════════════════════════════════════
# PAGE 3: Time Patterns
# ═══════════════════════════════════════════════════════════
with tab4:
    st.header("When do violations happen?")

    hdf, enf, cong = enforcement_gap_analysis(df)
    ratio = enf / max(cong, 1)
    st.info(
        f"**{enf:,} violations** recorded 4-7 AM vs **{cong:,}** at 5-8 PM "
        f"({ratio:.0f}x difference). This may reflect overnight parking, morning "
        "enforcement sweeps, or both."
    )

    st.subheader("Day x Hour")
    st.plotly_chart(plot_hourly_heatmap(df), width='stretch')

    st.subheader("By Hour")
    st.plotly_chart(plot_hourly_bar(peak_hour_profile(df)), width='stretch')

    with st.expander("What does the 4-7 AM spike mean?"):
        st.markdown(
            "**Possibility 1:** People park illegally overnight — commercial vehicles "
            "loading at dawn, night-shift workers, residents leaving cars on streets.\\n\\n"
            "**Possibility 2:** Officers conduct coordinated morning rounds when streets "
            "are empty and violations are easiest to record.\\n\\n"
            "**Takeaway:** Regardless of the cause, additional enforcement during "
            "afternoon and evening hours would catch violations that directly "
            "affect rush-hour traffic."
        )

    st.divider()
    st.markdown("### Ask ParkingIntel")
    render_chat("3")
# ═══════════════════════════════════════════════════════════
# PAGE 4: Gap Analysis
# ═══════════════════════════════════════════════════════════
with tab5:
    st.header("Where could coverage be improved?")

    st.markdown(
        "Since the data only shows where violations **were** recorded, we use "
        "indirect signals to identify areas that may benefit from more attention."
    )

    blindspots = get_blindspots(df)
    st.plotly_chart(plot_blindspot_ranking(blindspots), width='stretch')

    method = st.radio("Method", ["Combined","Vehicle Diversity","Time Coverage","Junction Density"], horizontal=True)

    if method == "Vehicle Diversity":
        st.caption("Areas with many vehicle types but few recorded violations.")
        dd = vehicle_diversity_blindspots(df)
        st.dataframe(dd.head(10)[["police_station","unique_vehicle_types","total_violations","blind_spot_score"]], use_container_width=True, hide_index=True)

    elif method == "Time Coverage":
        st.caption("Stations active for fewer hours of the day.")
        td = time_coverage_blindspots(df)
        st.dataframe(td.head(10)[["police_station","active_hours","coverage_pct","gap_score"]], use_container_width=True, hide_index=True)

    elif method == "Junction Density":
        st.caption("Stations covering many junctions with few recorded violations.")
        jd = junction_density_blindspots(df)
        st.dataframe(jd.head(10)[["police_station","unique_junctions","total_violations","violations_per_junction"]], use_container_width=True, hide_index=True)

    else:
        st.dataframe(blindspots.head(10)[["police_station","combined_blind_spot_score"]], use_container_width=True, hide_index=True)

    with st.expander("How reliable are these signals?"):
        st.markdown(
            "These methods flag areas where the data pattern is **unusual** — not "
            "where violations are definitely being missed. Each method has limitations. "
            "Vehicle diversity assumes more types = more activity. Time coverage "
            "assumes violations happen throughout the day. Junction density depends "
            "on junction data being complete. Use these as starting points for "
            "further review, not as conclusions."
        )

    st.divider()
    st.markdown("### Ask ParkingIntel")
    render_chat("4")
# ═══════════════════════════════════════════════════════════
# PAGE 5: Vehicle Profiles
# ═══════════════════════════════════════════════════════════
with tab6:
    st.header("What vehicles are involved?")

    station_pick = st.selectbox("Select a station", options=sorted(df["police_station"].dropna().unique()),
                                 index=sorted(df["police_station"].dropna().unique()).index("City Market") if "City Market" in df["police_station"].unique() else 0)

    st.plotly_chart(plot_vehicle_pie(df, station_pick), width='stretch')

    fp = vehicle_fingerprint(df, station_pick)
    top_v = max(fp, key=fp.get) if fp else "N/A"
    st.info(f"**{station_pick}:** Most common vehicle is **{top_v}** ({fp.get(top_v, 0):.0f}% of records).")

    with st.expander("Compare multiple stations"):
        comp = st.multiselect("Select stations", options=sorted(df["police_station"].dropna().unique()), max_selections=5)
        if comp:
            cd = pd.DataFrame({s: vehicle_fingerprint(df, s) for s in comp}).fillna(0)
            st.dataframe(cd, use_container_width=True)

    st.divider()
    st.markdown("### Ask ParkingIntel")
    render_chat("5")
# ═══════════════════════════════════════════════════════════
# PAGE 6: Weekday vs Weekend
# ═══════════════════════════════════════════════════════════
with tab7:
    st.header("Weekday vs Weekend patterns")

    st.caption(
        "Each station's recorded split. Some variation is natural — different "
        "areas have different rhythms. We don't have shift schedule or land-use data, "
        "so these are observations, not prescriptions."
    )

    weekend_df = get_weekend_data(df)
    st.plotly_chart(plot_weekend_scatter(weekend_df), width='stretch')

    lo = weekend_df.nsmallest(1, "weekend_pct").iloc[0]
    hi = weekend_df.nlargest(1, "weekend_pct").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Lowest weekend", f"{lo['weekend_pct']:.1f}%", lo.name)
    c2.metric("Highest weekend", f"{hi['weekend_pct']:.1f}%", hi.name)
    c3.metric("Spread", f"{hi['weekend_pct'] - lo['weekend_pct']:.1f} pp")

    disp = weekend_df.sort_values("weekend_pct", ascending=False)[["weekday","weekend","weekend_pct"]]
    disp.columns = ["Weekday","Weekend","Weekend %"]
    disp["Weekend %"] = disp["Weekend %"].round(1)
    st.dataframe(disp, use_container_width=True)

    st.divider()
    st.markdown("### Ask ParkingIntel")
    render_chat("6")

# ═══════════════════════════════════════════════════════════
# ── Footer ──
st.sidebar.divider()
st.sidebar.caption("ParkingIntel - Hackathon Project")
st.sidebar.caption("Theme 1: Parking-Induced Congestion")

