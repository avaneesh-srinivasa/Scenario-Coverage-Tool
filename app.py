import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scenario Coverage Tool", layout="wide")

st.title("Scenario Coverage Tool")

st.markdown("""
### Goal of the Tool
This tool is a simple way to check whether we have collected **enough test runs** for one selected hazardous scenario for identifying validation gaps.

The idea is not to prove full system safety but to check whether the test evidence is **sufficiently spread across the important parts of the scenario space**.

### Hazardous Scenario Under Consideration
From a SOTIF point of view, this can be viewed as a combination of:
- **Scenario**: ego vehicle is following a lead vehicle in the same lane
- **Hazardous behavior**: lead vehicle suddenly applies strong braking and the ego vehicle may not respond safely

So the hazardous scenario under study here is:

**Lead vehicle hard braking while the ego vehicle is following in the same lane**

### Assumed ODD
- Divided highway
- Dry road
- Daylight
- Good visibility
- Same lane following
- No cut-in
- No adjacent vehicle conflict

### Logic of the Tool
- Step 1. Generate synthetic test runs for the same hazardous scenario
- Step 2. Track initial ego speed and initial gap for each run
- Step 3. Create scenario bins using speed and gap
- Step 4. Assign a risk rating to each bin
- Step 5. Assign required test runs based on risk
- Step 6. Compare completed runs against required runs
- Step 7. Show a heatmap to highlight validation gaps
""")

n = 150
np.random.seed(7)

sources = np.random.choice(
    ["sim", "closed_track", "real_world"],
    size=n,
    p=[0.6, 0.25, 0.15]
)

ego_speed = np.random.uniform(20, 120, n).round(1)
initial_gap = np.random.uniform(5, 60, n).round(1)
lead_speed = (ego_speed + np.random.uniform(-10, 5, n)).round(1)
lead_decel = np.random.uniform(-8.5, -5.0, n).round(1)

risk_value = (ego_speed / 120) * 0.6 + ((60 - initial_gap) / 60) * 0.4
pass_fail = np.where(risk_value > 0.72, "fail", "pass")
collision = np.where(risk_value > 0.82, "yes", "no")
min_ttc = np.maximum(0.2, initial_gap / np.maximum((ego_speed - lead_speed) / 3.6 + 2, 2)).round(2)

df = pd.DataFrame({
    "test_id": [f"T{i+1:03d}" for i in range(n)],
    "source": sources,
    "ego_initial_speed_kph": ego_speed,
    "initial_gap_m": initial_gap,
    "lead_initial_speed_kph": lead_speed,
    "lead_decel_mps2": lead_decel,
    "min_ttc_s": min_ttc,
    "collision": collision,
    "pass_fail": pass_fail
})

speed_bins = [0, 30, 60, 90, 120]
gap_bins = [0, 10, 20, 30, 60]

speed_labels = ["0-30", "30-60", "60-90", "90-120"]
gap_labels = ["0-10", "10-20", "20-30", "30-60"]

df["speed_bin"] = pd.cut(df["ego_initial_speed_kph"], bins=speed_bins, labels=speed_labels, include_lowest=True)
df["gap_bin"] = pd.cut(df["initial_gap_m"], bins=gap_bins, labels=gap_labels, include_lowest=True)

counts = pd.crosstab(df["gap_bin"], df["speed_bin"]).reindex(
    index=gap_labels,
    columns=speed_labels,
    fill_value=0
)

risk_rating = pd.DataFrame(
    [
        ["High", "High", "Very High", "Very High"],
        ["Medium", "Medium", "High", "Very High"],
        ["Low", "Low", "Medium", "High"],
        ["Low", "Low", "Low", "Medium"]
    ],
    index=gap_labels,
    columns=speed_labels
)

required = risk_rating.replace({
    "Low": 10,
    "Medium": 20,
    "High": 40,
    "Very High": 60
}).astype(float)

coverage = (counts / required * 100).round(0).astype(float)

st.markdown("### Synthetic Test Data")
st.dataframe(df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Risk Rating per Scenario Bin")
    st.dataframe(risk_rating, use_container_width=True)

with col2:
    st.markdown("### Required Test Runs per Scenario Bin")
    st.dataframe(required, use_container_width=True)

st.markdown("""
The number of required test runs per scenario bin is assigned using a simple risk-based heuristic, where higher-risk conditions such as high speed and short gap require more validation evidence. These values act as placeholders for scenario-level validation targets and represent how test effort should be distributed across the scenario space.
""")

st.markdown("### Completed Test Runs per Scenario Bin")
st.dataframe(counts, use_container_width=True)

col3, col4 = st.columns([1.3, 1])

with col3:
    st.markdown("### Coverage Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(coverage.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=120)

    ax.set_xticks(range(len(speed_labels)))
    ax.set_xticklabels(speed_labels)
    ax.set_yticks(range(len(gap_labels)))
    ax.set_yticklabels(gap_labels)

    ax.set_xlabel("Initial Ego Speed (kph)")
    ax.set_ylabel("Initial Gap (m)")

    for i in range(len(gap_labels)):
        for j in range(len(speed_labels)):
            text = f"{int(counts.iloc[i, j])}/{int(required.iloc[i, j])}\n{int(coverage.iloc[i, j])}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="Coverage %")
    st.pyplot(fig)

with col4:
    st.markdown("### Summary")
    st.write(f"Total test runs: **{len(df)}**")
    st.write(f"Pass: **{(df['pass_fail'] == 'pass').sum()}**")
    st.write(f"Fail: **{(df['pass_fail'] == 'fail').sum()}**")
    st.write(f"Collisions: **{(df['collision'] == 'yes').sum()}**")

    st.markdown("### Validation Gaps")
    gaps = []

    for g in gap_labels:
        for s in speed_labels:
            if coverage.loc[g, s] < 100:
                gaps.append(f"Speed {s} kph, Gap {g} m")

    if len(gaps) > 0:
        for item in gaps:
            st.write("- " + item)
    else:
        st.write("All bins meet target.")

st.markdown("""
### Limitations and Future Work
- This tool uses **synthetic test data**, not real validation data
- Required test runs per bin are assumed from a simple risk rating table
- The tool does **not yet convert** number of test runs into a fleet-level validation target such as crashes per million miles
- The tool does **not yet use exposure** to connect scenario-level testing to overall validation targets
- Test runs from **simulation, closed track, and real world are counted equally**
- The tool does **not yet apply weighting** based on evidence strength or realism
- Only two parameters are used for binning: initial speed and initial gap
- ODD conditions are fixed as assumptions and not varied in the current version
- Future versions can include exposure-based targets, source weighting, confidence calculations, and more detailed scenario parameters
""")
