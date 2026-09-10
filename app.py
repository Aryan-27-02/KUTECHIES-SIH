import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time
import os
from PIL import Image

# Import SafeStreet Modules
from config import (
    SAFETY_PARAMETERS, RISK_LEVELS, AUDITS_CSV, DEFECTS_CSV,
    EVIDENCE_DIR, DEMO_DIR, DEFAULT_GPS_LAT, DEFAULT_GPS_LON
)
from core.video_processor import VideoProcessor
from core.object_detector import ObjectDetector
from core.condition_classifier import ConditionClassifier
from core.rule_engine import RuleEngine
from core.scoring_engine import ScoringEngine
from storage.audit_store import AuditStore
from demo_assets.generate_sample_video import generate_school_zone_video

# -------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="SafeStreet | School Zone Pedestrian Safety Audit",
    page_icon="🚸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling - High-contrast Obsidian Dark Mode
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F8FAFC !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8 !important;
        margin-bottom: 1.2rem;
    }
    .arch-badge {
        background: rgba(30, 58, 138, 0.35) !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        border-left: 4px solid #3B82F6 !important;
        padding: 0.65rem 1rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
        color: #93C5FD !important;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #F8FAFC !important;
    }
    .defect-card {
        background: rgba(220, 38, 38, 0.12) !important;
        border: 1px solid rgba(239, 68, 68, 0.45) !important;
        border-left: 5px solid #EF4444 !important;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .defect-card h4 {
        color: #F8FAFC !important;
        margin-top: 8px;
        margin-bottom: 4px;
        font-weight: 700;
    }
    .defect-card p {
        color: #CBD5E1 !important;
        font-size: 0.92rem;
        margin-bottom: 8px;
    }
    .defect-card small {
        color: #94A3B8 !important;
    }
    .warning-card {
        background: rgba(245, 158, 11, 0.12) !important;
        border: 1px solid rgba(245, 158, 11, 0.45) !important;
        border-left: 5px solid #F59E0B !important;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .warning-card h4 {
        color: #F8FAFC !important;
        margin-top: 8px;
        margin-bottom: 4px;
        font-weight: 700;
    }
    .warning-card p {
        color: #CBD5E1 !important;
        font-size: 0.92rem;
        margin-bottom: 8px;
    }
    .warning-card small {
        color: #94A3B8 !important;
    }
    .safe-badge {
        color: #10B981 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def show_image(container, img, caption=None, channels="RGB"):
    """Displays an image adapting seamlessly to Streamlit width deprecation rules."""
    try:
        container.image(img, caption=caption, channels=channels, width="stretch")
    except Exception:
        container.image(img, caption=caption, channels=channels, use_container_width=True)

# -------------------------------------------------------------
# Initialize Core Components (Cached for Performance)
# -------------------------------------------------------------
@st.cache_resource
def load_audit_pipeline():
    detector = ObjectDetector(model_name="yolov8n.pt", conf_thresh=0.35)
    classifier = ConditionClassifier()
    rule_engine = RuleEngine()
    scoring_engine = ScoringEngine()
    store = AuditStore()
    return detector, classifier, rule_engine, scoring_engine, store

detector, classifier, rule_engine, scoring_engine, audit_store = load_audit_pipeline()

# -------------------------------------------------------------
# Sidebar: Audit & Survey Configuration
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/school-crossing.png", width=64)
st.sidebar.title("SafeStreet Audit Hub")
st.sidebar.caption("Pedestrian Safety Vision System")

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Target School Zone")

school_name = st.sidebar.text_input(
    "School Name / ID", 
    value="St. Xavier's High School (Gate 2)"
)

city_type = st.sidebar.selectbox(
    "City Context (Contrasting Types)",
    ["Tier-1 Dense Metro (High Traffic)", "Tier-2 Compact City (Narrow Streets)", "Suburban Institutional Area"]
)

survey_mode = st.sidebar.selectbox(
    "Smartphone Survey Mode",
    ["Two-Wheeler Mount (15 km/h)", "Pedestrian Walking Survey (4 km/h)", "E-Rickshaw Mount"]
)

col_gps1, col_gps2 = st.sidebar.columns(2)
with col_gps1:
    base_lat = st.number_input("Base Latitude", value=DEFAULT_GPS_LAT, format="%.4f")
with col_gps2:
    base_lon = st.number_input("Base Longitude", value=DEFAULT_GPS_LON, format="%.4f")

sample_interval = st.sidebar.slider("Keyframe Sampling (Every Nth frame)", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.info("""
**5-Parameter Safety Rubric:**
1. 🦓 Zebra Crossing Condition (25%)
2. 🚶 Footpath Encroachment (20%)
3. 🚸 School Zone Signage (20%)
4. 🚗 Obstructing Parking / Blind Spots (20%)
5. 🛑 Speed Calming & Surface (15%)
""")

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="main-title">SafeStreet: School Zone Safety Vision Auditor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Transforming smartphone video into a geotagged list of pedestrian safety defects and municipal priority rankings.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="arch-badge">
⚙️ <b>Architecture:</b> Computer Vision Detects (YOLO + Keras/TF) &nbsp;➔&nbsp; Rule Engine Interprets (Safe vs Defect) &nbsp;➔&nbsp; Scoring Engine Evaluates (0-100 SSI)
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Navigation Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥 Live Video Audit", 
    "🔍 Defect Evidence Gallery", 
    "📊 Municipal Priority Ranking", 
    "📑 Civil Audit Report",
    "🧠 Architecture & Rubric"
])

# =============================================================
# TAB 1: Live Video Audit
# =============================================================
with tab1:
    st.subheader("1. Video Input & AI Analysis")
    
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        uploaded_video = st.file_uploader(
            "Upload Smartphone Video (.mp4, .mov, .avi)", 
            type=["mp4", "mov", "avi"]
        )
    
    with col_input2:
        st.write("Or try pre-loaded test scenario:")
        use_demo = st.button("🚀 Load Sample School Zone Video", use_container_width=True)
        if use_demo or ("current_video" not in st.session_state and not uploaded_video):
            demo_video_path = DEMO_DIR / "sample_school_zone.mp4"
            if not demo_video_path.exists():
                with st.spinner("Generating sample school zone video..."):
                    generate_school_zone_video(str(demo_video_path))
            st.session_state["current_video"] = str(demo_video_path)
            st.session_state["video_source_name"] = "sample_school_zone.mp4 (Built-in Demo)"

    # Resolve active video
    active_video_path = None
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        active_video_path = tfile.name
        video_name = uploaded_video.name
    elif "current_video" in st.session_state:
        active_video_path = st.session_state["current_video"]
        video_name = st.session_state.get("video_source_name", "sample_school_zone.mp4")

    if active_video_path and os.path.exists(active_video_path):
        try:
            vp = VideoProcessor(active_video_path, sample_interval=sample_interval)
            meta = vp.get_metadata()
            
            # Show Video Specs
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Video Duration", f"{meta['duration_sec']}s")
            mcol2.metric("Resolution", meta["resolution"])
            mcol3.metric("Source FPS", f"{meta['fps']}")
            mcol4.metric("Frames to Audit", f"{meta['sampled_frames_expected']}")

            st.markdown("---")
            run_audit_btn = st.button("▶️ Execute Full SafeStreet Audit", type="primary", use_container_width=True)

            if run_audit_btn:
                st.write("### Processing Video Stream...")
                
                # Layout for live inspection
                vis_col, param_col = st.columns([1.5, 1])
                video_placeholder = vis_col.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                with param_col:
                    st.write("#### Real-time Parameter Safety Status")
                    p1_metric = st.empty()
                    p2_metric = st.empty()
                    p3_metric = st.empty()
                    p4_metric = st.empty()
                    p5_metric = st.empty()
                    ssi_metric = st.empty()

                # Execution loop
                frame_evaluations = []
                audit_defects = []
                processed_count = 0
                total_expected = max(1, meta["sampled_frames_expected"])

                for f_idx, t_sec, lat, lon, frame in vp.frame_generator(base_lat, base_lon):
                    processed_count += 1
                    
                    # 1. Computer Vision Detection (YOLO)
                    detections = detector.detect(frame)
                    
                    # 2. AI Condition Classification (Keras / TF)
                    cross_cond = classifier.classify_zebra_condition(frame)
                    sign_boxes = [d for d in detections if "sign" in d["class"]]
                    sign_cond = classifier.classify_sign_visibility(frame, sign_boxes)
                    surf_cond = classifier.classify_road_surface(frame)

                    # 3. Rule Engine Interpretation
                    rule_eval = rule_engine.evaluate_frame(
                        detections, cross_cond, sign_cond, surf_cond
                    )
                    
                    # 4. Scoring Engine Frame Evaluation
                    frame_ssi, risk_level, breakdown = scoring_engine.calculate_frame_score(rule_eval["parameters"])
                    frame_evaluations.append(rule_eval)

                    # Check for defects to save as evidence
                    for defect in rule_eval.get("defects", []):
                        defect["timestamp_sec"] = round(t_sec, 2)
                        defect["gps"] = (lat, lon)
                        # Save unique evidence frame snapshot if high severity
                        if defect["severity"] in ["CRITICAL DEFECT", "WARNING"]:
                            ev_filename = f"evidence_{len(audit_defects)+1:03d}_{defect['type']}.jpg"
                            ev_path = str(EVIDENCE_DIR / ev_filename)
                            vp.save_evidence_frame(
                                frame, ev_path, defect["title"], defect["severity"], t_sec, (lat, lon)
                            )
                            defect["evidence_frame_path"] = ev_path
                            audit_defects.append(defect)

                    # Visual feedback with annotations and HUD
                    annotated = detector.annotate_frame(frame, detections)
                    hud_frame = vp.draw_hud(
                        annotated, school_name, f"Frame {f_idx} | {t_sec:.1f}s", frame_ssi, risk_level
                    )
                    # Convert BGR to RGB for Streamlit
                    rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                    show_image(video_placeholder, rgb_frame, channels="RGB")

                    # Update live gauges
                    params = rule_eval["parameters"]
                    p1_metric.markdown(f"**🦓 Zebra Crossing:** `{params['zebra_crossing']['state']}` ({params['zebra_crossing']['score']}/100)")
                    p2_metric.markdown(f"**🚶 Footpath Corridor:** `{params['footpath_obstruction']['state']}` ({params['footpath_obstruction']['score']}/100)")
                    p3_metric.markdown(f"**🚸 School Signage:** `{params['school_signage']['state']}` ({params['school_signage']['score']}/100)")
                    p4_metric.markdown(f"**🚗 Blind-Spot Parking:** `{params['obstructing_parking']['state']}` ({params['obstructing_parking']['score']}/100)")
                    p5_metric.markdown(f"**🛑 Speed Calming:** `{params['speed_calming']['state']}` ({params['speed_calming']['score']}/100)")
                    
                    risk_color = RISK_LEVELS[risk_level]["color"]
                    gauge_text_color = "#111827" if risk_level == "MODERATE" else "#FFFFFF"
                    ssi_metric.markdown(
                        f"<div style='background-color:{risk_color}; color:{gauge_text_color}; padding:12px; border-radius:8px; text-align:center; font-weight:bold; box-shadow:0 2px 5px rgba(0,0,0,0.1);'>"
                        f"School Safety Index (SSI): {frame_ssi:.1f}/100<br><span style='font-size:1.1em;'>{risk_level} RISK</span></div>",
                        unsafe_allow_html=True
                    )

                    pct = min(1.0, processed_count / float(total_expected))
                    progress_bar.progress(pct)
                    status_text.text(f"Auditing frame {processed_count} of ~{total_expected} ({int(pct*100)}%)")

                vp.release()
                status_text.success("✅ Audit Video Analysis Completed!")

                # Aggregate Audit via Scoring Engine
                final_audit = scoring_engine.aggregate_audit(
                    school_name, city_type, frame_evaluations, audit_defects
                )
                final_audit["survey_mode"] = survey_mode
                final_audit["video_filename"] = video_name

                # Save to CSV storage
                audit_id = audit_store.save_audit(final_audit, audit_defects)
                st.session_state["latest_audit"] = final_audit
                st.session_state["latest_defects"] = audit_defects
                st.session_state["latest_audit_id"] = audit_id

                st.balloons()
                st.success(f"Audit finalized and saved under ID: **{audit_id}**! Check the other tabs for Defect Evidence and Priority Rankings.")

        except Exception as e:
            st.error(f"Error processing video: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display Latest Audit Summary Cards if available
    if "latest_audit" in st.session_state:
        aud = st.session_state["latest_audit"]
        st.markdown("---")
        st.write("### 📋 Latest Audit Summary")
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Overall School Safety Index (SSI)", f"{aud['overall_ssi']} / 100")
        rcol2.metric("Safety Assessment", aud["risk_level"])
        rcol3.metric("Municipal Urgency Priority", f"{aud['priority_score']} / 100")
        rcol4.metric("Flagged Defects", f"{aud['total_defects']} ({aud['critical_defects']} Critical)")

# =============================================================
# TAB 2: Defect Evidence Gallery
# =============================================================
with tab2:
    st.subheader("2. Geotagged Defect Evidence & Violation Stamps")
    st.caption("Individual defect frames extracted, classified, and stamped with GPS coordinates for civic work orders.")

    defects_to_show = st.session_state.get("latest_defects", [])
    
    if not defects_to_show:
        # Load sample defects from CSV if session is fresh
        all_defects_df = pd.read_csv(DEFECTS_CSV) if DEFECTS_CSV.exists() else pd.DataFrame()
        st.info("Run an audit in Tab 1 to see freshly captured evidence, or view previously saved defects below.")
    
    if defects_to_show:
        # Filter controls
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            severity_filter = st.selectbox("Filter by Severity", ["All", "CRITICAL DEFECT", "WARNING"])
        with fcol2:
            param_filter = st.selectbox("Filter by Parameter", ["All"] + [v["name"] for v in SAFETY_PARAMETERS.values()])

        filtered = []
        for d in defects_to_show:
            if severity_filter != "All" and d.get("severity") != severity_filter:
                continue
            if param_filter != "All" and d.get("param_name") != param_filter:
                continue
            filtered.append(d)

        st.write(f"Showing **{len(filtered)}** defect instances:")

        cols = st.columns(2)
        for i, defect in enumerate(filtered):
            with cols[i % 2]:
                card_class = "defect-card" if defect.get("severity") == "CRITICAL DEFECT" else "warning-card"
                badge_bg = "#DC2626" if defect.get("severity") == "CRITICAL DEFECT" else "#D97706"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <span style="background-color:{badge_bg}; color:#FFFFFF; padding:4px 9px; border-radius:4px; font-size:0.75rem; font-weight:bold; letter-spacing:0.3px;">
                        {defect.get('severity')}
                    </span>
                    <h4>{defect.get('title')}</h4>
                    <p>{defect.get('description')}</p>
                    <small>⏱️ Video Time: <b>{defect.get('timestamp_sec', 0.0)}s</b> &nbsp;|&nbsp; 📍 Lat: <b>{defect.get('gps', (0,0))[0]:.5f}</b>, Lon: <b>{defect.get('gps', (0,0))[1]:.5f}</b></small>
                </div>
                """, unsafe_allow_html=True)

                img_path = defect.get("evidence_frame_path")
                if img_path and os.path.exists(img_path):
                    show_image(st, img_path, caption=f"Evidence Snapshot: {defect.get('title')}")

# =============================================================
# TAB 3: Municipal Priority Ranking
# =============================================================
with tab3:
    st.subheader("3. Municipal Priority Ranking (Contrasting City Types)")
    st.caption("Comparative safety index across surveyed school zones. Helps civic engineers allocate repair funds to highest-hazard zones first.")

    audits_df = audit_store.get_all_audits()

    if not audits_df.empty:
        # Highlights
        st.write("#### 🏆 Municipal Intervention Queue (Ranked by Urgency)")
        
        # Format table
        display_df = audits_df[[
            "audit_id", "school_name", "city_type", "priority_score",
            "overall_ssi", "risk_level", "total_defects", "critical_defects"
        ]].copy()
        
        st.dataframe(
            display_df,
            column_config={
                "priority_score": st.column_config.ProgressColumn(
                    "Priority Urgency",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "overall_ssi": st.column_config.NumberColumn(
                    "Safety Index (SSI)",
                    format="%.1f / 100"
                )
            },
            use_container_width=True,
            hide_index=True
        )

        # Comparative Visualization
        st.markdown("---")
        st.write("#### 🏙️ Contrasting City Types: Infrastructure Defect Breakdown")
        
        chart_data = audits_df[["school_name", "crossing_score", "footpath_score", "signage_score", "parking_score", "calming_score"]].set_index("school_name")
        st.bar_chart(chart_data)

    else:
        st.info("No audit records found yet.")

# =============================================================
# TAB 4: Civil Audit Report Export
# =============================================================
with tab4:
    st.subheader("4. Municipal Corporation Road Safety Defect Report")
    st.caption("Official technical memo for Municipal Engineers, Traffic Police, and Urban Development Authorities.")

    if "latest_audit" in st.session_state:
        aud = st.session_state["latest_audit"]
        audit_id = st.session_state.get("latest_audit_id", "AUD-CURRENT")
        
        report_text = f"""# SAFESTREET MUNICIPAL ROAD DEFECT AUDIT REPORT
**Audit ID:** {audit_id}  
**Date & Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Target Facility:** {aud['school_name']}  
**City Typology:** {aud['city_type']}  
**Survey Methodology:** {aud.get('survey_mode', 'Smartphone Vision Survey')}  

---

### EXECUTIVE SUMMARY
- **Composite School Safety Index (SSI):** {aud['overall_ssi']} / 100
- **Assessed Risk Level:** {aud['risk_level']}
- **Municipal Intervention Priority Score:** {aud['priority_score']} / 100
- **Total Flagged Defects:** {aud['total_defects']} (Critical: {aud['critical_defects']}, Warnings: {aud['warnings']})

---

### 5-PARAMETER SAFETY BREAKDOWN
1. **Zebra Crossing Markings:** {aud['parameter_averages'].get('zebra_crossing', 'N/A')}/100
2. **Footpath Walkability & Encroachment:** {aud['parameter_averages'].get('footpath_obstruction', 'N/A')}/100
3. **School Zone Signage Visibility:** {aud['parameter_averages'].get('school_signage', 'N/A')}/100
4. **Blind-Spot Obstructing Parking:** {aud['parameter_averages'].get('obstructing_parking', 'N/A')}/100
5. **Speed Calming Devices & Surface:** {aud['parameter_averages'].get('speed_calming', 'N/A')}/100

---

### ACTIONABLE WORK-ORDER RECOMMENDATIONS
1. Repaint thermoplastic reflective zebra crossing markings directly fronting school gate.
2. Install bollards or enforcement barriers along the sidewalk to prevent parked two-wheelers from blocking child pedestrian paths.
3. Erect high-visibility retro-reflective IRC:67 School Warning cautionary signs 50m in advance of the school crossing.
4. Mark a 30m 'No-Stopping / No-Parking' dispersal buffer to eliminate blind-spot conflicts.
"""
        st.markdown(report_text)
        
        st.download_button(
            label="📥 Download Audit Report (.MD)",
            data=report_text,
            file_name=f"SafeStreet_Audit_{audit_id}.md",
            mime="text/markdown"
        )
        
        # Download Audits CSV
        if AUDITS_CSV.exists():
            with open(AUDITS_CSV, "rb") as f:
                st.download_button(
                    label="📊 Download Audits Master Database (.CSV)",
                    data=f,
                    file_name="SafeStreet_Audits_Master.csv",
                    mime="text/csv"
                )
    else:
        st.info("Execute an audit in Tab 1 to generate an official civil audit report.")

# =============================================================
# TAB 5: Architecture & Hackathon Rubric
# =============================================================
with tab5:
    st.subheader("5. System Architecture & Technical Rubric")
    
    st.markdown("""
    ### Why SafeStreet is Structured This Way
    
    ```
    ┌─────────────────────────┐
    │ Computer Vision Detects │  (OpenCV + YOLO + Keras/TensorFlow)
    └────────────┬────────────┘
                 │ Raw bounding boxes, visual features, condition classifications
                 ▼
    ┌─────────────────────────┐
    │  Rule Engine Interprets │  (IRC:67 & IRC:103 Safety Rubric)
    └────────────┬────────────┘
                 │ SAFE vs WARNING vs CRITICAL DEFECT
                 ▼
    ┌─────────────────────────┐
    │ Scoring Engine Evaluates│  (Weighted composite School Safety Index 0-100)
    └────────────┬────────────┘
                 │ Actionable Municipal Priority Score & Evidence Frames
    ```
    
    #### 1. Decoupled Architecture
    - Separating **Computer Vision (Perception)** from the **Rule Engine (Policy)** and **Scoring Engine (Economics/Urgency)** means road safety standards can be updated without retraining the computer vision models.
    
    #### 2. Dual AI Model Stack
    - **YOLOv8**: Real-time object detection of moving and static entities (pedestrians, cars, two-wheelers, buses, signs).
    - **Keras + TensorFlow**: High-precision condition classifier assessing paint wear on crossings, sign occlusion, and surface defects.
    
    #### 3. Ready for Google Colab
    - Open `SafeStreet_Model_Training.ipynb` in Google Colab to retrain the Keras condition classifier or fine-tune YOLO on localized municipal datasets.
    """)

