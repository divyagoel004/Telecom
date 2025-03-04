import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import random

import urllib.parse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events
from http.server import BaseHTTPRequestHandler
import streamlit as st
import streamlit.components.v1 as components


# -------------------- Environment & Data Setup --------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = "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"
try:
    from groq import Groq
    from together import Together
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    os.environ['TOGETHER_API_KEY'] = "a566a4825abfe3bf9a18dc844ea8275bf9496105475b3a7211862ccc51a22316"
    together_client = Together(api_key=os.environ["TOGETHER_API_KEY"])
except ImportError:
    st.warning("Groq and Together modules not found; voice-to-SQL features may not work.")



# username = "postgres"
# password = urllib.parse.quote_plus("Az@di1947")  # URL-encode special chars
# engine = create_engine(f"postgresql://{username}:{password}@localhost:5432/telecom")

GRAPH_TYPES = [
    {"label": "Bar Chart", "value": "bar"},
    {"label": "Line Chart", "value": "line"},
    {"label": "Scatter Plot", "value": "scatter"},
    {"label": "Histogram", "value": "histogram"},
    {"label": "Box Plot", "value": "box"},
    {"label": "Violin Plot", "value": "violin"},
    {"label": "Area Chart", "value": "area"},
    {"label": "Bubble Chart", "value": "bubble"},
    {"label": "Heatmap", "value": "heatmap"},
    {"label": "Density Contour", "value": "density_contour"}
]

# -------------------- Load Synthetic Data --------------------
def transform_data(force=True):
    # """
    # Query KPI data from the database.
    # """
    # Q = "select * from kpi"
    # with engine.connect() as conn:
    #     data = pd.read_sql(text(Q), conn)
    data=pd.read_csv("synthetic_fiber_data_with_truckroll.csv",parse_dates=['recorded_at']) 
    data['recorded_at'] = pd.to_datetime(data['recorded_at'])
    rows = 40000
    start_index = np.random.randint(0, rows - 100)
    subset_index = range(start_index, start_index + 100)

    # Generate 100 random timestamps within the last 24 hours.
    now = datetime.now()
    start_time = now - timedelta(hours=24)
    start_ts = int(start_time.timestamp())
    end_ts = int(now.timestamp())
    random_ts = np.random.randint(start_ts, end_ts, 100)
    random_times = pd.to_datetime(random_ts, unit='s')

    # Update the 'recorded_at' column for the selected 100 rows.
    data.loc[subset_index, "recorded_at"] = random_times
    return data
df=transform_data()
# -------------------- Voice-to-SQL Functions --------------------
def recognize_speech():
  

    st.title("Voice Input Without SpeechRecognition")

    # Inline HTML/JS code for the voice input component
    html_code = """
<script src="https://unpkg.com/streamlit-component-lib@latest/dist/index.js"></script>
<script>
  // Check for browser support for SpeechRecognition
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    // Return a message if not supported
    Streamlit.setComponentValue("Speech recognition is not supported in this browser.");
  } else {
    // Create a new SpeechRecognition object
    var recognition = new SpeechRecognition();
    recognition.lang = "en-US";       // Adjust language as needed
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onresult = function(event) {
      // Get the transcribed text from the result
      var transcript = event.results[0][0].transcript;
      // Send the transcript back to Streamlit
      Streamlit.setComponentValue(transcript);
    };

    recognition.onerror = function(event) {
      Streamlit.setComponentValue("Error: " + event.error);
    };

    // Function to start speech recognition
    function startVoice() {
      recognition.start();
    }

    // Create a button dynamically if it doesn't exist
    if (!document.getElementById("voiceButton")) {
      var btn = document.createElement("button");
      btn.id = "voiceButton";
      btn.style.fontSize = "16px";
      btn.style.padding = "8px 16px";
      btn.innerHTML = "Start Voice Input";
      btn.onclick = startVoice;
      document.body.appendChild(btn);
    }
  }
</script>
"""

        # Render the custom component using components.html.
    # The returned value from Streamlit.setComponentValue() will be captured in 'result'.
    result = components.html(html_code, height=150, scrolling=False)

    # Display the result
    if result:
        st.write("Voice input result:", result)
        return result
    else:
        return "Click the button above and speak to see the result."

def generate_sql(query):
    schema = '''kpi(Fiber_Type, Cable_Length_km, Used_Fiber_Strands, Unused_Fiber_Strands,
      Installation_Date, Connector_Type, Patch_Panel_Type, Measurement_Time, Optical_Power_dBm, Optical_Loss_dB,
        Attenuation_Rate, Bit_Error_Rate, Latency_ms, Jitter_ms, Data_Transmission_Rate,
          Data_Rate_Unit, Packet_Loss_Percent, Error_Logs, Recent_Fiber_Breaks,
       Avg_Downtime_per_Fault_min, Signal_Degradation_Rate_Percent,
       Historical_Maintenance_Frequency, Fiber_Cut_Detection,
       Bending_Losses, Interference, Weather_Conditions,
       Fiber_Location, Nearby_Construction,
       Distance_from_Network_Node_km, Power_Supply_Stability,
       ONT_OLT_Signal_Strength, Router_Modem_Logs, Battery_Backup,
       Voltage_Stability, Device_Overheating, Alarms, Users_Affected,
       Customer_Complaints, SLA_Priority, Service_Type,
       Customer_Downtime_Tolerance, Truck_Roll_Requirement, recorded_at,
       Time_Range, Geographic_Region, Network_Node_Type, Issue_Type,
       Technician_Skill_Level, Customer_SLA_Priority,
       Truck_Roll_Decision, Customer_Satisfaction_Score,
       Complaint_Resolution_Time, recorded_at.1, region_id,
       network_node_id, fiber_id, issue_id, technician_id, sla_id,
       weather_id, service_id, truck_roll, kpi_name, kpi_value,
       Time_Range.1, Geographic_Region.1, Network_Node_Type.1,
       Issue_Type.1, Technician_Skill_Level.1, Customer_SLA_Priority.1,
       Truck_Roll_Decision.1)'''
    system_prompt = f"""
You are a SQL expert. Convert the following natural language query into a SQL statement.
Use this schema: {schema}.
Return ONLY the SQL code, no explanations.
""" 
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1
    )
    sql_query = response.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
    st.write("Extract Data:", sql_query)
    return sql_query

def fetch_data(query):
    data = df.copy()
    data = data.query(query)
    data = data[columns]
    return data

# -------------------- Global Filter Options --------------------
time_range_options = ["24h", "7d", "30d", "all", "custom"]

region_options = sorted(df["Geographic_Region"].unique())
node_type_options = sorted(df["Network_Node_Type"].unique())
fiber_type_options = sorted(df["Fiber_Type"].unique())  # Corrected column name
issue_type_options = sorted(df["Issue_Type"].unique())
tech_skill_options = sorted(df["Technician_Skill_Level"].unique())
sla_priority_options = sorted(df["Customer_SLA_Priority"].unique())
weather_options = sorted(df["Weather_Conditions"].unique())  # Corrected column name
service_options = sorted(df["Service_Type"].unique())          # Corrected column name
truck_roll_options = sorted(df["Truck_Roll_Decision"].unique())

group_by_options = ["none", "Geographic_Region", "Network_Node_Type", "fiber_id",
                    "Issue_Type", "Technician_Skill_Level", "Customer_SLA_Priority", "weather_id", "service_id", "Truck_Roll_Decision"]
graph_type_options = ["line", "bar", "scatter", "area"]

def get_filtered_df(time_range, region_val, node_val, fiber_val, issue_val,
                    tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = df.copy()
    now = filtered["recorded_at"].max()
    if time_range != "all":
        if time_range == "24h":
            cutoff = now - timedelta(hours=24)
            filtered = filtered[filtered["recorded_at"] >= cutoff]
        elif time_range == "7d":
            cutoff = now - timedelta(days=7)
            filtered = filtered[filtered["recorded_at"] >= cutoff]
        elif time_range == "30d":
            cutoff = now - timedelta(days=30)
            filtered = filtered[filtered["recorded_at"] >= cutoff]
    if region_val:
        filtered = filtered[filtered["Geographic_Region"].isin(region_val)]
    if node_val:
        filtered = filtered[filtered["Network_Node_Type"].isin(node_val)]
    if fiber_val:
        filtered = filtered[filtered["Fiber_Type"].isin(fiber_val)]
    if issue_val:
        filtered = filtered[filtered["Issue_Type"].isin(issue_val)]
    if tech_val:
        filtered = filtered[filtered["Technician_Skill_Level"].isin(tech_val)]
    if sla_val:
        filtered = filtered[filtered["Customer_SLA_Priority"].isin(sla_val)]
    if truck_roll_val:
        filtered = filtered[filtered["Truck_Roll_Decision"].isin(truck_roll_val)]
    return filtered

def generate_overview_cards(filtered):
    # Store full filtered data for drill-down purposes
    st.session_state.full_filtered_data = filtered
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🚨 Critical Fiber Issues", key="critical_card"):
            st.session_state.selected_card = {"type": "critical", "filters": {"Issue_Type": "Fiber Break"}}
    with col2:
        if st.button("🚚 Truck Rolls", key="truck_card"):
            st.session_state.selected_card = {"type": "truck", "filters": {"Truck_Roll_Decision": "Required"}}
    with col3:
        if st.button("📡 Healthy Connections", key="healthy_card"):
            st.session_state.selected_card = {"type": "healthy", "filters": {"Uptime": lambda x: x >= 95}}
    with col4:
        if st.button("🔧 Pending Maintenance", key="pending_card"):
            st.session_state.selected_card = {"type": "pending", "filters": {"Historical_Maintenance_Frequency": lambda x: x > 5}}
    
    if "selected_card" in st.session_state:
        show_card_details(st.session_state.selected_card)

def show_card_details(selected_card):
    filtered_data = st.session_state.full_filtered_data.copy()
    for key, value in selected_card["filters"].items():
        if callable(value):
            filtered_data = filtered_data[filtered_data[key].apply(value)]
        else:
            filtered_data = filtered_data[filtered_data[key] == value]
    with st.expander(f"🔍 {selected_card['type'].capitalize()} Details", expanded=True):
        st.write(f"Showing {len(filtered_data)} records matching:")
        st.write(selected_card["filters"])
        st.dataframe(filtered_data)
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Data Distribution")
            fig = px.histogram(filtered_data, x="kpi_value")
            st.plotly_chart(fig, use_container_width=True, height=300)
        with col2:
            st.write("### Trend Over Time")
            time_fig = px.line(filtered_data, x="recorded_at", y="kpi_value")
            st.plotly_chart(time_fig, use_container_width=True, height=300)
        csv = filtered_data.to_csv(index=False)
        st.download_button(label="📥 Download Filtered Data", data=csv, file_name=f"{selected_card['type']}_data.csv", mime="text/csv")

# -------------------- Dashboard Graph Functions --------------------
def update_fiber_util(time_range, region_val, node_val, fiber_val,
                      issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
        return px.histogram(title="No data available for Fiber Utilization Rate")
    fig = px.histogram(filtered, x="recorded_at", y="Data_Transmission_Rate",
                       labels={"recorded_at": "Time", "Fiber_Utilization_Rate": "Utilization (%)"})
    return fig

def update_packet_loss(time_range, region_val, node_val, fiber_val,
                       issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
        return px.scatter(title="No data available for Packet Loss Percentage")
    fig = px.scatter(filtered, x="recorded_at", y="Packet_Loss_Percent",
                     labels={"recorded_at": "Time", "Packet_Loss_Percent": "Packet Loss (%)"})
    return fig

def update_latency(time_range, region_val, node_val, fiber_val,
                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
        return px.area(title="No data available for Latency Trends")
    fig = px.area(filtered, x="recorded_at", y="Latency_ms",
                  labels={"recorded_at": "Time", "Latency_ms": "Latency (ms)"})
    return fig

def update_signal(time_range, region_val, node_val, fiber_val,
                  issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
        return px.area(title="No data available for Signal Strength & Noise Ratio")
    fig = px.area(filtered, x="recorded_at", y="ONT_OLT_Signal_Strength",
                  labels={"recorded_at": "Time", "ONT_OLT_Signal_Strength": "Signal Strength (dBm)"})
    return fig

def update_uptime(time_range, region_val, node_val, fiber_val,
                  issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    # df_kpi = filtered[filtered["kpi_name"] == "%Uptime / Performance"]
    if filtered.empty:
        return px.area(title="No data available for Uptime")
    fig = px.histogram(filtered, x="recorded_at", y="ONT_OLT_Signal_Strength",
                  labels={"recorded_at": "Time", "Uptime": "Uptime (%)"})
    return fig

def update_complaint(time_range, region_val, node_val, fiber_val,
                     issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
        return px.area(title="No data available for Complaint Resolution Time")
    fig = px.area(filtered, x="recorded_at", y="Complaint_Resolution_Time",
                  labels={"recorded_at": "Time", "Complaint_Resolution_Time": "Resolution Time (min)"})
    return fig

def update_csat(time_range, region_val, node_val, fiber_val,
               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
        return px.line(title="No data available for CSAT")
    fig = px.line(filtered, x="recorded_at", y="Customer_Satisfaction_Score",
                  labels={"recorded_at": "Time", "Customer_Satisfaction_Score": "CSAT Score"})
    return fig

def update_churn(time_range, region_val, node_val, fiber_val,
                 issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    # df_kpi = filtered[filtered["kpi_name"] == "Churn Prediction & Retention Ratio"]
    if filtered.empty:
        return px.line(title="No data available for Churn Prediction")
    fig = px.line(filtered, x="recorded_at", y="Customer_Satisfaction_Score",
                     labels={"recorded_at": "Time", "kpi_value": "Churn/Retention (%)"})
    return fig
def update_sla( time_range, region_val, node_val, fiber_val,
                    issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
        filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                                issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        if filtered.empty:
            return px.line(title="No data available for Churn Prediction")
        fig = px.area(filtered, x="recorded_at", y="SLA_Priority",
                        labels={"recorded_at": "Time", "kpi_value": "Service Level Agreement"})
        return fig
def update_firstfix( time_range, region_val, node_val, fiber_val,
                    issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
        filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                                issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        if filtered.empty:
            return px.line(title="No data available for Churn Prediction")
        fig = px.line(filtered, x="recorded_at", y="Complaint_Resolution_Time",
                        
                        labels={"recorded_at": "Time", "kpi_value": "Complaint_Resolution_Time"})
        return fig
def update_callcenter( time_range, region_val, node_val, fiber_val,
                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
            return px.line(title="No data available for Churn Prediction")
    fig = px.area(filtered, x="recorded_at", y="Complaint_Resolution_Time",
                      labels={"recorded_at": "Time", "kpi_value": "Call-Center Resolution Time"})
    return fig
def update_selfservice( time_range, region_val, node_val, fiber_val,
                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
            return px.line(title="No data available for Churn Prediction")
    fig = px.area(filtered, x="recorded_at", y="Customer_Satisfaction_Score",
                      
                      labels={"recorded_at": "Time", "kpi_value": "Self - Service Resolution Time"})
    return fig
def update_efficiency( time_range, region_val, node_val, fiber_val,
                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
            return px.line(title="No data available for Churn Prediction")
    fig = px.line(filtered, x="recorded_at", y="Historical_Maintenance_Frequency",
                      labels={"recorded_at": "Time", "kpi_value": "Efficiency Rate"})
    return fig

def update_ticketclosure( time_range, region_val, node_val, fiber_val,
                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    if filtered.empty:
            return px.line(title="No data available for Churn Prediction")
    fig = px.line(filtered, x="recorded_at", y="Complaint_Resolution_Time",
                      labels={"recorded_at": "Time", "kpi_value": "Closure Rate "})
    return fig
def update_planned(time_range, region_val, node_val, fiber_val,
                    issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val):
        filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                                issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        if filtered.empty:
            return px.line(title="No data available for Churn Prediction")
        fig = px.line(filtered, x="recorded_at", y="Historical_Maintenance_Frequency",
                        
                        labels={"recorded_at": "Time", "kpi_value": "Historical_Maintenance_Frequency"})
        return fig

def get_fiber_network_map():
    num_points = 50
    lats = np.random.uniform(29.5, 40, num_points)
    lons = np.random.uniform(-95, -74, num_points)
    regions = np.random.choice(["NY", "LA", "Chicago", "Houston"], num_points)
    fiber_types = np.random.choice(["Single-mode", "Multi-mode", "Aerial", "Underground"], num_points)
    df_map = pd.DataFrame({"lat": lats, "lon": lons, "Region": regions, "Fiber Type": fiber_types})
    fig = px.scatter_mapbox(df_map, lat="lat", lon="lon", color="Region",
                            size_max=15, zoom=3, mapbox_style="open-street-map",
                            title="Fiber Network Map")
    return fig

def get_ai_fault_diagnosis():
    diagnosis = """
### AI Fault Diagnosis & Fix Suggestions
- **🚨 AI Prediction:** Truck Roll Required – Yes
- **🛠 Suggested Fix:** Replace Fiber Connector (View Video/PDF)
- **👨‍🔧 Technician:** Fiber Splicing Expert
    """
    return diagnosis

def get_service_customer_impact():


    # Prepare the data for the pie chart
   

# Prepare data for the pie chart
    pie_data = pd.DataFrame({
        "Status": ["Affected", "Unaffected"],
        "Count": [30, 70]
    })

    # Create the pie chart and include custom data so that click events return the Status value.
    fig1 = px.pie(
        pie_data,
        names="Status",
        values="Count",
        title="Affected vs. Unaffected Customers",
        custom_data=["Status"]  # Add this line to pass Status as custom data
    )
    


        


    days = pd.date_range(end=datetime.now(), periods=14)
    complaints = np.random.randint(5, 20, len(days))
    df_complaints = pd.DataFrame({"Day": days, "Complaints": complaints})
    fig2 = px.area(df_complaints, x="Day", y="Complaints", title="Customer Complaints Trend")
    sla_categories = ["High", "Medium", "Low"]
    downtime_risk = np.random.uniform(0, 50, len(sla_categories))
    df_downtime = pd.DataFrame({"SLA Tier": sla_categories, "Downtime Risk (%)": downtime_risk})
    fig3 = px.bar(df_downtime, x="SLA Tier", y="Downtime Risk (%)", title="Downtime Risk by SLA Priority")
    return fig1, fig2, fig3

def get_technician_truck_analytics():
    days = [str((datetime.now()-timedelta(days=i)).date()) for i in range(7)]
    dispatched = np.random.randint(0, 20, 7)
    avoided = np.random.randint(0, 10, 7)
    df_truck = pd.DataFrame({"Day": days, "Dispatched": dispatched, "Avoided": avoided})
    fig1 = px.bar(df_truck, x="Day", y=["Dispatched", "Avoided"],
                  title="Truck Rolls Dispatched vs. Avoided", barmode="stack")
    days_ts = pd.date_range(end=datetime.now(), periods=14)
    remote = np.random.uniform(30, 120, len(days_ts))
    onsite = np.random.uniform(60, 180, len(days_ts))
    df_resolution = pd.DataFrame({"Day": days_ts, "Remote": remote, "On-Site": onsite})
    fig2 = px.line(df_resolution, x="Day", y=["Remote", "On-Site"],
                   title="Average Resolution Time (Remote vs. On-Site)")
    technicians = ["Tech A", "Tech B", "Tech C", "Tech D"]
    efficiency = np.random.uniform(20, 60, len(technicians))
    df_eff = pd.DataFrame({"Technician": technicians, "Avg Resolution Time (min)": efficiency})
    fig3 = px.bar(df_eff, x="Technician", y="Avg Resolution Time (min)",
                  title="Technician Efficiency")
    return fig1, fig2, fig3

# -------------------- Streamlit App Layout --------------------
# st.set_page_config(page_title="KPI Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>Telecom Fiber Dashboard</h1>", unsafe_allow_html=True)

# Global Filters at the top with reduced size
st.markdown(
    """
    <style>
    .global-filters {
         font-size: 10px;           /* Reduced font size */
         max-width: 400px;          /* Narrower container */
         margin: 0 auto;
         padding: 0.2px;              /* Adjusted padding */
         border: 1px solid #ccc;
         border-radius: 4px;
         background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True
)

# Wrap the container in a div with the global-filters class
st.markdown('<div class="global-filters">', unsafe_allow_html=True)

with st.container():
    st.subheader("Global Filters")
    col1, col2, col3, col4,col5 = st.columns(5)
    time_range = col1.selectbox("Time Range", options=time_range_options, index=0)
    region_val = col2.multiselect("Region", options=region_options)
    node_val = col3.multiselect("Node Type", options=node_type_options)
    fiber_val = col4.multiselect("Fiber Type", options=fiber_type_options)
    issue_val = col5.multiselect("Issue Type", options=issue_type_options)
    
    col6, col7, col8,col9 ,col10= st.columns(5)
    
    tech_val = col6.multiselect("Technician Skill", options=tech_skill_options)
    sla_val = col7.multiselect("SLA Priority", options=sla_priority_options)
    weather_val = col8.multiselect("Weather Conditions", options=weather_options)
    service_val = col9.multiselect("Service Type", options=service_options)
    truck_roll_val = col10.multiselect("Truck Roll Decision", options=truck_roll_options)

st.markdown("</div>", unsafe_allow_html=True)


# Create Tabs for different KPI categories
tabs = st.tabs(["Network Health KPIs", "Customer Experience KPIs", "Operational KPIs", "Comprehensive Overview", "Voice SQL Dashboard"])

# -------------------- Network Health KPIs Tab --------------------
with tabs[0]:
    st.header("Network Health KPIs")
    # Arrange two graphs per row with fixed height (300px)
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Fiber Utilization Rate")
        fig_fiber_util = update_fiber_util(time_range, region_val, node_val, fiber_val,
                                           issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_fiber_util, use_container_width=True, height=300, key="plotly_fiber_util")
    with colB:
        st.subheader("Packet Loss Percentage")
        fig_packet_loss = update_packet_loss(time_range, region_val, node_val, fiber_val,
                                             issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_packet_loss, use_container_width=True, height=300, key="plotly_packet_loss")
    
    colC, colD = st.columns(2)
    with colC:
        st.subheader("Latency Trends")
        fig_latency = update_latency(time_range, region_val, node_val, fiber_val,
                                     issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_latency, use_container_width=True, height=300, key="plotly_latency")
    with colD:
        st.subheader("Signal Strength & Noise Ratio")
        fig_signal = update_signal(time_range, region_val, node_val, fiber_val,
                                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_signal, use_container_width=True, height=300, key="plotly_signal")
    
    colE, colF = st.columns(2)
    with colE:
        st.subheader("% Uptime / Performance")
        fig_uptime = update_uptime(time_range, region_val, node_val, fiber_val,
                                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_uptime, use_container_width=True, height=300, key="plotly_uptime")
    with colF:
        st.subheader("Bandwidth Congestion Analysis")
        fig_bandwidth = update_signal(time_range, region_val, node_val, fiber_val,
                                      issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_bandwidth, use_container_width=True, height=300, key="plotly_bandwidth")

# -------------------- Customer Experience KPIs Tab --------------------
with tabs[1]:
    st.header("Customer Experience KPIs")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Average Complaint Resolution Time")
        fig_complaint = update_complaint(time_range, region_val, node_val, fiber_val,
                                         issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_complaint, use_container_width=True, height=300, key="plotly_complaint")
    with colB:
        st.subheader("Customer Satisfaction Score (CSAT)")
        fig_csat = update_csat(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_csat, use_container_width=True, height=300, key="plotly_csat")
    
    colC, colD = st.columns(2)
    with colC:
        st.subheader("Churn Prediction & Retention Ratio")
        fig_churn = update_churn(time_range, region_val, node_val, fiber_val,
                                 issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_churn, use_container_width=True, height=300, key="plotly_churn")
    with colD:
        st.subheader("Call center Response Time ")
        fig_churn = update_callcenter(time_range, region_val, node_val, fiber_val,
                                 issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_churn, use_container_width=True, height=300, key="plotly_callcenter")  

    colE, colF = st.columns(2)
    with colE:
        st.subheader("Self-Service Resolution Rate")
        fig_churn = update_selfservice(time_range, region_val, node_val, fiber_val,
                                 issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_churn, use_container_width=True, height=300, key="plotly_selfservice")
    with colF:
        st.write("")# Placeholder

# -------------------- Operational KPIs Tab --------------------
with tabs[2]:
    st.header("Operational KPIs")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Technician Efficiency & Response Time")
        fig_eff = update_complaint(time_range, region_val, node_val, fiber_val,
                                   issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_eff, use_container_width=True, height=300, key="plotly_eff")
    with colB:
        st.subheader("SLA Compliance")
        fig_sla = update_sla(time_range, region_val, node_val, fiber_val,
                                issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_sla, use_container_width=True, height=300, key="plotly_sla")

    colC, colD = st.columns(2)
    with colC:
        st.subheader("Ticket Clousure Rate")
        fig_churn = update_ticketclosure(time_range, region_val, node_val, fiber_val,
                                 issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_churn, use_container_width=True, height=300, key="plotly_tc")
    with colD:
        st.subheader("First-Fix Rate")
        fig_churn = update_firstfix(time_range, region_val, node_val, fiber_val,
                                 issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_churn, use_container_width=True, height=300, key="plotly_ff") 

    colE, colF = st.columns(2)
    with colE:
        st.subheader("Planned VS Unplanned Truck Rolls")
        fig_churn = update_planned(time_range, region_val, node_val, fiber_val,
                                 issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
        st.plotly_chart(fig_churn, use_container_width=True, height=300, key="plotly_pup")
        
        
    with colF:
        st.write("")
        

# -------------------- Comprehensive Overview Tab --------------------
with tabs[3]:
    st.header("Comprehensive Overview")
    filtered = get_filtered_df(time_range, region_val, node_val, fiber_val,
                               issue_val, tech_val, sla_val, weather_val, service_val, truck_roll_val)
    generate_overview_cards(filtered)
    # colA, colB = st.columns(2)
    # with colA:
    st.write("Fiber Network Map:")
    fig_network = get_fiber_network_map()
    st.plotly_chart(fig_network, use_container_width=True, height=300, key="plotly_network_map")

    # with colB:
    #     st.write("")
    #     # st.markdown(get_ai_fault_diagnosis())
    st.write("Service & Customer Impact:")
    colc, colD = st.columns(2)
    fig1, fig2, fig3 = get_service_customer_impact()
    with colc:
        
        st.plotly_chart(fig1, use_container_width=True, height=300, key="plotly_service1")
    with colD:
        st.plotly_chart(fig2, use_container_width=True, height=300, key="plotly_service2")
    st.plotly_chart(fig3, use_container_width=True, height=300, key="plotly_service3")
    st.write("Technician & Truck Roll Analytics:")
    colE, colF = st.columns(2)
    with colE:
        tfig1, tfig2, _ = get_technician_truck_analytics()
        st.plotly_chart(tfig1, use_container_width=True, height=300, key="plotly_tech1")
    with colF:
        _, _, tfig3 = get_technician_truck_analytics()
        st.plotly_chart(tfig3, use_container_width=True, height=300, key="plotly_tech3")

# -------------------- Voice SQL Dashboard Tab --------------------
with tabs[4]:
    st.header("Voice SQL Dashboard")
   

    
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = pd.DataFrame()
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎤 Start Recording"):
            st.session_state.transcript = recognize_speech()
    with col2:
        transcript = st.text_area("Transcript", value=st.session_state.get('transcript', ''))
    if st.button("Extract Data"):
        if st.session_state.get('transcript') and "Could not recognize" not in st.session_state.get('transcript'):
            try:
                sql_query = generate_sql(st.session_state.transcript)
                st.session_state.audio_data = fetch_data(sql_query)
            except Exception as e:
                st.error(f"SQL Error: {str(e)}")
    if not st.session_state.audio_data.empty:
        st.subheader("Query Results")
        st.dataframe(st.session_state.audio_data)
        st.subheader("Visualization")
        cols = st.session_state.audio_data.columns.tolist()
        col1, col2, col3 = st.columns(3)
        x_axis = col1.selectbox("X Axis", options=cols, index=0)
        y_axis = col2.selectbox("Y Axis", options=cols, index=1 if len(cols) > 1 else 0)
        plot_type = col3.selectbox("Plot Type", GRAPH_TYPES, format_func=lambda x: x['label'])
        if st.button("Generate Plot"):
            try:
                fig = getattr(px, plot_type['value'])(st.session_state.audio_data, x=x_axis, y=y_axis)
                st.plotly_chart(fig, use_container_width=True, height=300, key="plotly_voice_plot")
            except Exception as e:
                st.error(f"Plotting Error: {str(e)}")
