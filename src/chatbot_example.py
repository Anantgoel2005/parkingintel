"""
ParkingIntel Chatbot — answers questions about the dashboard, data, and findings.
Uses DeepSeek API with full system context.

SETUP: Set your DEEPSEEK_API_KEY environment variable before running.
"""

import streamlit as st
import json
import os
import requests

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are the ParkingIntel assistant, an AI that helps users understand 
Bangalore's parking enforcement data and the ParkingIntel dashboard.

ABOUT THE DASHBOARD:
ParkingIntel analyzes 298,450 parking violations recorded by Bangalore Traffic Police 
between November 2023 and April 2024 (152 days). It covers 54 police stations across 
Bangalore with GPS coordinates, timestamps, vehicle types, violation types, and 
validation statuses.

THE SIX TABS:
1. Hotspot Map — Interactive street map with violation heatmaps and DBSCAN-detected 
   hotspot clusters. Filter by station, time, day, and vehicle type.
2. Station Comparison — Compare 54 stations across multiple dimensions: ticket accuracy, 
   hours active, processing speed, weekday/weekend balance, and zone difficulty.
3. Time Patterns — Day x Hour heatmaps revealing when violations are recorded. 
   4-7 AM shows 4x more violations than 5-8 PM.
4. Gap Analysis — Three methods to identify areas that may benefit from additional 
   enforcement attention: vehicle diversity, time coverage, and junction density.
5. Vehicle Profiles — Each station's vehicle-type profile with contextual enforcement notes.
6. Weekday vs Weekend — Observed split between weekday and weekend recorded violations.

THE FIVE KEY FINDINGS:
1. 4-7 AM Spike — 104,685 violations recorded 4-7 AM vs 25,336 at 5-8 PM (4x difference).
   This may reflect overnight parking, enforcement sweep patterns, or both.
2. Quality Variation — Rejection rates range from 21% to 42% across stations.
3. Processing Times — Median processing ranges from 25 to 42 hours across stations.
4. Vehicle Mix — City Market is 41% scooters; Malleshwaram is 47% cars. Different zones 
   have different vehicle profiles.
5. Weekend Split — Stations range from 15.7% to 43.3% weekend share.

HOW TO USE:
- Use sidebar filters to narrow by station, hour, day, and vehicle type
- Click through the 6 tabs for different analyses
- On the map, toggle density overlay and hotspot clusters
- Station Comparison shows multi-dimensional metrics (not officer rankings)
- Gap Analysis flags areas for further review, not conclusions

Keep responses concise (2-4 sentences). Reference specific data when possible.
If asked something you don't know, say so.
"""


def init_chat():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hi! I'm the ParkingIntel assistant. "
             "Ask me anything about the dashboard, the data, or Bangalore's parking "
             "enforcement patterns."}
        ]


def call_deepseek(messages):
    try:
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Sorry, I couldn't reach the AI service. ({str(e)[:80]})"


def render_chat(key_suffix=""):
    init_chat()
    st.markdown("### Ask ParkingIntel")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about the data...", key=f"parking_chat_{key_suffix}"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        api_messages.extend(st.session_state.chat_messages[-10:])
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                reply = call_deepseek(api_messages)
            st.markdown(reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
    if len(st.session_state.chat_messages) > 1:
        if st.button("Clear chat", key=f"clear_chat_{key_suffix}"):
            st.session_state.chat_messages = []
            st.rerun()
