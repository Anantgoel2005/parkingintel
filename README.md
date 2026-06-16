# 🚗 ParkingIntel

**AI-Driven Parking Hotspot Intelligence for Bangalore Traffic Police**

> Theme 1: Poor Visibility on Parking-Induced Congestion  
> Hackathon Submission — June 2025

---

## Overview

ParkingIntel analyzes **298,450 parking violations** recorded across **54 police stations** in Bangalore (November 2023 – April 2024). It detects illegal parking hotspots using DBSCAN clustering and provides temporal, spatial, and vehicle-type analysis to enable targeted enforcement.

**Not a heatmap. Not a ranking tool. An intelligence platform to help officers deploy to the right place at the right time.**

---

## Quickstart

```bash
git clone https://github.com/Anantgoel2005/parkingintel.git
cd parkingintel
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Download the dataset from the hackathon organizers
# Place it in: data/raw/jan to may police violation_anonymized791b166.csv

streamlit run app/dashboard.py
```

Open http://localhost:8501

---

## Features

| Tab | Description |
|---|---|
| 🗺️ **Hotspot Map** | Interactive street map with density overlay + DBSCAN hotspot clusters. Filter by station, time, day, and vehicle type |
| 📊 **Station Comparison** | Multi-dimensional comparison across 54 stations — ticket accuracy, hours active, processing speed, zone difficulty |
| ⏰ **Time Patterns** | Day×Hour heatmaps and hourly distribution charts. Identifies the 4-7 AM spike pattern |
| 🔍 **Gap Analysis** | Three indirect methods to identify areas that may benefit from additional enforcement attention |
| 🚘 **Vehicle Profiles** | Per-station vehicle type breakdowns with contextual enforcement notes |
| 📅 **Weekday vs Weekend** | Observed weekday/weekend split per station — observations, not prescriptions |

---

## Key Findings

1. **4-7 AM Spike** — 104,685 violations recorded 4-7 AM vs 25,336 at 5-8 PM (4× difference). Consistent across all stations and vehicle types
2. **Quality Variation** — Rejection rates range from 21.3% to 41.7% across stations
3. **Processing Times** — Median processing ranges from 25 to 42 hours
4. **Vehicle Mix** — City Market: 41% scooters. Malleshwaram: 47% cars. Each station has a distinct vehicle profile
5. **Weekend Split** — Weekend share ranges from 15.7% (HAL Old Airport) to 43.3% (K.G. Halli)

---

## Tech Stack

| Component | Tool |
|---|---|
| Data Processing | Pandas, NumPy |
| Clustering | DBSCAN (scikit-learn) |
| Maps | Folium (OpenStreetMap) |
| Charts | Plotly |
| Dashboard | Streamlit |
| AI Assistant | DeepSeek (chatbot) |

---

## Project Structure

```
parkingintel/
├── app/dashboard.py           # Streamlit dashboard
├── src/
│   ├── data_loader.py         # CSV → clean DataFrame pipeline
│   ├── analytics.py           # Station metrics, temporal, vehicle analysis
│   ├── clustering.py          # DBSCAN hotspot detection
│   ├── blindspots.py          # Gap analysis (3 methods)
│   ├── visualization.py       # Folium maps + Plotly charts
│   ├── chatbot_example.py     # AI chatbot (requires API key)
│   └── zone_classifier.py     # Vehicle-based zone classification
├── data/                      # Dataset (gitignored — download separately)
├── requirements.txt
└── README.md
```

---

## Setup: AI Chatbot (Optional)

The dashboard includes an AI assistant that answers questions about the data. To enable:

1. Get a DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com)
2. Set the environment variable: `set DEEPSEEK_API_KEY=sk-...`
3. Copy `src/chatbot_example.py` to `src/chatbot.py`
4. Restart the dashboard

---

## Important Notes

- **Not a ranking tool**: Station comparison metrics identify where the *system* may need support (better devices, training, or resources). They are not officer performance rankings
- **Observational**: Gap analysis methods flag unusual data patterns for review. They do not conclusively prove enforcement gaps
- **Single dataset**: All analysis uses only the provided violation records. No external traffic, land-use, or staffing data was incorporated

---

## License

MIT
