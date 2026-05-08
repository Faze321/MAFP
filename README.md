# Multi-Agent Prescriptive Forecasting (MAPF)

This project implements the conference-work requirement in `Conference work.docx`:

1. Select five UrbanEV zones that behave like CBD/office, residential, transport hub, commercial/mall, and industrial demand profiles.
2. Build compact context snippets from `volume-11kW.csv`, `weather_airport.csv`, `poi.csv`, `inf.csv`, `occupancy.csv`, and `s_price.csv`.
3. Run a sequential multi-agent chain per zone:
   - Grid Analyst: forecast 1-4 days of load and assign a grid stress level.
   - Behavioural Agent: explain the demand drivers from POI mix, weather, and time markers.
   - Market Economist: prescribe a service-fee shift from stress and elasticity proxies.
4. Execute all five zone chains concurrently with `asyncio`.
5. Export an explainability table with predicted vs. actual load, rationale, and price shift.

The OpenRouter call path uses the OpenAI Python SDK with `base_url=https://openrouter.ai/api/v1`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For local validation without an API key:

```powershell
python main.py --dry-run
```

For OpenRouter-backed agents:

```powershell
Copy-Item config.example.json config.json
# Edit config.json and set openrouter.api_key / openrouter.model
python main.py
```

Useful options:

```powershell
python main.py --dry-run --horizon-days 4 --history-days 7
python main.py --config config.json --model anthropic/claude-sonnet-4.5 --forecast-start "2023-02-25 00:00:00"
python main.py --force-cache
```

## Outputs

Generated files are written to `output/`:

- `selected_zones.csv`: the five selected zones and the proxy features used for selection.
- `context_snippets.json`: token-efficient context passed to each agent.
- `rationale_trace.csv`: machine-readable explainability table.
- `rationale_trace.md`: markdown table for a report or paper appendix.
- `rationale_trace.json`: full structured agent outputs.

The first full run builds cached POI-to-zone assignments in `output/cache/`. Later runs reuse that cache unless `--force-cache` is passed.

## Data Notes

The POI file in this release contains only three broad POI labels: `food and beverage services`, `business and residential`, and `lifestyle services`. The five zone categories are therefore selected as operational proxies:

- CBD / Office: high business/residential density plus morning/noon load shape.
- Residential: night/evening charging plateau.
- Transport Hub: high charging capacity and bursty peaks.
- Commercial / Mall: high food/lifestyle density plus evening/weekend lift.
- Industrial: stable high base load with large charging capacity.

## Data Folder

**data**: 1-hour resolution zone-level data of the UrbanEV dataset, which has been cleaned through outlier detection, zero-value checks, etc., and includes data from **275 zones**, **1,362 charging stations**, and **17,532 charging piles**.

* `adj.csv`: Adjacency matrix.
* `duration.csv`: Hourly EV charging duration (Unit: hour).
* `e_price.csv`: Electricity price (Unit: Yuan/kWh).
* `inf.csv`: Filtered station-level data for the 275 zones, including coordinates, charging capacities, area (Unit: m^2), and perimeter (Unit: m).
* `inf_raw.csv`: All station-level data for the same 275 zones, including coordinates, charging capacities, area (Unit: m^2), and perimeter (Unit: m).
* `occupancy.csv`: Hourly EV charging occupancy rate (Unit: %).
* `s_price.csv`: Service price (Unit: Yuan/kWh).
* `volume.csv`: Hourly EV charging volume (Unit: kWh). The volume in *volume.csv* is derived from the rated power of charging piles
* `volume-11kW.csv` provides an alternative vehicle-side estimation of charging volume to mitigate potential overestimation in `volume.csv`. Specifically, for direct current charging stations, the volume is calculated using the standard power of the most commonly used electric vehicle, Tesla Model Y (11kW), instead of the rated power of the charging pile.
* `weather_airport.csv`: Weather data from the meteorological station at Bao'an Airport (Shenzhen). These are the raw data collected, and it is recommended to use the **Max-Min** method for normalization.
* `weather_central.csv`: Weather data from Futian Meteorological Station in the city center of Shenzhen.
* `weather_header.txt`: Descriptions of the table headers in `weather_airport.csv` and `weather_central.csv`.
* `distance.csv`: Distance matrix between the 275 zones.
* `poi.csv`: Points of Interest categorized into three types: `food and beverage services`, `business and residential`, and `lifestyle services`. The coordinates used are based on the `WGS84` coordinate system.
* Notes: Our occupancy data is gathered from an availability perspective, while the duration and volume data is collected from a utilization standpoint. Specifically, the occupancy data records all unavailable or busy charging piles. In contrast, the duration and volume data only account for the piles actively providing electricity. You can select the data according to your research purpose.
