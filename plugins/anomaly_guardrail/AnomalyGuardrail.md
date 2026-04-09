# Anomaly Guardrail Plugin

## Overview
This plugin, its configuration, and utilities provide a modular anomaly detection system that integrates configurable statistical methods with severity classification and log management. It includes:

- **Plugin Engine** (`plugin/anomaly_guardrail_plugin.py`): Core anomaly detection and classification logic.
- **Plugin Configuration** (`onair/config/config_anomaly_guardrail.ini`): Centralized setup for monitored columns, thresholds, severity scores, and action mappings.
- **Publisher Simulator** (`redis-experiment-publisher-anomalies.py`): Publishes both normal and anomalous telemetry into Redis channels for testing.
- **Log Post-Processor** (`count_anomalies.py`): Cleans and summarizes anomaly event logs after plugin execution.

---

## Features
- Outlier detection using Z-score or Interquartile Range (IQR).
- Frozen value detection for repeated identical readings (“stuck” signals).
- Severity classification of anomalies: `minor`, `moderate`, `critical`.
- Configurable severity-to-action mapping, e.g., `minor:LOG_ONLY, moderate:REDUCE_LOAD, critical:SAFE_MODE`.
- Cooldown interval to avoid repetitive alerts.
- CSV audit logging of anomaly events.
- Redis-based test publisher for generating test streams.
- Log summarization tool to collapse redundant anomaly runs.

---

## File Roles

| File | Location | Purpose |
|------|----------|---------|
| `plugin/anomaly_guardrail_plugin.py` | `plugin/` | Core anomaly detection and classification |
| `OnAIR/config/config_anomaly_guardrail.ini` | onair/config/ | Configuration of thresholds, severity mapping, and columns |
| `redis-experiment-publisher-anomalies.py` | Root | Simulator that publishes normal + anomalous telemetry samples |
| `count_anomalies.py` | Root | Post-processor that cleans and summarizes anomaly logs |

---

## Usage

### 1. Configure
Adjust detection parameters in `OnAIR/config/config_anomaly_guardrail.ini`:

```ini
[anomaly_guardrail]
columns                 = 1,2,3,4,5,6
method                  = zscore
z_threshold             = 3.0
stuck_threshold         = 25
severity_moderate_score = 2.5
severity_critical_score = 4.0
severity_actions        = minor:LOG_ONLY,moderate:REDUCE_LOAD,critical:SAFE_MODE
```

### 2. Publish Test Data
Generate mixed normal and anomalous telemetry for testing:

```bash
python3 redis-experiment-publisher-anomalies.py --normals 100 --anomalies 50 --repeat 3 --shuffle
```

### 3. Run the Plugin
Execute the plugin. Anomaly events are written to CSV in the `logs/` directory:

```bash
python driver.py onair/config/config_anomaly_guardrail.ini
```

### 4. Post-Process Logs
Summarize anomaly runs and collapse redundant events:

```bash
python3 count_anomalies.py --in logs/anomaly_events.csv --out logs/anomaly_events_clean.csv --mode edge --emit-end
```

---

## Applications
- Real-time telemetry guardrails  
- Safety validation and fault detection  
- System health monitoring  
- Severity-based anomaly handling  
