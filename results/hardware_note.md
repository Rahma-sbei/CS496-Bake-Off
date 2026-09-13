# Hardware note — local model (qwen2.5:7b via Ollama)

- **Machine:** [TODO: teammate to fill in — laptop CPU/GPU model, RAM]
- **Ollama version:** [TODO: run `ollama --version` on that machine]
- **Where it ran:** teammate's laptop
- **Hourly cost used in the report:** $0.0045/hour
  - **How we got this number:** estimated, not measured. Assumed 65W draw for a laptop
    running a 7B model (CPU/modest GPU) × Tunisia's residential electricity rate
    ($0.069/kWh, STEG, per globalpetrolprices.com) = 0.065 kW × $0.069/kWh = $0.0045/hour.
  - If you can get a real measured wattage (`upower -i /org/freedesktop/UPower/devices/battery_BAT0`
    while the eval is running, or check the laptop's charger label), swap it in — real beats
    estimated, but a documented estimate is still honest.
- **Measured throughput:** p50 latency = [X] ms, tokens/sec = [X] (from `results/summary.csv`
  once the full 3-model run is merged and `report_generator.py` is run)
