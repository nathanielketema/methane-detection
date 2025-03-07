# Methane Leak Detection Project

This project implements a quantum-enhanced machine learning system for detecting and tracking methane leaks using drone-based sensors. The system uses Gaussian Process Regression with quantum computing optimization to predict methane concentrations and guide drone navigation.

## Features

- Data processing and feature engineering for methane sensor data
- Quantum-enhanced Gaussian Process Regression for concentration prediction
- Intelligent drone navigation system
- Visualization of methane distribution patterns
- Memory-efficient handling of large datasets

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- qiskit (for quantum computing features)

## Project Structure

```plaintext
├── data/
│   ├── raw/             # Raw sensor data
│   └── processed/       # Processed and cleaned data
├── src/
│   ├── data_processing.py      # Data cleaning and preprocessing
│   ├── feature_engineering.py  # Feature extraction and engineering
│   ├── models/
│   │   └── quantum_gaussian_regression.py
│   │   └── predictive_route_planner.py
│   ├── drone_navigation.py     # Drone path planning
│   ├── visualization.py        # Data visualization
│   └── config.py               # Configuration parameters
├── methane_leak_detection.py         # Main execution script
├── requirements.txt
├── LICENSE
└── README.md

```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nathanielketema/methane-leak-detection.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python3 methane_leak_detection.py
```

This will:
1. Process the raw methane sensor data
2. Extract relevant features
3. Train the quantum-enhanced regression model
4. Simulate drone navigation decisions
5. Generate visualizations of methane distribution

## Data Format

The input data should include the following columns:
- Time (UTC)
- Latitude and longitude
- Wind components (u_west_to_east_wind, v_south_to_north_wind)
- Temperature
- Tracer concentration (target variable)

## License

Copyright © 2025. All Rights Reserved.

This project and its contents are proprietary and confidential. No part of this project may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright holder.

Unauthorized copying, modification, distribution, or use of this software is strictly prohibited.

## Contact

nathanielketema@gmail.com

