# Global Bikeability Benchmarking: Comparative Analysis of 100 Cities Using NetAScore

This repository contains the code used in the Master's thesis **"Global Bikeability Benchmarking: Comparative Analysis of 100 Cities Worldwide Using NetAScore"**. It supports the development of scalable, open, and globally comparable evaluations of urban cycling infrastructure.

## Overview

To promote active mobility and support Sustainable Development Goal 11.2, cities must improve their cycling networks. This project assesses bikeability in 100 cities across the globe using NetAScore — an open-source scoring method based on OpenStreetMap data. It incorporates key performance indicators for:

- **Bikeability**: Evaluates segment-level infrastructure quality.
- **Connectivity**: Measures structural coherence of cycling networks.
- **Accessibility**: Assesses how well destinations are reached via bike.

Cities are clustered based on these indicators to reveal global patterns and typologies.

## Repository Structure

```bash
.
├── workflow.py          # Main script for bikeability, connectivity & accessibility computation
├── clustering.py        # Clustering analysis based on computed indicators
└── README.md            # Project documentation

