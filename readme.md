# Penguin Species Prediction API

A simple Machine Learning API that predicts the species of a penguin based on its physical measurements.

The project uses the **Palmer Penguins dataset**, trains a Random Forest classification model, saves the trained model using Joblib, and exposes the model through a **FastAPI**.

---

## Project Overview

The API takes four physical measurements of a penguin:

- Bill length
- Bill depth
- Flipper length
- Body mass

and predicts the penguin species.

### Prediction Flow

```text
Penguin Measurements
        ↓
    FastAPI API
        ↓
  Trained ML Model
        ↓
 Species Prediction