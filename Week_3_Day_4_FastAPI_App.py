"""
Week 3 - Day 4
FastAPI Service Setup for Customer LTV Prediction

This module initializes the FastAPI application and provides
basic service endpoints for API health monitoring.
"""

from fastapi import FastAPI


app = FastAPI(
    title="Customer LTV Prediction API",
    description=(
        "A FastAPI service for the Customer Churn Prediction "
        "and Lifetime Value Prediction project."
    ),
    version="1.0.0"
)


@app.get("/", tags=["General"])
def read_root():
    """Return basic API information."""
    return {
        "message": "Customer LTV Prediction API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Check whether the API service is operational."""
    return {
        "status": "healthy",
        "service": "Customer LTV Prediction API"
    }