"""HarvestIQ Validation Utilities."""

from typing import Any, Dict, List, Tuple
from fastapi.exceptions import RequestValidationError


# Agro-climatic parameter ranges documented in HarvestIQ specifications
PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    "nitrogen": {
        "min": 0.0,
        "max": 200.0,
        "unit": "kg/ha",
        "label": "Nitrogen (N)",
        "description": "Ratio of Nitrogen content in soil",
    },
    "phosphorus": {
        "min": 0.0,
        "max": 200.0,
        "unit": "kg/ha",
        "label": "Phosphorus (P)",
        "description": "Ratio of Phosphorus content in soil",
    },
    "potassium": {
        "min": 0.0,
        "max": 300.0,
        "unit": "kg/ha",
        "label": "Potassium (K)",
        "description": "Ratio of Potassium content in soil",
    },
    "temperature": {
        "min": 0.0,
        "max": 50.0,
        "unit": "°C",
        "label": "Temperature",
        "description": "Ambient temperature in degrees Celsius",
    },
    "humidity": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "label": "Humidity",
        "description": "Relative humidity percentage",
    },
    "ph": {
        "min": 3.0,
        "max": 10.0,
        "unit": "pH",
        "label": "Soil pH",
        "description": "Soil pH value (acidity/alkalinity)",
    },
    "rainfall": {
        "min": 0.0,
        "max": 400.0,
        "unit": "mm",
        "label": "Rainfall",
        "description": "Precipitation in millimeters",
    },
}


def validate_crop_inputs(data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate a dictionary of crop recommendation inputs against defined physiological ranges.

    Returns:
        (is_valid, errors_list)
    """
    errors: List[Dict[str, Any]] = []

    for param, bounds in PARAMETER_RANGES.items():
        if param not in data or data[param] is None:
            errors.append({
                "field": param,
                "message": f"'{bounds['label']}' is required.",
                "invalid_value": None,
                "allowed_range": f"{bounds['min']} – {bounds['max']} {bounds['unit']}",
            })
            continue

        try:
            val = float(data[param])
        except (ValueError, TypeError):
            errors.append({
                "field": param,
                "message": f"'{bounds['label']}' must be a valid numeric value.",
                "invalid_value": data[param],
                "allowed_range": f"{bounds['min']} – {bounds['max']} {bounds['unit']}",
            })
            continue

        if val < bounds["min"] or val > bounds["max"]:
            errors.append({
                "field": param,
                "message": (
                    f"'{bounds['label']}' value {val} is outside valid physiological range "
                    f"[{bounds['min']} – {bounds['max']} {bounds['unit']}]."
                ),
                "invalid_value": val,
                "allowed_range": f"{bounds['min']} – {bounds['max']} {bounds['unit']}",
            })

    return len(errors) == 0, errors


def format_pydantic_validation_error(exc: RequestValidationError) -> Dict[str, Any]:
    """
    Format FastAPI/Pydantic RequestValidationError into meaningful, farmer-friendly error structures.
    """
    formatted_errors: List[Dict[str, Any]] = []

    for err in exc.errors():
        loc = err.get("loc", [])
        # Extract field name from loc (e.g., ['body', 'nitrogen'])
        field = str(loc[-1]) if loc else "unknown"
        msg = err.get("msg", "Validation error")
        val = err.get("input")

        # Enrich with parameter boundary metadata if known
        bounds = PARAMETER_RANGES.get(field.lower())
        allowed_range = (
            f"{bounds['min']} – {bounds['max']} {bounds['unit']}"
            if bounds
            else None
        )

        # Generate readable guidance message
        if bounds and "greater_than_equal" in err.get("type", ""):
            custom_msg = f"{bounds['label']} cannot be less than {bounds['min']} {bounds['unit']}."
        elif bounds and "less_than_equal" in err.get("type", ""):
            custom_msg = f"{bounds['label']} cannot exceed {bounds['max']} {bounds['unit']}."
        elif bounds and "missing" in err.get("type", ""):
            custom_msg = f"{bounds['label']} is required."
        else:
            custom_msg = msg

        formatted_errors.append({
            "field": field,
            "message": custom_msg,
            "invalid_value": val,
            "allowed_range": allowed_range,
        })

    return {
        "success": False,
        "detail": "Input parameter validation failed. Please check the values against agro-climatic boundaries.",
        "errors": formatted_errors,
    }
