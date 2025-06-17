


def validate_input(data, required_fields):
    """Validate and sanitize input data"""
    if not data:
        return False, "No data provided"
    
    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    
    # Sanitize strings (basic example)
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            data[key] = value.strip()[:1000]  # Limit length
            
            # Check for suspicious patterns
            suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', 'onload=']
            if any(pattern in value.lower() for pattern in suspicious_patterns):
                return False, f"Invalid content in field: {key}"
    
    return True, "Valid"