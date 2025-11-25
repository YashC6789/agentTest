from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b and return the result. Raises an error if b = 0."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"