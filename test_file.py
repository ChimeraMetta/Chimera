from typing import List, Dict, Tuple, Optional, Union

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def format_name(first: str, last: str) -> str:
    """Format a full name from first and last names."""
    return first + " " + last

def calculate_total(items: List[str], prices: List[float]) -> float:
    """Calculate the total price of items."""
    total = 0.0
    for i, item in enumerate(items):
        total += prices[i]
    return total

def format_receipt(items: List[str], prices: List[float]) -> str:
    """Format a receipt with items and prices."""
    receipt = "Receipt:\n"
    for i, item in enumerate(items):
        # Type error: concatenating string and number
        receipt += item + ": $" + prices[i] + "\n"
    
    # Type error: concatenating string and number
    receipt += "Total: $" + calculate_total(items, prices) + "\n"
    return receipt

def process_data(data: Dict[str, Union[str, int]]) -> Tuple[str, int]:
    """Process a data dictionary and return a tuple of results."""
    name = data.get("name", "")
    value = data.get("value", 0)
    
    # Type error: adding string and integer
    result = name + value
    
    return (name, value)

def divide_values(a: float, b: float) -> float:
    """Divide a by b."""
    # Potential division by zero
    return a / b

def get_discount(price: float, percent: float = 10.0) -> float:
    """Calculate discount amount."""
    return price * (percent / 100)

def apply_discount(price: float, discount_func=get_discount) -> float:
    """Apply a discount to a price."""
    return price - discount_func(price)

def print_items(items: List[str]) -> None:
    """Print each item in the list."""
    for item in items:
        print(item)

# Test variables and function calls
sample_items = ["Apples", "Milk", "Bread"]
sample_prices = [3.99, 2.50, 1.99]

# Type error: calling len with a number
item_count = len(5)

total_price = calculate_total(sample_items, sample_prices)
receipt = format_receipt(sample_items, sample_prices)

# Division by zero error
zero_division = divide_values(10, 0)

# More complex type error - mixing incompatible types in operations
data = {"name": "Test", "value": 42}
name, value = process_data(data)

# Function that has potential type errors in implementation
def problematic_function(value: Union[str, int]) -> str:
    # This might cause runtime errors depending on the input type
    result = "Value: " + value
    return result

# Call with string - would work
print(problematic_function("test"))

# Call with int - would fail at runtime
print(problematic_function(42))