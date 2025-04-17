
def divide_numbers(a, b):
    """Divide a by b without any checks."""
    return a / b  # Potential zero division

def format_user(name, age):
    """Format user information."""
    return name + ": " + age  # Potential type error

class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.data = []
    
    def process_data(self, items):
        """Process a list of items."""
        result = ""
        for item in items:
            result += str(item) + ", "  # String concatenation in loop
        return result.rstrip(", ")
