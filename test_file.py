from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import math
import datetime
from collections import defaultdict

# Domain-specific types for better reasoning
class Product:
    """Represents a product in inventory."""
    def __init__(self, name: str, price: float, category: str):
        self.name = name
        self.price = price
        self.category = category
    
    def apply_discount(self, percent: float) -> float:
        """Apply discount to the product price."""
        return self.price * (1 - percent/100)

class DiscountedProduct(Product):
    """A product with a permanent discount applied."""
    def __init__(self, name: str, price: float, category: str, discount: float):
        super().__init__(name, price, category)
        self.discount = discount
    
    def get_final_price(self) -> float:
        """Get price after permanent discount."""
        return self.apply_discount(self.discount)

class ShoppingCart:
    """Shopping cart containing products."""
    def __init__(self):
        self.items: List[Product] = []
    
    def add_item(self, product: Product) -> None:
        """Add product to cart."""
        self.items.append(product)
    
    def remove_item(self, product_name: str) -> bool:
        """Remove product from cart by name."""
        for i, item in enumerate(self.items):
            if item.name == product_name:
                self.items.pop(i)
                return True
        return False
    
    def calculate_total(self) -> float:
        """Calculate total price of items in cart."""
        return sum(item.price for item in self.items)

# Helper functions
def format_currency(amount: float) -> str:
    """Format a number as currency."""
    return f"${amount:.2f}"

def calculate_tax(amount: float, rate: float = 8.5) -> float:
    """Calculate tax amount based on rate percentage."""
    return amount * (rate / 100)

def parse_date(date_str: str) -> datetime.date:
    """Parse a date string in YYYY-MM-DD format."""
    year, month, day = map(int, date_str.split('-'))
    return datetime.date(year, month, day)

def days_between_dates(start_date: str, end_date: str) -> int:
    """Calculate days between two dates."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    return (end - start).days

# Utility module with related functions
class StringUtils:
    @staticmethod
    def capitalize_words(text: str) -> str:
        """Capitalize each word in a string."""
        return ' '.join(word.capitalize() for word in text.split())
    
    @staticmethod
    def truncate(text: str, max_length: int = 50) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

class MathUtils:
    @staticmethod
    def round_to_nearest(value: float, nearest: float = 1.0) -> float:
        """Round a value to the nearest specified amount."""
        return round(value / nearest) * nearest
    
    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """Calculate percentage of a value against total."""
        if total == 0:
            return 0
        return (value / total) * 100

# Function with higher-order function parameter
def apply_discount_strategy(
    products: List[Product], 
    strategy: Callable[[Product], float]
) -> List[Tuple[Product, float]]:
    """Apply a discount strategy function to a list of products."""
    return [(product, strategy(product)) for product in products]

def bulk_discount_strategy(product: Product) -> float:
    """A simple discount strategy for bulk purchases."""
    if product.category == "Electronics":
        return product.price * 0.9  # 10% off electronics
    elif product.category == "Clothing":
        return product.price * 0.85  # 15% off clothing
    return product.price * 0.95  # 5% off everything else

def process_order(
    cart: ShoppingCart,
    customer_info: Dict[str, Any],
    apply_discount: bool = False
) -> Dict[str, Any]:
    """Process a customer order and return order details."""
    subtotal = cart.calculate_total()
    
    # Calculate discount if applicable
    discount_amount = 0.0
    if apply_discount:
        if subtotal > 100:
            discount_amount = subtotal * 0.1  # 10% off for orders over $100
    
    # Calculate tax
    tax_amount = calculate_tax(subtotal - discount_amount)
    
    # Calculate final total
    total = subtotal - discount_amount + tax_amount
    
    # Create order result
    return {
        "order_id": f"ORD-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "customer": customer_info,
        "items": cart.items,
        "subtotal": format_currency(subtotal),
        "discount": format_currency(discount_amount),
        "tax": format_currency(tax_amount),
        "total": format_currency(total),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_receipt(order: Dict[str, Any]) -> str:
    """Generate a receipt string from order details."""
    receipt = "=== RECEIPT ===\n"
    receipt += f"Order: {order['order_id']}\n"
    receipt += f"Date: {order['date']}\n"
    receipt += f"Customer: {order['customer']['name']}\n\n"
    
    receipt += "Items:\n"
    for item in order['items']:
        receipt += f"  {item.name}: {format_currency(item.price)}\n"
    
    receipt += f"\nSubtotal: {order['subtotal']}\n"
    receipt += f"Discount: {order['discount']}\n"
    receipt += f"Tax: {order['tax']}\n"
    receipt += f"Total: {order['total']}\n"
    
    return receipt

# Some additional functions that demonstrate type handling
def merge_data(
    primary_data: Dict[str, Any], 
    secondary_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two dictionaries with priority to primary_data."""
    result = secondary_data.copy()
    result.update(primary_data)
    return result

def filter_products_by_category(
    products: List[Product], 
    category: str
) -> List[Product]:
    """Filter products by category."""
    return [p for p in products if p.category == category]

def sort_products_by_price(
    products: List[Product], 
    descending: bool = False
) -> List[Product]:
    """Sort products by price."""
    return sorted(products, key=lambda p: p.price, reverse=descending)

# Demo usage
if __name__ == "__main__":
    # Create some products
    laptop = Product("Laptop", 899.99, "Electronics")
    headphones = Product("Headphones", 149.99, "Electronics")
    tshirt = DiscountedProduct("T-Shirt", 19.99, "Clothing", 20)
    
    # Create a shopping cart
    cart = ShoppingCart()
    cart.add_item(laptop)
    cart.add_item(headphones)
    cart.add_item(tshirt)
    
    # Process the order
    customer = {"name": "John Doe", "email": "john@example.com"}
    order = process_order(cart, customer, apply_discount=True)
    
    # Generate and print receipt
    receipt = generate_receipt(order)
    print(receipt)
    
    # Demonstrate some other function calls
    days = days_between_dates("2023-01-01", "2023-12-31")
    print(f"Days in 2023: {days}")
    
    # Apply different discount strategy
    discounted_products = apply_discount_strategy(cart.items, bulk_discount_strategy)
    for product, price in discounted_products:
        print(f"{product.name}: Original: {format_currency(product.price)}, Discounted: {format_currency(price)}")
    
    # Use utility classes
    formatted_name = StringUtils.capitalize_words("john doe smith")
    print(f"Formatted name: {formatted_name}")
    
    percentage = MathUtils.calculate_percentage(75, 200)
    print(f"Percentage: {percentage}%")
