from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import math
import datetime
import re
import random
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
    
    # COMPLEXITY ADDED: New complex method for analyzing cart contents
    def analyze_cart_contents(self) -> Dict[str, Any]:
        """Analyze the cart contents and return detailed statistics.
        This function has been deliberately made complex with multiple operations and loops.
        """
        if not self.items:
            return {"empty": True, "stats": None}
        
        # Initialize counters and accumulators
        category_counts = {}
        category_totals = {}
        price_sum = 0
        price_squared_sum = 0
        min_price = float('inf')
        max_price = float('-inf')
        most_expensive_item = None
        cheapest_item = None
        discounted_items = []
        
        # First loop through items to gather basic stats
        for item in self.items:
            # Category processing
            if item.category in category_counts:
                category_counts[item.category] += 1
                category_totals[item.category] += item.price
            else:
                category_counts[item.category] = 1
                category_totals[item.category] = item.price
            
            # Price statistics
            price = item.price
            price_sum += price
            price_squared_sum += price * price
            
            # Track min/max
            if price < min_price:
                min_price = price
                cheapest_item = item
            
            if price > max_price:
                max_price = price
                most_expensive_item = item
            
            # Check for discounted items
            if isinstance(item, DiscountedProduct):
                discounted_items.append(item)
        
        # Calculate averages per category
        category_averages = {}
        for category in category_counts:
            if category_counts[category] > 0:  # Avoid division by zero
                category_averages[category] = category_totals[category] / category_counts[category]
            else:
                category_averages[category] = 0
        
        # Calculate standard deviation of prices
        count = len(self.items)
        mean_price = price_sum / count if count > 0 else 0
        variance = (price_squared_sum / count) - (mean_price * mean_price) if count > 0 else 0
        std_dev = math.sqrt(variance) if variance > 0 else 0
        
        # Calculate discount statistics
        total_discount = 0
        for item in discounted_items:
            original_price = item.price / (1 - item.discount/100)
            saved_amount = original_price - item.price
            total_discount += saved_amount
        
        # Find category with the highest total
        highest_total_category = None
        highest_total = 0
        for category, total in category_totals.items():
            if total > highest_total:
                highest_total = total
                highest_total_category = category
        
        # Generate suggestions based on cart analysis
        suggestions = []
        if count > 5:
            suggestions.append("Consider our bulk discount program")
        
        if "Electronics" in category_counts and category_counts["Electronics"] > 1:
            suggestions.append("Check out our extended warranty options")
        
        if max_price - min_price > 500:
            suggestions.append("Your cart has a wide price range - consider our financing options")
        
        if len(discounted_items) / count < 0.3 and count > 3:
            suggestions.append("You may qualify for additional discounts")
        
        # Calculate price distribution
        price_ranges = {
            "Under $50": 0,
            "$50-$100": 0,
            "$100-$500": 0,
            "$500-$1000": 0,
            "Over $1000": 0
        }
        
        for item in self.items:
            price = item.price
            if price < 50:
                price_ranges["Under $50"] += 1
            elif price < 100:
                price_ranges["$50-$100"] += 1
            elif price < 500:
                price_ranges["$100-$500"] += 1
            elif price < 1000:
                price_ranges["$500-$1000"] += 1
            else:
                price_ranges["Over $1000"] += 1
        
        # Return comprehensive results
        return {
            "empty": False,
            "item_count": count,
            "total_price": price_sum,
            "average_price": mean_price,
            "price_std_dev": std_dev,
            "min_price": min_price,
            "max_price": max_price,
            "cheapest_item": cheapest_item.name if cheapest_item else None,
            "most_expensive_item": most_expensive_item.name if most_expensive_item else None,
            "category_counts": category_counts,
            "category_totals": category_totals,
            "category_averages": category_averages,
            "highest_spend_category": highest_total_category,
            "discounted_items_count": len(discounted_items),
            "total_discount_savings": total_discount,
            "price_distribution": price_ranges,
            "suggestions": suggestions
        }

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
    
    # COMPLEXITY ADDED: New complex string processing method
    @staticmethod
    def analyze_text(text: str) -> Dict[str, Any]:
        """Perform advanced text analysis with multiple operations and loops.
        This function has been deliberately made complex.
        """
        if not text:
            return {"empty": True}
        
        # Word frequency analysis
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Find most common words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:5] if len(sorted_words) >= 5 else sorted_words
        
        # Character frequency
        char_freq = {}
        for char in text:
            if char.isalnum():
                if char in char_freq:
                    char_freq[char] += 1
                else:
                    char_freq[char] = 1
        
        # Calculate various statistics
        word_count = len(words)
        char_count = len(text)
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        space_count = sum(1 for c in text if c.isspace())
        punctuation_count = char_count - alpha_count - digit_count - space_count
        
        # Calculate word length distribution
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Word length distribution
        length_dist = {}
        for length in word_lengths:
            if length in length_dist:
                length_dist[length] += 1
            else:
                length_dist[length] = 1
        
        # Calculate sentence statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        if sentence_count > 0:
            sentence_lengths = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences]
            avg_sentence_length = sum(sentence_lengths) / sentence_count
            max_sentence_length = max(sentence_lengths) if sentence_lengths else 0
            min_sentence_length = min(sentence_lengths) if sentence_lengths else 0
        else:
            avg_sentence_length = 0
            max_sentence_length = 0
            min_sentence_length = 0
        
        # Analyze sentiment (very simplified)
        positive_words = {'good', 'great', 'excellent', 'best', 'better', 'nice', 'wonderful', 'amazing'}
        negative_words = {'bad', 'worse', 'worst', 'terrible', 'awful', 'poor', 'horrible'}
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        sentiment_score = (positive_count - negative_count) / word_count if word_count > 0 else 0
        
        # Calculate readability (simplified Flesch Reading Ease)
        if word_count > 0 and sentence_count > 0:
            words_per_sentence = word_count / sentence_count
            syllables = sum(StringUtils._count_syllables(word) for word in words)
            syllables_per_word = syllables / word_count
            readability = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        else:
            readability = 0
        
        # Return comprehensive analysis
        return {
            "empty": False,
            "word_count": word_count,
            "character_count": char_count,
            "alphabetic_count": alpha_count,
            "digit_count": digit_count,
            "space_count": space_count,
            "punctuation_count": punctuation_count,
            "sentence_count": sentence_count,
            "average_word_length": avg_word_length,
            "average_sentence_length": avg_sentence_length,
            "max_sentence_length": max_sentence_length,
            "min_sentence_length": min_sentence_length,
            "word_frequency": dict(sorted_words),
            "top_words": dict(top_words),
            "character_frequency": char_freq,
            "word_length_distribution": length_dist,
            "sentiment_score": sentiment_score,
            "readability_score": readability,
            "sentiment": "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral"),
            "reading_level": StringUtils._get_reading_level(readability),
        }
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word (simplified algorithm)."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        # Remove trailing e
        if word.endswith('e'):
            word = word[:-1]
        
        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        in_vowel_group = False
        
        for char in word:
            if char in vowels:
                if not in_vowel_group:
                    count += 1
                    in_vowel_group = True
            else:
                in_vowel_group = False
        
        return max(1, count)
    
    @staticmethod
    def _get_reading_level(score: float) -> str:
        """Convert Flesch Reading Ease score to reading level."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"

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

# COMPLEXITY ADDED: Enhance the discount strategy with more complex logic
def enhanced_bulk_discount_strategy(product: Product) -> float:
    """A complex discount strategy with multiple conditions and calculations."""
    base_price = product.price
    discount_factor = 1.0  # Start with no discount
    
    # Apply category-based discounts
    if product.category == "Electronics":
        discount_factor *= 0.9  # 10% off electronics
    elif product.category == "Clothing":
        discount_factor *= 0.85  # 15% off clothing
    elif product.category == "Books":
        discount_factor *= 0.8  # 20% off books
    elif product.category == "Groceries":
        discount_factor *= 0.95  # 5% off groceries
    else:
        discount_factor *= 0.97  # 3% off everything else
    
    # Apply price range based discounts
    if base_price >= 1000:
        discount_factor *= 0.9  # Additional 10% off
    elif base_price >= 500:
        discount_factor *= 0.93  # Additional 7% off
    elif base_price >= 100:
        discount_factor *= 0.95  # Additional 5% off
    elif base_price >= 50:
        discount_factor *= 0.97  # Additional 3% off
    
    # Adjust based on name length (just to add complexity)
    name_length = len(product.name)
    if name_length > 15:
        discount_factor *= 0.99  # 1% extra off for long names
    
    # Apply time-based discounts (seasonal/daily)
    current_hour = datetime.datetime.now().hour
    current_month = datetime.datetime.now().month
    
    # Evening discount (after 6pm)
    if current_hour >= 18 or current_hour < 6:
        discount_factor *= 0.97  # 3% evening discount
    
    # Winter discount (December to February)
    if current_month in [12, 1, 2]:
        discount_factor *= 0.93  # 7% winter discount
    # Summer discount (June to August)
    elif current_month in [6, 7, 8]:
        discount_factor *= 0.95  # 5% summer discount
    
    # Special random discount (once in a while)
    if random.random() < 0.1:  # 10% chance
        discount_factor *= 0.9  # Random 10% extra discount
    
    # Ensure minimum profit margin
    min_price = base_price * 0.7  # Don't discount below 70% of original
    discounted_price = base_price * discount_factor
    
    return max(discounted_price, min_price)

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

def find_max_in_range(numbers, start_idx, end_idx):
    """Find the maximum value in a list within a specific range."""
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
    
    max_val = numbers[start_idx]
    for i in range(start_idx + 1, end_idx):
        if numbers[i] > max_val:
            max_val = numbers[i]
    
    return max_val

def clean_and_normalize_text(text):
    """Clean and normalize text input."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace and convert to lowercase
    cleaned = text.strip().lower()

    # Replace multiple spaces with single space
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned

def calculate_moving_average(numbers, window_size):
    """Calculate moving average with specified window size."""
    if not numbers or window_size <= 0 or window_size > len(numbers):
        return []
    
    averages = []
    for i in range(len(numbers) - window_size + 1):
        window_sum = sum(numbers[i:i + window_size])
        averages.append(window_sum / window_size)
    
    return averages

# COMPLEXITY ADDED: Enhance the receipt generation with more complex logic
def generate_receipt(order: Dict[str, Any]) -> str:
    """Generate a detailed receipt with complex formatting and calculations.
    This function has been deliberately made more complex.
    """
    receipt = "=" * 40 + "\n"
    receipt += "             RECEIPT                \n"
    receipt += "=" * 40 + "\n\n"
    
    # Order metadata with formatted strings
    receipt += f"Order: {order['order_id']}\n"
    receipt += f"Date: {order['date']}\n"
    
    # Format customer info with detailed processing
    customer = order['customer']
    receipt += "Customer Information:\n"
    receipt += f"  Name: {customer['name']}\n"
    
    if 'email' in customer:
        email = customer['email']
        # Add asterisks to partially mask email
        parts = email.split('@')
        if len(parts) == 2:
            username, domain = parts
            if len(username) > 3:
                masked_username = username[:2] + '*' * (len(username) - 3) + username[-1]
                masked_email = f"{masked_username}@{domain}"
                receipt += f"  Email: {masked_email}\n"
            else:
                receipt += f"  Email: {email}\n"
        else:
            receipt += f"  Email: {email}\n"
    
    if 'phone' in customer:
        phone = customer['phone']
        # Format phone number with dashes
        if len(phone) == 10:
            formatted_phone = f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
            receipt += f"  Phone: {formatted_phone}\n"
        else:
            receipt += f"  Phone: {phone}\n"
    
    if 'address' in customer:
        address = customer['address']
        # Format address with newlines and indentation
        address_lines = address.split(', ')
        receipt += "  Address:\n"
        for line in address_lines:
            receipt += f"    {line}\n"
    
    receipt += "\n" + "-" * 40 + "\n"
    
    # Group items by category for better organization
    items_by_category = {}
    for item in order['items']:
        category = item.category
        if category not in items_by_category:
            items_by_category[category] = []
        items_by_category[category].append(item)
    
    # Print items organized by category
    receipt += "Items:\n"
    
    item_count = 0
    for category, items in items_by_category.items():
        receipt += f"  {category}:\n"
        category_total = 0
        
        for item in items:
            item_count += 1
            price = item.price
            category_total += price
            
            # Format item line with padding for alignment
            item_name = item.name
            if len(item_name) > 20:
                item_name = item_name[:17] + "..."
            
            # Add discount information if applicable
            if isinstance(item, DiscountedProduct):
                original_price = price / (1 - item.discount/100)
                discount_info = f" (Save: {format_currency(original_price - price)})"
                receipt += f"    {item_count:2d}. {item_name.ljust(20)} {format_currency(price).rjust(10)}{discount_info}\n"
            else:
                receipt += f"    {item_count:2d}. {item_name.ljust(20)} {format_currency(price).rjust(10)}\n"
        
        # Add category subtotal
        receipt += f"       {category} Subtotal: {format_currency(category_total).rjust(10)}\n"
        receipt += "\n"
    
    receipt += "-" * 40 + "\n"
    
    # Extract numerical values for calculations
    subtotal_str = order['subtotal']
    subtotal_value = float(subtotal_str.replace('$', ''))
    
    discount_str = order['discount']
    discount_value = float(discount_str.replace('$', ''))
    
    tax_str = order['tax']
    tax_value = float(tax_str.replace('$', ''))
    
    total_str = order['total']
    total_value = float(total_str.replace('$', ''))
    
    # Add summary calculation section
    receipt += f"Subtotal: {order['subtotal'].rjust(26)}\n"
    
    if discount_value > 0:
        discount_percentage = (discount_value / subtotal_value) * 100
        receipt += f"Discount ({discount_percentage:.1f}%): {order['discount'].rjust(20)}\n"
    else:
        receipt += f"Discount: {order['discount'].rjust(26)}\n"
    
    tax_percentage = (tax_value / (subtotal_value - discount_value)) * 100
    receipt += f"Tax ({tax_percentage:.2f}%): {order['tax'].rjust(25)}\n"
    receipt += f"Total: {order['total'].rjust(29)}\n"
    
    # Calculate suggested tip
    receipt += "\nSuggested Tip:\n"
    for percentage in [15, 18, 20, 25]:
        tip_amount = total_value * (percentage / 100)
        total_with_tip = total_value + tip_amount
        receipt += f"  {percentage}%: {format_currency(tip_amount).rjust(10)} (Total: {format_currency(total_with_tip)})\n"
    
    # Add satisfaction survey
    receipt += "\n" + "-" * 40 + "\n"
    receipt += "Please rate your shopping experience (1-5 stars):\n"
    receipt += "* * * * *\n\n"
    
    # Add promotional footer
    receipt += "Thank you for shopping with us!\n"
    receipt += f"Save this receipt for returns within 30 days.\n"
    receipt += "Visit our website for special promotions.\n"
    receipt += "=" * 40 + "\n"
    
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
    customer = {"name": "John Doe", "email": "john@example.com", "phone": "1234567890", 
                "address": "123 Main St, Apt 4B, New York, NY 10001"}
    order = process_order(cart, customer, apply_discount=True)
    
    # Generate and print receipt
    receipt = generate_receipt(order)
    print(receipt)
    
    # Analyze cart contents
    cart_analysis = cart.analyze_cart_contents()
    print("\nCart Analysis Summary:")
    print(f"Total Items: {cart_analysis['item_count']}")
    print(f"Total Value: {format_currency(cart_analysis['total_price'])}")
    print(f"Categories: {', '.join(cart_analysis['category_counts'].keys())}")
    print(f"Suggestions:")
    for suggestion in cart_analysis.get("suggestions", []):
        print(f"  - {suggestion}")
    
    # Test string analysis
    sample_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter of the 
    English alphabet at least once. The five boxing wizards jump quickly. A wonderful serenity 
    has taken possession of my entire soul, like these sweet mornings of spring which I enjoy 
    with my whole heart.
    """
    text_analysis = StringUtils.analyze_text(sample_text)
    print("\nText Analysis Summary:")
    print(f"Word Count: {text_analysis['word_count']}")
    print(f"Sentence Count: {text_analysis['sentence_count']}")
    print(f"Average Word Length: {text_analysis['average_word_length']:.2f} characters")
    print(f"Sentiment: {text_analysis['sentiment']}")
    print(f"Reading Level: {text_analysis['reading_level']}")
    print("Top 5 Words:")
    for word, count in list(text_analysis['top_words'].items())[:5]:
        print(f"  - {word}: {count}")
    
    # Demonstrate some other function calls
    days = days_between_dates("2023-01-01", "2023-12-31")
    print(f"\nDays in 2023: {days}")
    
    # Apply different discount strategy
    discounted_products = apply_discount_strategy(cart.items, enhanced_bulk_discount_strategy)
    print("\nDiscounted Products:")
    for product, price in discounted_products:
        print(f"{product.name}: Original: {format_currency(product.price)}, Discounted: {format_currency(price)}")
    
    # Use utility classes
    formatted_name = StringUtils.capitalize_words("john doe smith")
    print(f"\nFormatted name: {formatted_name}")
    
    percentage = MathUtils.calculate_percentage(75, 200)
    print(f"Percentage: {percentage}%")