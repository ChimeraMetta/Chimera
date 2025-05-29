import logging
from colorama import init, Fore, Style

# Initialize colorama globally
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to different log levels"""

    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Get the plain levelname string (e.g., "INFO", "CRITICAL")
        plain_levelname = record.levelname 
        
        # Determine the color for the entire line based on the plain levelname
        line_color = self.COLORS.get(plain_levelname)

        # Temporarily use the plain levelname for super().format()
        # This ensures that super().format() uses the uncolored levelname
        # when creating the log message string.
        # We save and restore record.levelname around this call.
        
        original_record_levelname_value = record.levelname # Save current value
        record.levelname = plain_levelname # Ensure super().format sees the plain levelname

        # Generate the full log message string using the plain levelname
        log_message_content = super().format(record)
        
        # Restore record.levelname to its original value for this record object
        record.levelname = original_record_levelname_value

        # Apply the determined line color to the entire log message content
        if line_color:
            # Prepend the color and append RESET_ALL to color the whole line.
            # colorama's autoreset=True handles resets after individual color codes,
            # but explicitly adding RESET_ALL here ensures the entire line is bracketed.
            return f"{line_color}{log_message_content}{Style.RESET_ALL}"
        else:
            # No color defined for this level, return the plain message
            return log_message_content

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger instance configured with ColoredFormatter.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if logger already has them
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Use a more standard format string, let ColoredFormatter handle colors
        formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Re-export Fore and Style if direct usage is still needed in some places,
# though using the logger is preferred.
__all__ = ['get_logger', 'Fore', 'Style', 'ColoredFormatter'] 