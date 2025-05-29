import logging
from colorama import init, Fore, Style

# Initialize colorama globally
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to different log levels"""

    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Add color to the level name
        log_level_color = self.COLORS.get(record.levelname)
        if log_level_color:
            record.levelname = f"{log_level_color}{record.levelname}{Style.RESET_ALL}"
        
        # Color the entire message based on level for some cases if desired,
        # or just keep it to the level name.
        # For now, let's stick to coloring the level name primarily.
        # Example: if record.levelname == 'ERROR':
        #    return f"{Fore.RED}{super().format(record)}{Style.RESET_ALL}"
            
        return super().format(record)

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