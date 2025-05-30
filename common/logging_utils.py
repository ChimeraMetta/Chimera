import logging
from colorama import init, Fore, Style
import argparse
from inquirer import themes

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

# Custom theme for inquirer that matches our color scheme
class ChimeraTheme(themes.GreenPassion):
    def __init__(self):
        """
        This is a custom theme for inquirer that matches our color scheme.
        It's used to style the inquirer prompts.
        It's based on the GreenPassion theme from inquirer.
        It's used to style the inquirer prompts.
        """
        super().__init__()
        # Assuming Fore and Style are available (e.g., imported from logging_utils or globally)
        self.Checkbox.selected_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        self.Checkbox.unselected_icon = " "
        self.Checkbox.selected_color = Fore.GREEN # Inquirer might handle RESET_ALL
        self.Checkbox.unselected_color = Style.RESET_ALL # Or rely on autoreset
        # For List prompt, cursor color can be set if supported by theme
        if hasattr(self.List, 'selection_cursor'):
            self.List.selection_cursor = f"{Fore.GREEN}❯{Style.RESET_ALL}"
        if hasattr(self.List, 'selection_color'):
            self.List.selection_color = Fore.GREEN

# Custom HelpFormatter for colored output (moved from cli.py)
class ColoredHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        """
        This is a custom help formatter for argparse that adds colors to the output.
        It's used to style the help text.
        It's based on the HelpFormatter from argparse.
        It's used to style the help text.
        """
        super().__init__(prog)

    def _format_action_invocation(self, action):
        """
        This is a custom help formatter for argparse that adds colors to the output.

        Args:
            action: The action to format.

        Returns:
            The formatted action invocation.
        """
        if not action.option_strings:
            # For positional arguments
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return f"{Fore.CYAN}{metavar}{Style.RESET_ALL}"
        else:
            # For optional arguments
            parts = []
            if action.nargs == 0:
                parts.extend([f"{Fore.GREEN}{opt}{Style.RESET_ALL}" for opt in action.option_strings])
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for opt in action.option_strings:
                    parts.append(f"{Fore.GREEN}{opt}{Style.RESET_ALL} {Fore.CYAN}{args_string}{Style.RESET_ALL}")
            return ", ".join(parts)

    def _format_usage(self, usage, actions, groups, prefix):
        """
        This is a custom help formatter for argparse that adds colors to the output

        Args:
            usage: The usage string to format.
            actions: The actions to format.
            groups: The groups to format.
            prefix: The prefix to format.

        Returns:
            The formatted usage string.
        """
        if prefix is None:
            prefix = f'{Fore.YELLOW}Usage: {Style.RESET_ALL}'
        
        # Let the superclass handle the initial formatting (including generating usage if it's None)
        formatted_usage = super()._format_usage(usage, actions, groups, prefix)
        
        # Now, ensure the program name is colored if it's part of the usage string
        if self._prog and formatted_usage:
            formatted_usage = formatted_usage.replace(self._prog, f"{Fore.GREEN}{self._prog}{Style.RESET_ALL}")
            
        return formatted_usage

    def start_section(self, heading):
        super().start_section(f"{Fore.YELLOW}{heading.capitalize()}{Style.RESET_ALL}")

    # Color choices for subparsers (commands)
    def _format_action(self, action):
        parts = super()._format_action(action)
        if action.nargs == argparse.ZERO_OR_MORE or action.nargs == argparse.ONE_OR_MORE:
            # This is likely for the 'command' positional argument with choices
            if action.choices is not None:
                 # Color the choices list in the help text
                 # The original parts string looks like: "usage: PROG command {cmd1,cmd2} ...\n..."
                 # or for the choices section: "  {cmd1,cmd2,cmd3,cmd4} The command to execute..."
                
                # Color choices in the main help line for the command argument
                # Example: "{summary,analyze,import,export}"
                colored_choices_inner = f"{Style.RESET_ALL},{Fore.GREEN}{Style.BRIGHT}".join(action.choices)
                colored_choices = f"{{{Fore.GREEN}{Style.BRIGHT}{colored_choices_inner}{Style.RESET_ALL}}}"
                
                # Attempt to replace the uncolored choices in the generated parts string
                # This is a bit fragile as it depends on argparse's internal formatting
                import re
                # Pattern to find "{choice1,choice2,...}"
                choices_pattern = r"\{" + r",".join(re.escape(c) for c in action.choices) + r"\}"
                
                # Color the {command1, command2} part in the "positional arguments" section
                parts = re.sub(choices_pattern, colored_choices, parts)

        return parts

    def format_help(self):
        #ascii_art = """
        # ___           ___                       ___           ___           ___           ___     
        # /\  \         /\__\          ___        /\__\         /\  \         /\  \         /\  \    
        #/::\  \       /:/  /         /\  \      /::|  |       /::\  \       /::\  \       /::\  \   
        #/:/\:\  \     /:/__/          \:\  \    /:|:|  |      /:/\:\  \     /:/\:\  \     /:/\:\  \  
        #/:/  \:\  \   /::\  \ ___      /::\__\  /:/|:|__|__   /::\~\:\  \   /::\~\:\  \   /::\~\:\  \ 
        #/:/__/ \:\__\ /:/\:\  /\__\  __/:/\/__/ /:/ |::::\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\
        #\:\  \  \/__/ \/__\:\/:/  / /\/:/  /    \/__/~~/:/  / \:\~\:\ \/__/ \/_|::\/:/  / \/__\:\/:/  /
        # \:\  \            \::/  /  \::/__/           /:/  /   \:\ \:\__\      |:|::/  /       \::/  / 
        #  \:\  \           /:/  /    \:\__\          /:/  /     \:\ \/__/      |:|\/__/        /:/  /  
        #   \:\__\         /:/  /      \/__/         /:/  /       \:\__\        |:|  |         /:/  /   
        #    \/__/         \/__/                     \/__/         \/__/         \|__|         \/__/    
        #
        #"""
        # Note: The ASCII art contains backslashes, which need to be escaped in a Python string literal.
        # Or, use a raw string if the art doesn't contain characters that would be misinterpreted by raw strings (like trailing backslash).
        # For simplicity in this example, direct embedding is tricky with complex escape sequences.
        # It's often better to load from a file or define carefully with escaped characters.
        # The user has the art in cli.py, so we will copy it exactly, escaping as necessary.
        
        ascii_art = (
            "     ___           ___                       ___           ___           ___           ___     \n"
            "     /\  \         /\__\          ___        /\__\         /\  \         /\  \         /\  \    \n"
            "    /::\  \       /:/  /         /\  \      /::|  |       /::\  \       /::\  \       /::\  \   \n"
            "   /:/\:\  \     /:/__/          \:\  \    /:|:|  |      /:/\:\  \     /:/\:\  \     /:/\:\  \  \n"
            "  /:/  \:\  \   /::\  \ ___      /::\__\  /:/|:|__|__   /::\~\:\  \   /::\~\:\  \   /::\~\:\  \ \n"
            " /:/__/ \:\__\ /:/\:\  /\__\  __/:/\/__/ /:/ |::::\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ \n"
            "  \:\  \  \/__/ \/__\:\/:/  / /\/:/  /    \/__/~~/:/  / \:\~\:\ \/__/ \/_|::\/:/  / \/__\:\/:/  /\n"
            "  \:\  \            \::/  /  \::/__/           /:/  /   \:\ \:\__\      |:|::/  /       \::/  / \n"
            "   \:\  \           /:/  /    \:\__\          /:/  /     \:\ \/__/      |:|\/__/        /:/  /  \n"
            "    \:\__\         /:/  /      \/__/         /:/  /       \:\__\        |:|  |         /:/  /   \n"
            "     \/__/         \/__/                     \/__/         \/__/         \|__|         \/__/    \n"
            "\n"
        )
        help_message = super().format_help()
        return f"{Fore.CYAN}{ascii_art}{Style.RESET_ALL}\n{help_message}" 