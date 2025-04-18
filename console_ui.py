"""
Console UI module for the Hybrid Code Analyzer.
Provides UI components with optional Rich formatting support.
"""

# Check for optional dependencies
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ConsoleUI:
    """Simple console UI that works with or without Rich."""
    
    def __init__(self):
        """Initialize the UI with or without Rich."""
        self.rich_enabled = RICH_AVAILABLE
        if self.rich_enabled:
            self.console = Console()
        
    def print(self, text, style=None):
        """Print text with optional styling."""
        if self.rich_enabled and style:
            self.console.print(text, style=style)
        else:
            print(text)
    
    def print_title(self, title):
        """Print a title."""
        if self.rich_enabled:
            self.console.print(f"\n[bold blue]{title}[/bold blue]")
            self.console.print("=" * len(title))
        else:
            print(f"\n{title}")
            print("=" * len(title))
    
    def print_error(self, message):
        """Print an error message."""
        if self.rich_enabled:
            self.console.print(f"[bold red]ERROR:[/bold red] {message}")
        else:
            print(f"ERROR: {message}")
    
    def print_warning(self, message):
        """Print a warning message."""
        if self.rich_enabled:
            self.console.print(f"[bold yellow]WARNING:[/bold yellow] {message}")
        else:
            print(f"WARNING: {message}")
    
    def print_success(self, message):
        """Print a success message."""
        if self.rich_enabled:
            self.console.print(f"[bold green]SUCCESS:[/bold green] {message}")
        else:
            print(f"SUCCESS: {message}")
    
    def print_info(self, message):
        """Print an info message."""
        if self.rich_enabled:
            self.console.print(f"[bold cyan]INFO:[/bold cyan] {message}")
        else:
            print(f"INFO: {message}")
    
    def print_code(self, code, language="python"):
        """Print code with syntax highlighting."""
        if self.rich_enabled:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print(code)
    
    def create_table(self, columns):
        """Create a table with the given columns."""
        if self.rich_enabled:
            table = Table()
            for column in columns:
                table.add_column(column)
            return table
        else:
            print("\t".join(columns))
            return columns
    
    def add_row(self, table, *args):
        """Add a row to a table."""
        if self.rich_enabled:
            table.add_row(*args)
        else:
            print("\t".join(args))
    
    def print_table(self, table):
        """Print a table."""
        if self.rich_enabled:
            self.console.print(table)
    
    def progress(self, task_name, total):
        """Create a progress bar."""
        if self.rich_enabled:
            return Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            )
        else:
            class SimpleProgress:
                def __init__(self, task_name, total):
                    self.task_name = task_name
                    self.total = total
                    self.task_id = None
                
                def __enter__(self):
                    print(f"Starting {self.task_name}...")
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    print(f"Completed {self.task_name}")
                
                def add_task(self, description, total):
                    self.task_id = 1
                    return self.task_id
                
                def update(self, task_id, advance=1, description=None):
                    pass
            
            return SimpleProgress(task_name, total)
    
    def prompt(self, message, default=None):
        """Prompt the user for input."""
        if self.rich_enabled:
            return Prompt.ask(message, default=default)
        else:
            if default:
                result = input(f"{message} [{default}]: ")
                return result if result else default
            else:
                return input(f"{message}: ")
    
    def confirm(self, message, default=False):
        """Prompt the user for confirmation."""
        if self.rich_enabled:
            return Confirm.ask(message, default=default)
        else:
            response = input(f"{message} [{'Y/n' if default else 'y/N'}]: ")
            if not response:
                return default
            return response.lower() in ['y', 'yes']
    
    def select_option(self, message, options):
        """Let the user select from a list of options."""
        self.print(message)
        for i, option in enumerate(options, 1):
            self.print(f"  {i}. {option}")
        
        while True:
            choice = self.prompt("Enter your choice (number)")
            try:
                choice = int(choice)
                if 1 <= choice <= len(options):
                    return choice - 1
                else:
                    self.print_error(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                self.print_error("Please enter a valid number")