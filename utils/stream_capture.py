"""
stream_capture.py
-----------------
Utility module for capturing and redirecting stdout to Streamlit.
Allows terminal output from agent nodes to appear in the Streamlit UI.
"""

import sys
import io
from contextlib import contextmanager
from typing import Optional, Callable, List


class StreamCapture:
    """
    Context manager to capture stdout and call a callback for each line.
    Perfect for redirecting print statements to Streamlit UI in real-time.
    """
    
    def __init__(self, callback: Callable[[str], None]):
        """
        Initialize stream capture.
        
        Args:
            callback: Function to call with each captured line
        """
        self.callback = callback
        self.original_stdout = None
        self._buffer = ""
    
    def write(self, text: str):
        """Capture and forward text."""
        if text:
            self._buffer += text
            
            # Send complete lines to callback
            if '\n' in self._buffer:
                lines = self._buffer.split('\n')
                # Keep last incomplete line in buffer
                self._buffer = lines[-1]
                
                # Send complete lines
                for line in lines[:-1]:
                    if line:  # Skip empty lines
                        self.callback(line)
    
    def flush(self):
        """Flush any remaining buffered content."""
        if self._buffer:
            self.callback(self._buffer)
            self._buffer = ""
    
    def __enter__(self):
        """Enter context: redirect stdout."""
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self
    
    def __exit__(self, *args):
        """Exit context: restore stdout."""
        self.flush()
        if self.original_stdout:
            sys.stdout = self.original_stdout


class CapturedOutput:
    """Store captured output lines."""
    
    def __init__(self):
        self.lines: List[str] = []
    
    def add_line(self, line: str):
        """Add a captured line."""
        self.lines.append(line)
    
    def get_text(self) -> str:
        """Get all captured output as single string."""
        return "\n".join(self.lines)
    
    def clear(self):
        """Clear captured output."""
        self.lines = []


@contextmanager
def capture_output(callback: Optional[Callable[[str], None]] = None):
    """
    Context manager to capture stdout.
    
    If no callback provided, returns a CapturedOutput object to collect lines.
    
    Usage with callback:
        def my_callback(line):
            print(f"Captured: {line}")
        
        with capture_output(callback=my_callback):
            print("Hello")  # Will be captured
            print("World")
    
    Usage without callback:
        with capture_output() as captured:
            print("Hello")
            print("World")
        
        output = captured.get_text()
    """
    
    if callback:
        # Use provided callback
        capture = StreamCapture(callback)
        with capture:
            yield
    else:
        # Return CapturedOutput object
        captured = CapturedOutput()
        capture = StreamCapture(captured.add_line)
        
        with capture:
            yield captured


def create_log_accumulator():
    """
    Create a function that accumulates log lines.
    Useful for Streamlit session state logging.
    
    Returns:
        Tuple of (add_line_function, get_all_lines_function)
    """
    lines = []
    
    def add_line(line: str):
        lines.append(line)
    
    def get_all():
        return "\n".join(lines)
    
    def clear():
        lines.clear()
    
    return add_line, get_all, clear


# Batch output capture for testing
class BatchOutputCapture:
    """Capture all output from a batch of operations."""
    
    def __init__(self):
        self.sections = {}  # {section_name: [lines]}
        self.current_section = None
    
    def start_section(self, name: str):
        """Start a new output section."""
        self.current_section = name
        self.sections[name] = []
    
    def add_line(self, line: str):
        """Add line to current section."""
        if self.current_section and self.current_section in self.sections:
            self.sections[self.current_section].append(line)
    
    def get_section(self, name: str) -> str:
        """Get output from a specific section."""
        return "\n".join(self.sections.get(name, []))
    
    def get_all(self) -> str:
        """Get all captured output."""
        result = []
        for name, lines in self.sections.items():
            result.append(f"--- {name} ---")
            result.extend(lines)
        return "\n".join(result)
