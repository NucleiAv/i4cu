#!/usr/bin/env python3
"""
Wrapper script to run deepfake detector with NNPACK warnings suppressed.
Redirects stderr to filter out NNPACK warnings at the OS level.
Shows output in real-time for better progress visibility.
"""

import sys
import subprocess
import os
import threading
import queue

# Filter function for stderr
def filter_stderr(line):
    """Filter out NNPACK warnings."""
    if 'NNPACK' in line or 'Could not initialize NNPACK' in line:
        return False
    return True

def read_output(pipe, output_queue, filter_func=None):
    """Read from pipe and put lines in queue."""
    try:
        for line in iter(pipe.readline, ''):
            if filter_func:
                if filter_func(line):
                    output_queue.put(('line', line))
            else:
                output_queue.put(('line', line))
        output_queue.put(('done', None))
    except:
        output_queue.put(('done', None))

if __name__ == '__main__':
    # Get the original command
    cmd = [sys.executable, 'cli.py'] + sys.argv[1:]
    
    # Run with filtered stderr and real-time stdout
    process = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Create queues for output
    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()
    
    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue, filter_stderr))
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Process output in real-time
    stdout_done = False
    stderr_done = False
    
    while not (stdout_done and stderr_done):
        # Check stdout
        try:
            msg_type, line = stdout_queue.get(timeout=0.1)
            if msg_type == 'done':
                stdout_done = True
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
        except queue.Empty:
            pass
        
        # Check stderr
        try:
            msg_type, line = stderr_queue.get(timeout=0.1)
            if msg_type == 'done':
                stderr_done = True
            else:
                sys.stderr.write(line)
                sys.stderr.flush()
        except queue.Empty:
            pass
    
    # Wait for process to complete
    returncode = process.wait()
    
    # Wait for threads to finish
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    
    sys.exit(returncode)

