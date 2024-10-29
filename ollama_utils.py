# ollama_utils.py
import subprocess
import pandas as pd

def get_ollama_model_list():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:", result.stderr)
        return []
    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        print("No models found.")
        return []
    header = lines[0].split()
    model_entries = [line.split(maxsplit=3) for line in lines[1:]]
    df = pd.DataFrame(model_entries, columns=header)
    return list(df['NAME'])