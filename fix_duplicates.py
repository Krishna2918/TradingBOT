"""
Fix duplicate callbacks in interactive_trading_dashboard.py
"""

import re

# Read the file
with open('interactive_trading_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all @app.callback decorators with their functions
callback_pattern = r'(@app\.callback\([^)]+\)[^)]*\)\s*def\s+\w+[^@]*?)(?=@app\.callback|if __name__|$)'
callbacks = re.findall(callback_pattern, content, re.DOTALL)

print(f"Found {len(callbacks)} callbacks")

# Track seen function names
seen_functions = {}
duplicates_to_remove = []

for i, callback in enumerate(callbacks):
    # Extract function name
    func_match = re.search(r'def\s+(\w+)', callback)
    if func_match:
        func_name = func_match.group(1)
        if func_name in seen_functions:
            print(f"Duplicate found: {func_name} (occurrence #{seen_functions[func_name] + 1})")
            duplicates_to_remove.append(callback)
            seen_functions[func_name] += 1
        else:
            seen_functions[func_name] = 1

print(f"\nFound {len(duplicates_to_remove)} duplicate callbacks to remove")

# Remove duplicates
for dup in duplicates_to_remove:
    content = content.replace(dup, '', 1)

# Write back
with open('interactive_trading_dashboard_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✓ Fixed file saved as: interactive_trading_dashboard_fixed.py")
print("Review it, then rename to replace the original")
