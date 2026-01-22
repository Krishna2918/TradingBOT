#!/usr/bin/env python3
"""
API Key Check Script

Checks for potential API key leaks and validates API key security.
"""

import os
import re
import sys
from typing import List, Dict, Any, Set

def find_potential_api_keys() -> List[Dict[str, Any]]:
    """Find potential API keys in the codebase."""
    potential_keys = []
    
    # Common API key patterns
    api_key_patterns = [
        r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r'secret["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r'password["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{8,})["\']?',
        r'apikey["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r'access[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r'private[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?'
    ]
    
    # Files to check
    files_to_check = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
        
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.json', '.yaml', '.yml', '.env', '.config')):
                file_path = os.path.join(root, file)
                files_to_check.append(file_path)
    
    # Check each file
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in api_key_patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            potential_key = match.group(1)
                            
                            # Skip if it's clearly a placeholder or example
                            if is_placeholder_key(potential_key):
                                continue
                            
                            potential_keys.append({
                                'file': file_path,
                                'line': line_num,
                                'content': line.strip(),
                                'key': potential_key,
                                'pattern': pattern
                            })
        
        except Exception as e:
            print(f"Warning: Could not check file {file_path}: {e}")
    
    return potential_keys

def is_placeholder_key(key: str) -> bool:
    """Check if a key is clearly a placeholder or example."""
    placeholder_patterns = [
        r'^your[_-]?api[_-]?key$',
        r'^example[_-]?key$',
        r'^placeholder$',
        r'^test[_-]?key$',
        r'^demo[_-]?key$',
        r'^sample[_-]?key$',
        r'^replace[_-]?me$',
        r'^changeme$',
        r'^xxx+$',
        r'^123+$',
        r'^abc+$'
    ]
    
    for pattern in placeholder_patterns:
        if re.match(pattern, key, re.IGNORECASE):
            return True
    
    return False

def check_environment_variables() -> List[str]:
    """Check for hardcoded environment variables that might contain secrets."""
    issues = []
    
    # Files to check for hardcoded env vars
    files_to_check = []
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
        
        for file in files:
            if file.endswith(('.py', '.js', '.ts')):
                file_path = os.path.join(root, file)
                files_to_check.append(file_path)
    
    # Patterns for hardcoded environment variables
    env_patterns = [
        r'os\.environ\[["\']([A-Z_]+)["\']\]\s*=\s*["\']([^"\']+)["\']',
        r'os\.getenv\(["\']([A-Z_]+)["\']\s*,\s*["\']([^"\']+)["\']\)',
        r'process\.env\.([A-Z_]+)\s*=\s*["\']([^"\']+)["\']'
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in env_patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            env_var = match.group(1)
                            value = match.group(2)
                            
                            # Check if it's a sensitive environment variable
                            if is_sensitive_env_var(env_var) and not is_placeholder_value(value):
                                issues.append(f"{file_path}:{line_num} - Hardcoded value for {env_var}")
        
        except Exception as e:
            print(f"Warning: Could not check file {file_path}: {e}")
    
    return issues

def is_sensitive_env_var(env_var: str) -> bool:
    """Check if an environment variable is sensitive."""
    sensitive_patterns = [
        r'.*API[_-]?KEY.*',
        r'.*SECRET.*',
        r'.*TOKEN.*',
        r'.*PASSWORD.*',
        r'.*PRIVATE[_-]?KEY.*',
        r'.*ACCESS[_-]?KEY.*',
        r'.*AUTH[_-]?TOKEN.*'
    ]
    
    for pattern in sensitive_patterns:
        if re.match(pattern, env_var, re.IGNORECASE):
            return True
    
    return False

def is_placeholder_value(value: str) -> bool:
    """Check if a value is clearly a placeholder."""
    placeholder_values = [
        'your_api_key',
        'your_secret',
        'your_token',
        'placeholder',
        'example',
        'test',
        'demo',
        'sample',
        'replace_me',
        'changeme',
        'xxx',
        '123',
        'abc'
    ]
    
    return value.lower() in placeholder_values

def check_gitignore() -> List[str]:
    """Check if sensitive files are properly ignored."""
    issues = []
    
    gitignore_files = ['.gitignore', '.git/info/exclude']
    sensitive_patterns = [
        r'\.env',
        r'\.env\.local',
        r'\.env\.production',
        r'\.env\.development',
        r'config\.json',
        r'secrets\.json',
        r'api_keys\.txt',
        r'credentials\.json',
        r'\.pem$',
        r'\.key$',
        r'\.p12$',
        r'\.pfx$'
    ]
    
    for gitignore_file in gitignore_files:
        if os.path.exists(gitignore_file):
            try:
                with open(gitignore_file, 'r') as f:
                    content = f.read()
                    
                    for pattern in sensitive_patterns:
                        if not re.search(pattern, content, re.IGNORECASE):
                            issues.append(f"Missing pattern in {gitignore_file}: {pattern}")
            
            except Exception as e:
                issues.append(f"Could not read {gitignore_file}: {e}")
    
    return issues

def main() -> int:
    """Main API key check function."""
    print("Checking for potential API key leaks...")
    
    issues = []
    
    # Check for potential API keys
    potential_keys = find_potential_api_keys()
    if potential_keys:
        issues.append(f"Found {len(potential_keys)} potential API keys:")
        for key_info in potential_keys:
            issues.append(f"  - {key_info['file']}:{key_info['line']} - {key_info['key'][:10]}...")
    
    # Check for hardcoded environment variables
    env_issues = check_environment_variables()
    if env_issues:
        issues.append(f"Found {len(env_issues)} hardcoded environment variables:")
        for issue in env_issues:
            issues.append(f"  - {issue}")
    
    # Check gitignore
    gitignore_issues = check_gitignore()
    if gitignore_issues:
        issues.append(f"Found {len(gitignore_issues)} gitignore issues:")
        for issue in gitignore_issues:
            issues.append(f"  - {issue}")
    
    # Report results
    if issues:
        print("❌ API key check failed!")
        print("Please review the following issues:")
        for issue in issues:
            print(issue)
        
        print("\nRecommendations:")
        print("1. Use environment variables for API keys")
        print("2. Add sensitive files to .gitignore")
        print("3. Use placeholder values in example code")
        print("4. Consider using a secrets management system")
        
        return 1
    else:
        print("✅ API key check passed!")
        return 0

if __name__ == "__main__":
    exit(main())
