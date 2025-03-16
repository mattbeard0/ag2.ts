#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel
import time
from dotenv import load_dotenv; load_dotenv()
# Hardcoded values
API_KEY = os.getenv("API_KEY")
INPUT_PATH = os.getenv("INPUT_PATH")
ALL_OVERRIDE = os.getenv("ALL_OVERRIDE", "ask")

def extract_initial_comments(content: str) -> tuple[str, str]:
    """
    Extract comments at the beginning of a Python file and convert them to TypeScript format.
    
    Returns:
        tuple: (ts_comments, remaining_content)
    """
    lines = content.splitlines()
    comment_lines = []
    non_comment_index = 0
    
    # Find consecutive comment lines at the beginning of the file
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped == '':
            if stripped.startswith('#'):
                # Convert Python # comments to TypeScript // comments
                comment_lines.append('//' + stripped[1:])
            else:
                # Keep empty lines
                comment_lines.append('')
            non_comment_index = i + 1
        else:
            # Stop when we hit the first non-comment, non-empty line
            break
    
    # Join the comments back together and get the remaining content
    ts_comments = '\n'.join(comment_lines)
    remaining_content = '\n'.join(lines[non_comment_index:])
    
    return ts_comments, remaining_content

def process_imports(content: str) -> Tuple[List[str], str]:
    """
    Extract Python import statements and convert them to TypeScript format.
    Handles both single-line and multi-line imports.
    
    Returns:
        Tuple[List[str], str]: (typescript_imports, remaining_content)
    """
    # Regular expressions to match different types of Python imports
    import_patterns = [
        # from X import Y, Z as Z1, ...
        r'from\s+([\w.]+)\s+import\s+([^#\n]+)(?:\s*#.*)?',
        # import X, Y, Z as Z1, ...
        r'import\s+([^#\n]+)(?:\s*#.*)?',
    ]
    
    # Extract all import lines, handling multi-line imports with parentheses
    lines = content.splitlines()
    processed_lines = []
    typescript_imports = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if line starts an import statement
        if line.strip().startswith(('import ', 'from ')):
            # Collect the complete import statement (may span multiple lines)
            full_import = line
            
            # Handle multi-line imports with parentheses
            open_parentheses = line.count('(') - line.count(')')
            
            # Keep adding lines until all parentheses are closed or line ends with comma
            while (open_parentheses > 0 or full_import.rstrip().endswith(',')) and i + 1 < len(lines):
                i += 1
                next_line = lines[i]
                full_import += '\n' + next_line
                open_parentheses += next_line.count('(') - next_line.count(')')
            
            # Skip imports from typing module or import of export_module
            if (re.search(r'from\s+typing\b', full_import) or 
                re.search(r'import\s+typing\b', full_import) or 
                'export_module' in full_import):
                i += 1
                continue
                
            # Process the complete import statement
            for pattern in import_patterns:
                if re.match(pattern, full_import):
                    ts_import = convert_import_to_ts(full_import)
                    if ts_import:
                        typescript_imports.append(ts_import)
                    break
        else:
            processed_lines.append(line)
        
        i += 1
    
    remaining_content = '\n'.join(processed_lines)
    return typescript_imports, remaining_content

def convert_import_to_ts(import_statement: str) -> str:
    """
    Convert a Python import statement to TypeScript import format.
    
    Args:
        import_statement: A complete Python import statement (may span multiple lines)
        
    Returns:
        str: TypeScript import statement or empty string if conversion not needed
    """
    # Remove comments and normalize whitespace
    cleaned_import = re.sub(r'#.*$', '', import_statement, flags=re.MULTILINE)
    cleaned_import = re.sub(r'\s+', ' ', cleaned_import.replace('\n', ' ')).strip()
    
    # Skip imports from typing module or imports of export_module
    if (re.search(r'from\s+typing\b', cleaned_import) or 
        re.search(r'import\s+typing\b', cleaned_import) or 
        'export_module' in cleaned_import):
        return ""
    
    # Handle 'from X import Y' format
    from_import_match = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', cleaned_import)
    if from_import_match:
        module = from_import_match.group(1)
        imported_items = from_import_match.group(2)
        
        # Handle parentheses in imported items
        imported_items = imported_items.replace('(', '').replace(')', '')
        
        # Parse the imported items
        items = []
        for item in re.split(r',\s*', imported_items):
            item = item.strip()
            if not item:
                continue
                
            # Handle 'X as Y' format
            as_match = re.match(r'(\w+)\s+as\s+(\w+)', item)
            if as_match:
                original = as_match.group(1)
                alias = as_match.group(2)
                items.append(f"{original} as {alias}")
            else:
                items.append(item)
        
        if items:
            # Convert to TypeScript format
            items_str = ', '.join(items)
            # For relative imports, we need to add .js extension for ES modules
            if module.startswith('.'):
                module = module# + '.js'
            return f'import {{ {items_str} }} from "{module}";'
    
    # Handle 'import X' or 'import X, Y, Z' format
    import_match = re.match(r'import\s+(.+)', cleaned_import)
    if import_match:
        imported_modules = import_match.group(1)
        
        # Parse imported modules
        modules = []
        for module in re.split(r',\s*', imported_modules):
            module = module.strip()
            if not module:
                continue
                
            # Handle 'X as Y' format
            as_match = re.match(r'(\w+)\s+as\s+(\w+)', module)
            if as_match:
                original = as_match.group(1)
                alias = as_match.group(2)
                # For direct imports, we handle it differently in TypeScript
                modules.append(f'* as {alias} from "{original}"')
            else:
                # Standard import
                modules.append(f'* as {module} from "{module}"')
        
        if modules:
            # Generate multiple import statements if needed
            return '\n'.join([f'import {module};' for module in modules])
    
    # Return empty string for imports we couldn't convert properly
    return ""

def split_python_file(content: str) -> List[str]:
    """
    Split Python file content into chunks based on class and function definitions.
    Each chunk starts with a class or function definition.
    
    Rules:
    1. First split by classes
    2. If a class is too large (>300 lines), split it by its methods
    3. Never split individual functions/methods
    4. Remove @export_module, __all__, and @runtime_checkable lines
    5. Keep global content before the first class/function as a separate chunk
    """
    # Remove @export_module and __all__ = lines
    content_lines = []
    for line in content.splitlines():
        if '@export_module' not in line and '__all__ =' not in line and '@runtime_checkable' not in line:
            content_lines.append(line)
    
    content = '\n'.join(content_lines)
    chunks = []
    
    # Find all class and function definitions
    class_pattern = r"^class\s+\w+"
    function_pattern = r"^(async\s+)?def\s+\w+"
    
    class_matches = list(re.finditer(class_pattern, content, re.MULTILINE))
    function_matches = list(re.finditer(function_pattern, content, re.MULTILINE))
    
    # Combine and sort all matches by position
    all_matches = []
    for match in class_matches:
        all_matches.append(('class', match.start(), match))
    
    for match in function_matches:
        all_matches.append(('function', match.start(), match))
    
    all_matches.sort(key=lambda x: x[1])
    
    # If no matches, return the whole content as one chunk
    if not all_matches:
        return [content]
    
    # Handle content before first match if any
    if all_matches[0][1] > 0:
        chunks.append(content[:all_matches[0][1]])
    
    # Process matches to create appropriate chunks
    i = 0
    while i < len(all_matches):
        match_type, start_pos, match = all_matches[i]
        
        # Find the end of this match (start of next match or end of file)
        if i < len(all_matches) - 1:
            end_pos = all_matches[i + 1][1]
        else:
            end_pos = len(content)
        
        chunk = content[start_pos:end_pos]
        chunk_line_count = chunk.count('\n') + 1
        
        if match_type == 'class' and chunk_line_count > 300:
            # Large class - need to split it by methods
            class_content = chunk
            class_lines = class_content.splitlines()
            
            # Extract class definition line
            class_def_line = class_lines[0]
            class_body_lines = class_lines[1:]
            class_body = '\n'.join(class_body_lines)
            
            # Find all method definitions within the class
            # Adjust pattern to match methods with proper indentation
            method_pattern = r"^\s+(async\s+)?def\s+\w+"
            method_matches = list(re.finditer(method_pattern, class_body, re.MULTILINE))
            
            if not method_matches:
                # No methods to split, use the whole class
                chunks.append(class_content)
            else:
                # Handle class definition and any content before first method
                class_chunk = [class_def_line]
                
                if method_matches[0].start() > 0:
                    # Add content between class definition and first method
                    first_method_start = method_matches[0].start()
                    class_chunk.append(class_body[:first_method_start])
                
                class_start_chunk = '\n'.join(class_chunk)
                chunks.append(class_start_chunk)
                
                # Process each method
                current_chunk_lines = []
                current_chunk_size = 0
                
                for j, method_match in enumerate(method_matches):
                    method_start = method_match.start()
                    
                    # Determine end of method
                    if j < len(method_matches) - 1:
                        method_end = method_matches[j + 1].start()
                    else:
                        method_end = len(class_body)
                    
                    method_content = class_body[method_start:method_end]
                    method_line_count = method_content.count('\n') + 1
                    
                    # Always keep a method together
                    if current_chunk_size + method_line_count > 300 and current_chunk_size > 0:
                        # Complete the current chunk with proper class context
                        chunks.append('\n'.join(current_chunk_lines))
                        current_chunk_lines = []
                        current_chunk_size = 0
                    
                    # Add method to current chunk
                    if current_chunk_size == 0:
                        # Start a new chunk with class context
                        current_chunk_lines.append(f"{class_def_line}")
                        current_chunk_lines.append("    # ...continuing class...")
                    
                    current_chunk_lines.append(method_content)
                    current_chunk_size += method_line_count
                
                # Add the final method chunk if any
                if current_chunk_lines:
                    chunks.append('\n'.join(current_chunk_lines))
        else:
            # Regular chunk (function or smaller class)
            chunks.append(chunk)
        
        i += 1
    
    return chunks

def convert_chunk_to_ts(chunk: str, client: OpenAI) -> str:
    """
    Convert a Python chunk to TypeScript using OpenAI.
    """
    if chunk.strip() == '':
        return ''
    system_prompt = """
    Convert the following Python code to TypeScript. Follow these rules strictly:
    1. Convert line by line as a direct port
    2. Use Zod for validation where appropriate, if required, use it to replace items from the typing module. Try to use typescript types, and never make dummy types.
    3. Do not change any comments, ensure all comments are returned. If you only return comments, as the code should not be converted as per these rules, thats ok.
    4. Do not add new code
    5. Assume imports and references will be taken care of separately. Never Write import statements.
    6. Return only the converted TypeScript code, nothing else
    7. If choosing between inherit and extend, use extend
    8. Never use interfaces, use class instead of types as much as possible.
    9. NEVER IMPLEMENT DUMMY FUNCTIONS, assume the function is already implemented, even if you cannot see it
    10. If you are converting a static function, assume it is already associated to a class, dont make up a dummy class
    11. Ensure function names are identical
    12. If something is 'exported', then ensure the export is in the same line as the declaration (e.g 'export const x = 5')
    13. Make sure the code is formatted well on return
    14. Python Classes with Methods must stay as classes with methods, do not convert to types. Even if the methods are dummy methods, assume they are implemented elsewhere, so the class must remain a class, not a type.

    NEVER MAKE DUMMY ANYTHING, ASSUME EVERYTHING IS ALREADY IMPLEMENTED
    """
    
    try:
        class Code(BaseModel):
            code: str
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.2,  # Lower temperature for more consistent results
            response_format=Code
        )
        
        return response.choices[0].message.parsed.code
    except Exception as e:
        print(f"Error converting chunk: {e}")
        print(f"Problematic chunk: {chunk}")
        return f"// Error converting chunk: {str(e)}\n// Original Python:\n/*\n{chunk}\n*/"

def should_process_file(ts_file_path: str, all_override: str) -> bool:
    """
    Determine if a file should be processed based on existence and override parameter.
    
    Args:
        ts_file_path: Path to the TypeScript file (which may or may not exist)
        all_override: One of "true", "false", or "ask" to determine override behavior
    
    Returns:
        Boolean indicating whether to process the file
    """
    if not os.path.exists(ts_file_path):
        # If TypeScript file doesn't exist, always process
        return True
        
    if all_override.lower() == "true":
        # Always override existing files
        print(f"File {ts_file_path} exists, overriding as requested.")
        return True
    elif all_override.lower() == "false":
        # Never override existing files
        print(f"File {ts_file_path} exists, skipping as requested.")
        return False
    elif all_override.lower() == "ask":
        # Ask the user whether to override
        response = input(f"File {ts_file_path} already exists. Override? (y/n): ")
        return response.lower().startswith('y')
    else:
        # Default behavior for invalid parameters
        print(f"Invalid override parameter '{all_override}', defaulting to not override.")
        return False

def process_file(py_file_path: str, client: OpenAI, all_override: str = "ask") -> None:
    """
    Process a single Python file, converting it to TypeScript.
    """
    # Create TypeScript file path
    ts_file_path = str(py_file_path).replace('.py', '.ts')
    
    # Check if we should process this file
    if not should_process_file(ts_file_path, all_override):
        return
    
    # Read Python file
    with open(py_file_path, 'r', encoding='utf-8') as f:
        py_content = f.read()
    
    # Extract initial comments and convert them to TypeScript format
    ts_comments, content_after_comments = extract_initial_comments(py_content)
    
    # Process imports
    ts_imports, content_after_imports = process_imports(content_after_comments)
    
    # Split the remaining content into chunks
    chunks = split_python_file(content_after_imports)
    
    # Convert each chunk
    ts_content = []
    for i, chunk in enumerate(chunks):
        print(f"Converting chunk {i+1}/{len(chunks)} from {py_file_path}")
        ts_chunk = convert_chunk_to_ts(chunk, client)
        ts_content.append(ts_chunk)
    
    # Write TypeScript file with comments at the top
    with open(ts_file_path, 'w', encoding='utf-8') as f:
        # Write initial comments if any
        if ts_comments:
            f.write(ts_comments + '\n\n')
        
        # Write imports if any
        if ts_imports:
            f.write('\n'.join(ts_imports) + '\n\n')
        
        # Write converted content
        f.write('\n\n'.join(ts_content))
    
    print(f"Converted {py_file_path} to {ts_file_path}")

def process_path(input_path: str, client: OpenAI, all_override: str = "ask") -> None:
    """
    Process a file or directory, converting Python files to TypeScript.
    """
    path = Path(input_path)
    
    if path.is_file() and path.suffix == '.py':
        process_file(str(path), client, all_override)
    elif path.is_dir():
        for py_file in path.glob('**/*.py'):
            process_file(str(py_file), client, all_override)
            # Show countdown timer with option to cancel
            print("-" * 80)

            for i in range(5, 0, -1):
                print(f"\rPress Ctrl+C to stop processing files ({i} sec remaining)", end="", flush=True)
                time.sleep(1)
            print("\r" + " " * 80, end="", flush=True)
            print("Continuing...")
            print()  # Add newline after countdown


    else:
        print(f"Skipping {input_path} - not a Python file or directory")

def main() -> None:
    """
    Main entry point for the script.
    """
    # Initialize OpenAI client with hardcoded API key
    client = OpenAI(api_key=API_KEY)
    
    # Process the hardcoded input path
    print(f"Processing {INPUT_PATH}...")
    process_path(INPUT_PATH, client, ALL_OVERRIDE)
    print("Conversion complete!")
    
if __name__ == "__main__":
    main()