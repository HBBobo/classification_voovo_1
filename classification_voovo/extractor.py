import os
import argparse  # Import the argument parsing library

def parse_rules(file_path):
    """
    Parses the rules file (e.g., folders.txt).
    Lines starting with '+' are additions.
    Lines starting with '-' are subtractions (exclusions).
    Returns two lists: one for inclusions, one for exclusions.
    """
    inclusions = []
    exclusions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue
                
                # Normalize path separators for cross-platform consistency
                path = os.path.normpath(stripped_line[1:])

                if stripped_line.startswith('+'):
                    inclusions.append(path)
                elif stripped_line.startswith('-'):
                    exclusions.append(path)
                else:
                    print(f"[Warning] Skipping invalid line in '{file_path}': {stripped_line}")
    except FileNotFoundError:
        print(f"[Error] The specified rule file was not found: '{file_path}'")
        return None, None
        
    return inclusions, exclusions

def should_process_path(path, inclusions, exclusions):
    """
    Determines if a given path should be processed based on inclusion/exclusion rules.
    The most specific (longest) matching rule wins.
    """
    path = os.path.normpath(path)
    
    longest_incl = ""
    longest_excl = ""

    for incl in inclusions:
        if path.startswith(incl) and len(incl) > len(longest_incl):
            longest_incl = incl
    
    for excl in exclusions:
        if path.startswith(excl) and len(excl) > len(longest_excl):
            longest_excl = excl
            
    # If the longest matching exclusion is more specific than the longest
    # matching inclusion, we do not process the path.
    if len(longest_excl) > len(longest_incl):
        return False
        
    # If an inclusion rule matches (and was not overridden by a more specific
    # exclusion), we process the path.
    if longest_incl:
        return True
        
    # Default case: if no rules match, don't process.
    return False

def walk_and_collect_structure(inclusions, exclusions):
    """
    Walks the file system based on inclusion/exclusion rules and collects
    the structure and file contents.
    """
    structure = []
    file_contents = []
    processed_paths = set()

    # We only need to start walking from the top-level inclusion paths
    # to avoid redundant work.
    walk_roots = sorted([p for p in inclusions if not any(p.startswith(q) and p != q for q in inclusions)])

    for path in walk_roots:
        if path in processed_paths:
            continue
            
        if os.path.isfile(path):
            if should_process_path(path, inclusions, exclusions):
                structure.append(os.path.basename(path))
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    content = f"[Error reading file: {e}]"
                file_contents.append((path, content))
                processed_paths.add(path)
            continue

        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=True):
                if not should_process_path(root, inclusions, exclusions):
                    dirs[:] = []
                    continue

                if root not in processed_paths:
                    level = root.replace(path, '').count(os.sep) if root != path else 0
                    indent = '  ' * level
                    structure.append(f"{indent}{os.path.basename(root)}/")
                    processed_paths.add(root)
                
                sub_indent = '  ' * (level + 1)
                
                for file in sorted(files):
                    filepath = os.path.join(root, file)
                    if should_process_path(filepath, inclusions, exclusions):
                        if filepath not in processed_paths:
                            structure.append(f"{sub_indent}{file}")
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    content = f.read()
                            except Exception as e:
                                content = f"[Error reading file: {e}]"
                            file_contents.append((filepath, content))
                            processed_paths.add(filepath)

                dirs_to_keep = []
                for d in sorted(dirs):
                    dirpath = os.path.join(root, d)
                    if should_process_path(dirpath, inclusions, exclusions) or \
                       any(incl.startswith(dirpath) for incl in inclusions):
                        dirs_to_keep.append(d)
                dirs[:] = dirs_to_keep
        else:
            if path not in processed_paths:
                structure.append(f"[Not found: {path}]")
                processed_paths.add(path)

    return structure, file_contents

def write_output(output_path, structure, file_contents):
    """Writes the collected structure and contents to the output file."""
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("--- File Structure ---\n")
        for line in structure:
            out.write(line + '\n')

        out.write("\n--- File Contents ---\n")
        for path, content in file_contents:
            out.write(f"\n## {path} ##\n")
            out.write(content + '\n')

if __name__ == "__main__":
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Extracts a project's structure and file contents into a single text file based on rules."
    )
    parser.add_argument(
        '-i', '--input',
        default='folders.txt',
        help='The input file with inclusion/exclusion rules. Default: folders.txt'
    )
    parser.add_argument(
        '-o', '--output',
        default='output.txt',
        help='The path for the generated output file. Default: output.txt'
    )
    args = parser.parse_args()
    
    # Use the filenames from the parsed arguments
    folders_txt = args.input
    output_txt = args.output
    # --- End of argument parsing ---

    print(f"Reading rules from '{folders_txt}'...")
    inclusions, exclusions = parse_rules(folders_txt)
    
    # Proceed only if the rules file was successfully read
    if inclusions is not None:
        print(f"Found {len(inclusions)} inclusion and {len(exclusions)} exclusion rules.")
        print("Processing...")
        
        structure, contents = walk_and_collect_structure(inclusions, exclusions)
        write_output(output_txt, structure, contents)
        
        print(f"\nSuccess! Project structure and contents written to '{output_txt}'.")
    
    input("Press Enter to exit...")