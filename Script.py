import os

# Set to the root directory of your project
root_dir = os.path.dirname(os.path.abspath(__file__))
output_file = "combined_output.md"

frontend_src = os.path.join(root_dir, "frontend", "src")
main_py = os.path.join(root_dir, "backend", "app", "main.py")

def collect_files():
    files = []
    # Add main.py if it exists
    if os.path.exists(main_py):
        files.append((main_py, "backend/app/main.py"))
    # Add all files in frontend/src and subfolders
    for dirpath, _, filenames in os.walk(frontend_src):
        for filename in sorted(filenames):
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir)
            files.append((full_path, rel_path))
    return files

# Start writing
with open(output_file, "w", encoding="utf-8") as outfile:
    files_to_include = collect_files()
    for full_path, rel_path in files_to_include:
        outfile.write(f"\n\n# {rel_path}\n\n")
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                outfile.write(f.read())
        except Exception as e:
            outfile.write(f"_Error reading {rel_path}: {e}_")
        outfile.write("\n\n---\n")
