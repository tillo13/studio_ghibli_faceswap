import os
from datetime import datetime

# Global variable to define directories to exclude
DIRECTORIES_TO_EXCLUDE = ["ip_adapter", "test123", "othersecretefolder"]

def gather_python_files(root_dir, directories_to_exclude):
    """
    Gathers all .py files within the specified root directory and its subdirectories,
    excluding specified directories.

    Parameters:
        root_dir (str): The root directory to search for Python files.
        directories_to_exclude (list): List of directory names to exclude from scanning.

    Returns:
        tuple: List containing tuples of file path and file contents, list of directories.
    """
    python_files = []
    directories = set()
    excluded_directories = set()

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the directory is excluded
        if any(excluded_dir in os.path.relpath(dirpath, root_dir).split(os.sep) for excluded_dir in directories_to_exclude):
            excluded_directories.add(os.path.relpath(dirpath, root_dir))
            continue

        # Add directory structure
        directories.add(os.path.relpath(dirpath, root_dir))
        
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as file:
                    file_contents = file.read()
                python_files.append((filepath, file_contents))

    return python_files, sorted(directories), sorted(excluded_directories)

def write_to_file(output_filepath, data, directories, excluded_directories):
    """
    Writes the gathered data to a file with the number of Python files and directories,
    and the directory structure at the top, followed by a list of all Python file paths.

    Parameters:
        output_filepath (str): The file where the data will be written.
        data (list of tuples): The data to write to the file. Each tuple contains (filepath, file_content).
        directories (list): The list of directories containing .py files.
        excluded_directories (list): The list of directories that were excluded from scanning.
    """
    with open(output_filepath, 'w') as file:
        file.write(f"Number of python files: {len(data)}\n")
        file.write(f"Number of directories: {len(directories) + len(excluded_directories)}\n\n")
        
        file.write("Directory structure:\n")
        for directory in directories:
            file.write(f"{directory}\n")
        for excluded_directory in excluded_directories:
            file.write(f"{excluded_directory} [excluded from files_within.txt]\n")
        file.write("\n")
        
        file.write("List of Python file paths:\n")
        for filepath, _ in data:
            file.write(f"{filepath}\n")
        file.write("\n")
        
        for filepath, file_contents in data:
            file.write(f"{filepath}:\n{file_contents}\n")

def main():
    root_dir = "."  # Change this if your script is in a different directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = f"{timestamp}_files_within.txt"
    python_files, directories, excluded_directories = gather_python_files(root_dir, DIRECTORIES_TO_EXCLUDE)
    write_to_file(output_filepath, python_files, directories, excluded_directories)
    print(f"Files have been gathered and written to {output_filepath}")

if __name__ == "__main__":
    main()