import os
import zipfile
import logging
import json
import tkinter as tk
from tkinter import simpledialog, filedialog
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import networkx as nx
import jpype
import jpype.imports
from jpype.types import *
from threading import Thread, Event
import unittest

# Setup logging to a file in the user's "Documents" directory
documents_folder = os.path.join(os.path.expanduser("~"), "Documents")
log_file_path = os.path.join(documents_folder, "mod_incompatibility_manager.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
known_incompatibilities = {}
mod_directory = os.path.join(os.path.expanduser("~"), "curseforge/minecraft/Instances/Crazy Craft Updated/mods")
default_mod_directory = mod_directory
default_javassist_directory = os.path.expanduser("~")
text_file_path = os.path.join(documents_folder, "mod_list.txt")
json_file_path = os.path.join(documents_folder, "incompatibilities.json")
javassist_jar_path = "C:/Users/jay5a/Documents/javassist/jboss-javassist-javassist-4c998e0/javassist.jar"  # Path to javassist.jar


# Start the JVM
def start_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[javassist_jar_path])


def import_javassist_classes():
    global ClassPool, NotFoundException, CtClass
    ClassPool = jpype.JClass('javassist.ClassPool')
    NotFoundException = jpype.JClass('javassist.NotFoundException')
    CtClass = jpype.JClass('javassist.CtClass')


# Load and save incompatibilities to/from JSON file
def load_incompatibilities_from_file(filepath):
    global known_incompatibilities
    try:
        with open(filepath, 'r') as file:
            known_incompatibilities = json.load(file)
        logging.info(f"Incompatibilities loaded from {filepath}")
    except FileNotFoundError:
        logging.warning(f"{filepath} not found. Starting with an empty incompatibilities list.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {filepath}: {e}")


def save_incompatibilities_to_file(filepath):
    global known_incompatibilities
    with open(filepath, 'w') as file:
        json.dump(known_incompatibilities, file, indent=4)
    logging.info(f"Incompatibilities saved to {filepath}")


# Function to write mod information to text file
def write_mods_to_text_file(mods, filepath):
    with open(filepath, 'w') as file:
        for mod in mods:
            file.write(f"Mod ID: {mod['mod_id']}, Name: {mod['name']}, Version: {mod['version']}\n")
    logging.info(f"Mod list saved to {filepath}")


# Extract metadata from JAR files
def extract_metadata_from_jar(jar_file):
    metadata = {}
    try:
        if 'mcmod.info' in jar_file.namelist():
            with jar_file.open('mcmod.info') as info_file:
                mod_metadata = json.load(info_file)
                metadata['name'] = mod_metadata[0].get('name', 'Unknown')
                metadata['version'] = mod_metadata[0].get('version', 'Unknown')
                metadata['mod_id'] = mod_metadata[0].get('modid', 'Unknown')
                metadata['mcversion'] = mod_metadata[0].get('mcversion', 'Unknown')
                metadata['url'] = mod_metadata[0].get('url', 'Unknown')
        elif 'META-INF/mods.toml' in jar_file.namelist():
            with jar_file.open('META-INF/mods.toml') as info_file:
                toml_data = info_file.read().decode('utf-8').splitlines()
                for line in toml_data:
                    if line.startswith('modId'):
                        metadata['mod_id'] = line.split('=')[1].strip().strip('"')
                    elif line.startswith('version'):
                        metadata['version'] = line.split('=')[1].strip().strip('"')
                    elif line.startswith('displayName'):
                        metadata['name'] = line.split('=')[1].strip().strip('"')
                    elif line.startswith('minecraftVersion'):
                        metadata['mcversion'] = line.split('=')[1].strip().strip('"')
                    elif line.startswith('url'):
                        metadata['url'] = line.split('=')[1].strip().strip('"')
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")

    return metadata


# Scan a single mod
def scan_mod(mod_path):
    logging.info(f"Scanning mod: {mod_path}")
    mod_info = {"name": mod_path, "version": None, "mod_id": None, "classes": [], "methods": {}, "bytecode": {}}

    try:
        with zipfile.ZipFile(mod_path, 'r') as jar_file:
            metadata = extract_metadata_from_jar(jar_file)
            mod_info.update(metadata)

            for file_name in jar_file.namelist():
                if file_name.endswith(".class"):
                    mod_info['classes'].append(file_name)
                    with jar_file.open(file_name) as class_file:
                        class_data = class_file.read()
                        method_signatures, bytecode = extract_method_signatures_and_bytecode(class_data)
                        mod_info['methods'][file_name] = method_signatures
                        mod_info['bytecode'][file_name] = bytecode
    except Exception as e:
        logging.error(f"Error reading mod {mod_path}: {e}")

    return mod_info


# Extract method signatures and bytecode from class data
def extract_method_signatures_and_bytecode(class_data):
    method_signatures = []
    bytecode = {}

    pool = ClassPool.getDefault()
    try:
        ct_class = pool.makeClass(JArray(JByte)(class_data))
        for method in ct_class.getDeclaredMethods():
            signature = method.getLongName()
            method_signatures.append(signature)
            code_attr = method.getMethodInfo().getCodeAttribute()
            if code_attr:
                bytecode[method.getName()] = code_attr.toString()
    except NotFoundException as e:
        logging.error(f"Error parsing class: {e}")

    return method_signatures, bytecode


# Scan all mods in the directory
def scan_mods(mods_directory):
    if not os.path.exists(mods_directory):
        logging.error(f"The directory {mods_directory} does not exist.")
        return []

    mods = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(scan_mod, os.path.join(mods_directory, filename))
                   for filename in os.listdir(mods_directory) if filename.endswith(".jar")]

        for future in as_completed(futures):
            try:
                mod_info = future.result()
                mods.append(mod_info)
            except Exception as e:
                logging.error(f"Error processing mod: {e}")

    return mods


# Compare bytecode for conflicts
def compare_bytecode(bytecode1, bytecode2):
    return bytecode1 == bytecode2


# Check for incompatibilities between mods
def check_incompatibilities(mods):
    incompatibilities = []
    mixin_conflicts = []

    for mod in mods:
        mod_id = mod['mod_id']
        if mod_id in known_incompatibilities:
            for incompatible_mod_id, details in known_incompatibilities[mod_id]['incompatible_mods'].items():
                for other_mod in mods:
                    if other_mod['mod_id'] == incompatible_mod_id:
                        incompatibilities.append((mod, other_mod, details['reason'], details['severity']))

    # Check for class, method, and bytecode conflicts
    for i, mod1 in enumerate(mods):
        for mod2 in mods[i + 1:]:
            common_classes = set(mod1['classes']) & set(mod2['classes'])
            if common_classes:
                incompatibilities.append((mod1, mod2, 'Conflicting classes', 'high'))

            common_methods = {}
            for class_name, methods1 in mod1['methods'].items():
                if class_name in mod2['methods']:
                    methods2 = mod2['methods'][class_name]
                    common = set(methods1) & set(methods2)
                    if common:
                        common_methods[class_name] = list(common)
            if common_methods:
                incompatibilities.append((mod1, mod2, 'Conflicting methods', 'high'))

            common_bytecode = {}
            for class_name, bytecode1 in mod1['bytecode'].items():
                if class_name in mod2['bytecode']:
                    bytecode2 = mod2['bytecode'][class_name]
                    for method_name, bc1 in bytecode1.items():
                        if method_name in bytecode2:
                            bc2 = bytecode2[method_name]
                            if compare_bytecode(bc1, bc2):
                                if class_name not in common_bytecode:
                                    common_bytecode[class_name] = []
                                common_bytecode[class_name].append(method_name)
            if common_bytecode:
                incompatibilities.append((mod1, mod2, 'Conflicting bytecode', 'high'))

            # Check for mixin conflicts
            if 'mixins' in mod1 and 'mixins' in mod2:
                for mixin1 in mod1['mixins']:
                    for mixin2 in mod2['mixins']:
                        if mixin1['target'] == mixin2['target']:
                            mixin_conflicts.append((mod1, mod2, mixin1, mixin2, 'Conflicting mixins', 'high'))

    return incompatibilities, mixin_conflicts


# Build control flow graph from bytecode
def build_control_flow_graph(bytecode):
    graph = nx.DiGraph()
    instructions = bytecode.splitlines()
    for i, instruction in enumerate(instructions):
        graph.add_node(i, instruction=instruction)
        if i < len(instructions) - 1:
            graph.add_edge(i, i + 1)
    return graph


# Compare control flow of two bytecode sequences
def compare_control_flow(bytecode1, bytecode2):
    cfg1 = build_control_flow_graph(bytecode1)
    cfg2 = build_control_flow_graph(bytecode2)
    return nx.is_isomorphic(cfg1, cfg2)


# Save mod metadata to file
def save_mod_metadata(mods, filepath):
    with open(filepath, 'w') as file:
        json.dump(mods, file, indent=4)


# GUI Implementation
class ModIncompatibilityGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mod Incompatibility Manager")
        self.geometry("400x200")

        self.label = tk.Label(self, text="Mod Incompatibility Manager")
        self.label.pack(pady=10)

        self.add_button = tk.Button(self, text="Add Mod", command=self.add_mod)
        self.add_button.pack(pady=5)

        self.modify_button = tk.Button(self, text="Modify Mod", command=self.modify_mod)
        self.modify_button.pack(pady=5)

        self.remove_button = tk.Button(self, text="Remove Mod", command=self.remove_mod)
        self.remove_button.pack(pady=5)

        self.scan_button = tk.Button(self, text="Scan Mods", command=self.scan_mods)
        self.scan_button.pack(pady=5)

        self.cancel_button = tk.Button(self, text="Cancel", command=self.cancel_action)
        self.cancel_button.pack(pady=5)

    def add_mod(self):
        mod_path = filedialog.askopenfilename(title="Select Mod File", filetypes=[("JAR Files", "*.jar")])
        if mod_path:
            mod_info = scan_mod(mod_path)
            known_incompatibilities[mod_info['mod_id']] = {
                'name': mod_info['name'],
                'version': mod_info['version'],
                'incompatible_mods': {}
            }
            save_incompatibilities_to_file(json_file_path)

    def modify_mod(self):
        mod_id = simpledialog.askstring("Modify Mod", "Enter Mod ID to Modify:")
        if mod_id in known_incompatibilities:
            mod_name = simpledialog.askstring("Modify Mod", "Enter new Mod Name:",
                                              initialvalue=known_incompatibilities[mod_id]['name'])
            mod_version = simpledialog.askstring("Modify Mod", "Enter new Mod Version:",
                                                 initialvalue=known_incompatibilities[mod_id]['version'])
            known_incompatibilities[mod_id]['name'] = mod_name
            known_incompatibilities[mod_id]['version'] = mod_version
            save_incompatibilities_to_file(json_file_path)

    def remove_mod(self):
        mod_id = simpledialog.askstring("Remove Mod", "Enter Mod ID to Remove:")
        if mod_id in known_incompatibilities:
            del known_incompatibilities[mod_id]
            save_incompatibilities_to_file(json_file_path)

    def scan_mods(self):
        mods = scan_mods(mod_directory)
        write_mods_to_text_file(mods, text_file_path)
        save_mod_metadata(mods, json_file_path)
        incompatibilities, mixin_conflicts = check_incompatibilities(mods)

        # Display results
        result_window = tk.Toplevel(self)
        result_window.title("Scan Results")
        result_window.geometry("600x400")
        result_text = tk.Text(result_window)
        result_text.pack(expand=True, fill=tk.BOTH)
        for incompatibility in incompatibilities:
            result_text.insert(tk.END,
                               f"Incompatibility found between {incompatibility[0]['name']} and {incompatibility[1]['name']}: {incompatibility[2]} (Severity: {incompatibility[3]})\n")
        for conflict in mixin_conflicts:
            result_text.insert(tk.END,
                               f"Mixin conflict found between {conflict[0]['name']} and {conflict[1]['name']} targeting {conflict[2]['target']}: {conflict[4]} (Severity: {conflict[5]})\n")

    def cancel_action(self):
        self.destroy()


# Run the GUI application
if __name__ == "__main__":
    start_jvm()
    import_javassist_classes()
    load_incompatibilities_from_file(json_file_path)
    app = ModIncompatibilityGUI()
    app.mainloop()
    save_incompatibilities_to_file(json_file_path)


# Unit tests for the functionality
class TestModIncompatibilityManager(unittest.TestCase):

    def test_extract_metadata_from_jar(self):
        with zipfile.ZipFile("test_mod.jar", 'w') as jar_file:
            jar_file.writestr("mcmod.info", json.dumps([{
                "modid": "testmod",
                "name": "Test Mod",
                "version": "1.0.0",
                "mcversion": "1.16.5",
                "url": "http://example.com"
            }]))
        metadata = extract_metadata_from_jar(zipfile.ZipFile("test_mod.jar"))
        self.assertEqual(metadata['mod_id'], "testmod")
        self.assertEqual(metadata['name'], "Test Mod")
        self.assertEqual(metadata['version'], "1.0.0")
        self.assertEqual(metadata['mcversion'], "1.16.5")
        self.assertEqual(metadata['url'], "http://example.com")

    def test_scan_mod(self):
        mod_info = scan_mod("test_mod.jar")
        self.assertEqual(mod_info['mod_id'], "testmod")
        self.assertEqual(mod_info['name'], "Test Mod")
        self.assertEqual(mod_info['version'], "1.0.0")
        self.assertEqual(mod_info['mcversion'], "1.16.5")
        self.assertEqual(mod_info['url'], "http://example.com")

    def test_check_incompatibilities(self):
        mods = [
            {"mod_id": "mod1", "classes": ["Class1"], "methods": {"Class1": ["method1"]},
             "bytecode": {"Class1": {"method1": "bytecode1"}}},
            {"mod_id": "mod2", "classes": ["Class1"], "methods": {"Class1": ["method1"]},
             "bytecode": {"Class1": {"method1": "bytecode1"}}},
        ]
        incompatibilities, mixin_conflicts = check_incompatibilities(mods)
        self.assertTrue(len(incompatibilities) > 0)


if __name__ == "__main__":
    unittest.main()
