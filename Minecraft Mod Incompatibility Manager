import os
import zipfile
import json
import jpype
import jpype.imports
from jpype.types import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from difflib import SequenceMatcher
import networkx as nx
import tkinter as tk
from tkinter import simpledialog, filedialog
from threading import Thread, Event

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced known incompatibilities dictionary with advanced methods
known_incompatibilities = {
    'modid1': {
        'incompatible_mods': {
            'incompatible_modid1': {'reason': 'Causes crashes', 'severity': 'high'},
            'incompatible_modid2': {'reason': 'Breaks functionality', 'severity': 'medium'}
        }
    },
    'modid2': {
        'incompatible_mods': {
            'incompatible_modid3': {'reason': 'Conflicts with features', 'severity': 'high'},
            'incompatible_modid4': {'reason': 'Causes lag', 'severity': 'low'}
        }
    },
    # Add more known incompatibilities here
}

# Determine the path to the user's "Documents" folder
documents_path = os.path.join(os.path.expanduser("~"), "Documents")
json_file_path = os.path.join(documents_path, "known_incompatibilities.json")
text_file_path = os.path.join(documents_path, "mod_incompatibilities.txt")

# Default mod directory
default_mod_directory = "C:/Users/jay5a/curseforge/minecraft/Instances/Crazy Craft Updated/mods"
mod_directory = default_mod_directory

def load_incompatibilities_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            known_incompatibilities.update(data)
            logging.info(f"Loaded incompatibilities from {file_path}")

def save_incompatibilities_to_file(file_path):
    with open(file_path, 'w') as file:
        json.dump(known_incompatibilities, file, indent=4)
        logging.info(f"Saved incompatibilities to {file_path}")

def write_mods_to_text_file(mods, file_path):
    with open(file_path, 'w') as file:
        if not mods:
            file.write("No mods found.\n")
            return

        file.write("Found mods:\n")
        for mod in mods:
            file.write(f"Mod ID: {mod['mod_id']}, Name: {mod['name']}, Version: {mod['version']}\n")
            mcversion = mod.get('mcversion', 'Unknown')
            url = mod.get('url', 'Unknown')
            file.write(f"Minecraft Version: {mcversion}, URL: {url}\n")

def write_incompatibilities_to_text_file(incompatibilities, mixin_conflicts, file_path):
    with open(file_path, 'a') as file:
        if not incompatibilities:
            file.write("No incompatibilities found.\n")
        else:
            file.write("Incompatibilities found:\n")
            for entry in incompatibilities:
                if len(entry) == 4:
                    mod1, mod2, reason, severity = entry
                    file.write(
                        f"Mod '{mod1['name']}' (ID: {mod1['mod_id']}) is incompatible with Mod '{mod2['name']}' (ID: {mod2['mod_id']}) "
                        f"due to {reason} (Severity: {severity})\n")
                elif len(entry) == 3:
                    mod1, mod2, conflicts = entry
                    if isinstance(conflicts, list):
                        file.write(
                            f"Mod '{mod1['name']}' (ID: {mod1['mod_id']}) and Mod '{mod2['name']}' (ID: {mod2['mod_id']}) have conflicting classes: {', '.join(conflicts)}\n")
                    elif isinstance(conflicts, dict):
                        file.write(
                            f"Mod '{mod1['name']}' (ID: {mod1['mod_id']}) and Mod '{mod2['name']}' (ID: {mod2['mod_id']}) have conflicting methods:\n")
                        for class_name, methods in conflicts.items():
                            file.write(f"  Class '{class_name}' has conflicting methods: {', '.join(methods)}\n")

        if not mixin_conflicts:
            file.write("No mixin conflicts found.\n")
        else:
            file.write("Mixin conflicts found:\n")
            for mod1, mod2, class_name, method_name in mixin_conflicts:
                file.write(f"Mod '{mod1['name']}' (ID: {mod1['mod_id']}) and Mod '{mod2['name']}' (ID: {mod2['mod_id']}) "
                           f"have conflicting mixins on class '{class_name}', method '{method_name}'\n")

# Function to continuously save the incompatibilities to a text file
def background_save(event, mods, file_path, interval=5):
    while not event.is_set():
        write_mods_to_text_file(mods, file_path)
        incompatibilities, mixin_conflicts = check_incompatibilities(mods)
        write_incompatibilities_to_text_file(incompatibilities, mixin_conflicts, file_path)
        event.wait(interval)

# Path to the javassist.jar file
javassist_jar_path = os.path.join(documents_path, 'C:/Users/jay5a/Documents/javassist/jboss-javassist-javassist-4c998e0/javassist.jar')

# Start the JVM if not already started
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[javassist_jar_path])

# Import Java classes using JPype
from jpype.imports import registerDomain

registerDomain('javassist')

# Importing javassist classes
try:
    from javassist import ClassPool, NotFoundException
    from javassist.bytecode import CodeAttribute, ConstPool, Opcode

    logging.info("Successfully imported javassist classes.")
except ImportError as e:
    logging.error(f"Error importing javassist classes: {e}")
    jpype.shutdownJVM()
    raise

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

def extract_method_signatures_and_bytecode(class_data):
    # Using javassist to extract method signatures and bytecode
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

def compare_opcodes(opcode1, opcode2):
    # Compare opcodes using SequenceMatcher for better accuracy
    matcher = SequenceMatcher(None, opcode1, opcode2)
    return matcher.ratio() > 0.9  # Adjust the threshold as needed

def build_control_flow_graph(bytecode):
    # Build a control flow graph (CFG) from the bytecode
    graph = nx.DiGraph()
    instructions = bytecode.splitlines()
    for i, instruction in enumerate(instructions):
        graph.add_node(i, instruction=instruction)
        if i < len(instructions) - 1:
            graph.add_edge(i, i + 1)
    return graph

def compare_control_flow(bytecode1, bytecode2):
    # Compare control flow graphs using graph isomorphism
    cfg1 = build_control_flow_graph(bytecode1)
    cfg2 = build_control_flow_graph(bytecode2)
    return nx.is_isomorphic(cfg1, cfg2)

def compare_bytecode(bytecode1, bytecode2):
    # Advanced bytecode comparison logic
    # Compare opcode sequences
    if compare_opcodes(bytecode1, bytecode2):
        return True

    # Compare control flow graphs
    if compare_control_flow(bytecode1, bytecode2):
        return True

    return False

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
                            if compare_bytecode(bc1, bc2):  # Advanced comparison logic
                                if class_name not in common_bytecode:
                                    common_bytecode[class_name] = []
                                common_bytecode[class_name].append(method_name)
            if common_bytecode:
                incompatibilities.append((mod1, mod2, 'Conflicting bytecode', 'high'))

            # Check for mixin conflicts
            if 'mixins' in mod1 and 'mixins' in mod2:
                for mixin1 in mod1['mixins']:
                    for mixin2 in mod2['mixins']:
                        if mixin1['class'] == mixin2['class'] and mixin1['method'] == mixin2['method']:
                            if mixin1['priority'] == mixin2['priority']:
                                mixin_conflicts.append((mod1, mod2, mixin1['class'], mixin1['method']))

    return incompatibilities, mixin_conflicts

def add_incompatibility(mod_id, incompatible_mod_id, reason, severity):
    if mod_id not in known_incompatibilities:
        known_incompatibilities[mod_id] = {'incompatible_mods': {}}
    known_incompatibilities[mod_id]['incompatible_mods'][incompatible_mod_id] = {'reason': reason, 'severity': severity}
    logging.info(f"Added incompatibility: {mod_id} -> {incompatible_mod_id} (Reason: {reason}, Severity: {severity})")
    write_incompatibilities_to_text_file(check_incompatibilities(scan_mods(mod_directory))[0], check_incompatibilities(scan_mods(mod_directory))[1], text_file_path)

def remove_incompatibility(mod_id, incompatible_mod_id):
    if mod_id in known_incompatibilities and incompatible_mod_id in known_incompatibilities[mod_id]['incompatible_mods']:
        del known_incompatibilities[mod_id]['incompatible_mods'][incompatible_mod_id]
        logging.info(f"Removed incompatibility: {mod_id} -> {incompatible_mod_id}")
        write_incompatibilities_to_text_file(check_incompatibilities(scan_mods(mod_directory))[0], check_incompatibilities(scan_mods(mod_directory))[1], text_file_path)

def modify_incompatibility(mod_id, incompatible_mod_id, reason=None, severity=None):
    if mod_id in known_incompatibilities and incompatible_mod_id in known_incompatibilities[mod_id]['incompatible_mods']:
        if reason:
            known_incompatibilities[mod_id]['incompatible_mods'][incompatible_mod_id]['reason'] = reason
        if severity:
            known_incompatibilities[mod_id]['incompatible_mods'][incompatible_mod_id]['severity'] = severity
        logging.info(f"Modified incompatibility: {mod_id} -> {incompatible_mod_id} (Reason: {reason}, Severity: {severity})")
        write_incompatibilities_to_text_file(check_incompatibilities(scan_mods(mod_directory))[0], check_incompatibilities(scan_mods(mod_directory))[1], text_file_path)

def open_add_dialog():
    mod_id = simpledialog.askstring("Input", "Enter the mod ID:")
    incompatible_mod_id = simpledialog.askstring("Input", "Enter the incompatible mod ID:")
    reason = simpledialog.askstring("Input", "Enter the reason for incompatibility:")
    severity = simpledialog.askstring("Input", "Enter the severity (low, medium, high, critical):")
    if mod_id and incompatible_mod_id and reason and severity:
        add_incompatibility(mod_id, incompatible_mod_id, reason, severity)
        refresh_listbox()

def open_remove_dialog():
    mod_id = simpledialog.askstring("Input", "Enter the mod ID:")
    incompatible_mod_id = simpledialog.askstring("Input", "Enter the incompatible mod ID:")
    if mod_id and incompatible_mod_id:
        remove_incompatibility(mod_id, incompatible_mod_id)
        refresh_listbox()

def open_modify_dialog():
    mod_id = simpledialog.askstring("Input", "Enter the mod ID:")
    incompatible_mod_id = simpledialog.askstring("Input", "Enter the incompatible mod ID:")
    reason = simpledialog.askstring("Input", "Enter the new reason for incompatibility (leave blank to keep current):")
    severity = simpledialog.askstring("Input", "Enter the new severity (leave blank to keep current):")
    if mod_id and incompatible_mod_id:
        modify_incompatibility(mod_id, incompatible_mod_id, reason, severity)
        refresh_listbox()

def open_directory_dialog():
    global mod_directory
    mod_directory = filedialog.askdirectory(initialdir=default_mod_directory, title="Select Mod Directory")
    if mod_directory:
        mods.clear()
        mods.extend(scan_mods(mod_directory))
        write_mods_to_text_file(mods, text_file_path)
        refresh_listbox()

def refresh_listbox():
    listbox.delete(0, tk.END)
    for mod_id, details in known_incompatibilities.items():
        for incompatible_mod_id, info in details['incompatible_mods'].items():
            listbox.insert(tk.END, f"{mod_id} -> {incompatible_mod_id}: {info['reason']} (Severity: {info['severity']})")

if __name__ == "__main__":
    # Load incompatibilities from file
    load_incompatibilities_from_file(json_file_path)

    # Create GUI
    root = tk.Tk()
    root.title("Incompatibility Manager")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    listbox = tk.Listbox(frame, width=80, height=20)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    add_button = tk.Button(button_frame, text="Add Incompatibility", command=open_add_dialog)
    add_button.grid(row=0, column=0, padx=10)

    remove_button = tk.Button(button_frame, text="Remove Incompatibility", command=open_remove_dialog)
    remove_button.grid(row=0, column=1, padx=10)

    modify_button = tk.Button(button_frame, text="Modify Incompatibility", command=open_modify_dialog)
    modify_button.grid(row=0, column=2, padx=10)

    directory_button = tk.Button(button_frame, text="Select Mod Directory", command=open_directory_dialog)
    directory_button.grid(row=0, column=3, padx=10)

    refresh_listbox()

    mods = scan_mods(mod_directory)

    # Start background save thread
    stop_event = Event()
    save_thread = Thread(target=background_save, args=(stop_event, mods, text_file_path))
    save_thread.start()

    root.mainloop()

    # Stop the background save thread
    stop_event.set()
    save_thread.join()

    # Save incompatibilities to file before exiting
    save_incompatibilities_to_file(json_file_path)

    # Shutdown the JVM
    jpype.shutdownJVM()
