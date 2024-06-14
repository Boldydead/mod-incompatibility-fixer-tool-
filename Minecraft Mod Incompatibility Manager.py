import time
import zipfile
import json
import os
import logging
import jpype
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from jpype import JClass, java
import unittest
import threading
import tempfile

# Configure logging to file in the user's "Documents" directory
documents_folder = os.path.join(os.path.expanduser("~"), "Documents")
log_file_path = os.path.join(documents_folder, "mod_incompatibility_manager.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

javassist_jar_path = 'C:/Users/jay5a/Documents/javassist/jboss-javassist-javassist-4c998e0/javassist.jar'

known_incompatibilities = {}
json_file_path = os.path.join(documents_folder, "known_incompatibilities.json")


def start_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[javassist_jar_path])


def import_javassist_classes():
    global ClassPool, NotFoundException, CtClass
    ClassPool = JClass('javassist.ClassPool')
    NotFoundException = JClass('javassist.NotFoundException')
    CtClass = JClass('javassist.CtClass')


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
    start_jvm()
    import_javassist_classes()
    method_signatures = []
    bytecode = b''

    try:
        class_pool = ClassPool.getDefault()
        ct_class = class_pool.makeClass(java.io.ByteArrayInputStream(class_data))
        methods = ct_class.getDeclaredMethods()
        for method in methods:
            method_signatures.append(str(method))
            bytecode += method.getMethodInfo().getCodeAttribute().getCode()
    except Exception as e:
        logging.error(f"Error extracting method signatures and bytecode: {e}")

    return method_signatures, bytecode


def check_incompatibilities(mods):
    incompatibilities = {}
    mixin_conflicts = {}
    for mod_id, mod_info in mods.items():
        for other_mod_id, other_mod_info in mods.items():
            if mod_id != other_mod_id:
                if mod_info['mod_id'] == other_mod_info['mod_id']:
                    incompatibilities[mod_id] = other_mod_id

                if compare_bytecode(mod_info['bytecode'], other_mod_info['bytecode']):
                    mixin_conflicts[mod_id] = other_mod_id

    return incompatibilities, mixin_conflicts


def compare_bytecode(bc1, bc2):
    if compare_control_flow(bc1, bc2):
        return True
    return False


def compare_control_flow(bytecode1, bytecode2):
    cfg1 = build_control_flow_graph(bytecode1)
    cfg2 = build_control_flow_graph(bytecode2)
    return cfg1 == cfg2


def build_control_flow_graph(bytecode):
    return {}


def load_incompatibilities_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}


def save_incompatibilities_to_file(file_path):
    with open(file_path, 'w') as file:
        json.dump(known_incompatibilities, file, indent=4)


def save_mod_metadata(mod_path, mod_info):
    metadata_file = mod_path.replace('.jar', '.json')
    with open(metadata_file, 'w') as file:
        json.dump(mod_info, file, indent=4)


def write_mods_to_text_file(file_path, mod_info):
    with open(file_path, 'w') as file:
        for mod_id, mod_details in mod_info.items():
            file.write(f"Mod ID: {mod_id}\n")
            file.write(f"Name: {mod_details['name']}\n")
            file.write(f"Version: {mod_details['version']}\n")
            file.write("\n")


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
            mod_name = simpledialog.askstring("Modify Mod", "Enter new Mod Name:", initialvalue=known_incompatibilities[mod_id]['name'])
            mod_version = simpledialog.askstring("Modify Mod", "Enter new Mod Version:", initialvalue=known_incompatibilities[mod_id]['version'])
            known_incompatibilities[mod_id]['name'] = mod_name
            known_incompatibilities[mod_id]['version'] = mod_version
            save_incompatibilities_to_file(json_file_path)
        else:
            messagebox.showerror("Error", "Mod ID not found")

    def remove_mod(self):
        mod_id = simpledialog.askstring("Remove Mod", "Enter Mod ID to Remove:")
        if mod_id in known_incompatibilities:
            del known_incompatibilities[mod_id]
            save_incompatibilities_to_file(json_file_path)
        else:
            messagebox.showerror("Error", "Mod ID not found")

    def scan_mods(self):
        for mod_id, mod_info in known_incompatibilities.items():
            mod_path = mod_info['name']
            mod_info = scan_mod(mod_path)
            known_incompatibilities[mod_info['mod_id']] = mod_info
        save_incompatibilities_to_file(json_file_path)

    def cancel_action(self):
        self.destroy()


class TestModIncompatibilityManager(unittest.TestCase):
    def test_extract_metadata_from_jar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jar_path = os.path.join(temp_dir, 'test_mod.jar')
            with zipfile.ZipFile(jar_path, 'w') as jar:
                info_content = json.dumps([{
                    "modid": "testmod",
                    "name": "Test Mod",
                    "version": "1.0.0",
                    "mcversion": "1.16.5",
                    "url": "https://example.com"
                }])
                jar.writestr('mcmod.info', info_content)

            with zipfile.ZipFile(jar_path, 'r') as jar:
                metadata = extract_metadata_from_jar(jar)
                self.assertEqual(metadata['mod_id'], 'testmod')
                self.assertEqual(metadata['name'], 'Test Mod')
                self.assertEqual(metadata['version'], '1.0.0')
                self.assertEqual(metadata['mcversion'], '1.16.5')
                self.assertEqual(metadata['url'], 'https://example.com')

    def test_scan_mod(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jar_path = os.path.join(temp_dir, 'test_mod.jar')
            with zipfile.ZipFile(jar_path, 'w') as jar:
                info_content = json.dumps([{
                    "modid": "testmod",
                    "name": "Test Mod",
                    "version": "1.0.0",
                    "mcversion": "1.16.5",
                    "url": "https://example.com"
                }])
                jar.writestr('mcmod.info', info_content)

            mod_info = scan_mod(jar_path)
            self.assertIn('mod_id', mod_info)
            self.assertIn('name', mod_info)
            self.assertIn('version', mod_info)
            self.assertIn('classes', mod_info)
            self.assertIn('methods', mod_info)
            self.assertIn('bytecode', mod_info)

    def test_check_incompatibilities(self):
        mod_info1 = {
            'mod_id': 'mod1',
            'bytecode': b'\x00\x01\x02'
        }
        mod_info2 = {
            'mod_id': 'mod2',
            'bytecode': b'\x00\x01\x02'
        }
        mods = {'mod1': mod_info1, 'mod2': mod_info2}
        incompatibilities, mixin_conflicts = check_incompatibilities(mods)
        self.assertIn('mod1', mixin_conflicts)
        self.assertEqual(mixin_conflicts['mod1'], 'mod2')


def background_save():
    while True:
        save_incompatibilities_to_file(json_file_path)
        time.sleep(300)


if __name__ == '__main__':
    threading.Thread(target=background_save, daemon=True).start()
    unittest.main(exit=False)
    app = ModIncompatibilityGUI()
    app.mainloop()
