import subprocess
import json
import os
import logging
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from queue import Queue, Empty
import tempfile
import unittest
import unittest.mock as mock
import threading
import time
import base64
import zipfile
import sys

# Configure logging to file in the user's "Documents" directory
documents_folder = os.path.join(os.path.expanduser("~"), "Documents")
log_file_path = os.path.join(documents_folder, "mod_incompatibility_manager.log")
logging.basicConfig(filename=log_file_path, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

conflicts_txt_file_path = os.path.join(documents_folder, "mod_conflicts.txt")

known_incompatibilities = {}
json_file_path = os.path.join(documents_folder, "known_incompatibilities.json")

# Java code to be written to ModScanner.java
java_code = """
import javassist.*;
import java.io.*;
import java.util.zip.*;
import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;
import java.nio.file.*;
import java.util.stream.*;
import java.util.Properties;

public class ModScanner {

    public static JSONObject extractMetadataFromJar(String jarPath) throws IOException, JSONException {
        JSONObject metadata = new JSONObject();
        ZipFile jarFile = new ZipFile(jarPath);
        try {
            System.out.println("Processing JAR: " + jarPath);

            boolean metadataFound = false;

            // Check for mcmod.info
            if (jarFile.getEntry("mcmod.info") != null) {
                metadataFound = true;
                InputStream infoStream = jarFile.getInputStream(jarFile.getEntry("mcmod.info"));
                BufferedReader reader = new BufferedReader(new InputStreamReader(infoStream));
                StringBuilder jsonString = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    jsonString.append(line);
                }
                System.out.println("mcmod.info content: " + jsonString.toString());
                JSONArray modMetadata = new JSONArray(jsonString.toString());
                metadata.put("name", modMetadata.getJSONObject(0).optString("name", "Unknown"));
                metadata.put("version", modMetadata.getJSONObject(0).optString("version", "Unknown"));
                metadata.put("mod_id", modMetadata.getJSONObject(0).optString("modid", "Unknown"));
                metadata.put("mcversion", modMetadata.getJSONObject(0).optString("mcversion", "Unknown"));
                metadata.put("url", modMetadata.getJSONObject(0).optString("url", "Unknown"));
            }
            // Check for fabric.mod.json
            else if (jarFile.getEntry("fabric.mod.json") != null) {
                metadataFound = true;
                InputStream infoStream = jarFile.getInputStream(jarFile.getEntry("fabric.mod.json"));
                BufferedReader reader = new BufferedReader(new InputStreamReader(infoStream));
                StringBuilder jsonString = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    jsonString.append(line);
                }
                System.out.println("fabric.mod.json content: " + jsonString.toString());
                JSONObject modMetadata = new JSONObject(jsonString.toString());
                metadata.put("name", modMetadata.optString("name", "Unknown"));
                metadata.put("version", modMetadata.optString("version", "Unknown"));
                metadata.put("mod_id", modMetadata.optString("id", "Unknown"));
                JSONArray depends = modMetadata.optJSONArray("depends");
                if (depends != null) {
                    JSONObject firstDependency = depends.optJSONObject(0);
                    if (firstDependency != null) {
                        metadata.put("mcversion", firstDependency.optString("version", "Unknown"));
                    }
                }
                metadata.put("url", modMetadata.optString("contact", "Unknown"));
            }
            // Check for mods.toml
            else if (jarFile.getEntry("META-INF/mods.toml") != null) {
                metadataFound = true;
                InputStream infoStream = jarFile.getInputStream(jarFile.getEntry("META-INF/mods.toml"));
                BufferedReader reader = new BufferedReader(new InputStreamReader(infoStream));
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("modid")) {
                        metadata.put("mod_id", line.split("=")[1].trim());
                    } else if (line.startsWith("version")) {
                        metadata.put("version", line.split("=")[1].trim());
                    } else if (line.startsWith("displayName")) {
                        metadata.put("name", line.split("=")[1].trim());
                    } else if (line.startsWith("displayURL")) {
                        metadata.put("url", line.split("=")[1].trim());
                    }
                }
                System.out.println("mods.toml content processed");
            }
            // Check for META-INF/MANIFEST.MF
            else if (jarFile.getEntry("META-INF/MANIFEST.MF") != null) {
                metadataFound = true;
                InputStream infoStream = jarFile.getInputStream(jarFile.getEntry("META-INF/MANIFEST.MF"));
                Properties props = new Properties();
                props.load(infoStream);
                metadata.put("name", props.getProperty("Implementation-Title", "Unknown"));
                metadata.put("version", props.getProperty("Implementation-Version", "Unknown"));
                metadata.put("mod_id", props.getProperty("Specification-Title", "Unknown"));
                metadata.put("mcversion", props.getProperty("Specification-Version", "Unknown"));
                metadata.put("url", props.getProperty("Implementation-Vendor-URL", "Unknown"));
                System.out.println("MANIFEST.MF content processed");
            }

            if (!metadataFound) {
                System.out.println("No known metadata files found in " + jarPath);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            jarFile.close();
        }
        return metadata;
    }

    public static JSONArray scanFolder(String folderPath) throws IOException, JSONException {
        JSONArray results = new JSONArray();
        try (Stream<Path> paths = Files.walk(Paths.get(folderPath))) {
            paths.filter(Files::isRegularFile)
                 .filter(path -> path.toString().endsWith(".jar"))
                 .forEach(path -> {
                     try {
                         JSONObject metadata = extractMetadataFromJar(path.toString());
                         if (metadata.has("mod_id") && !metadata.getString("mod_id").equals("Unknown")) {
                             metadata.put("file", path.toString());
                             results.put(metadata);
                         } else {
                             System.out.println("No valid metadata found for JAR: " + path.toString());
                         }
                     } catch (Exception e) {
                         e.printStackTrace();
                     }
                 });
        }
        return results;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java ModScanner <path_to_folder>");
            System.exit(1);
        }
        String folderPath = args[0];
        JSONArray results = scanFolder(folderPath);
        // Write results to a file for Python to read
        try (PrintWriter out = new PrintWriter("scan_results.json")) {
            out.println(results.toString());
        }
    }
}
"""

# Function to create ModScanner.java and compile it
def create_and_compile_java():
    java_file_path = os.path.join(documents_folder, "ModScanner.java")
    class_output_directory = documents_folder

    with open(java_file_path, "w") as java_file:
        java_file.write(java_code)

    javassist_jar = os.path.join(documents_folder, "javassist.jar")
    json_jar = os.path.join(documents_folder, "json.jar")

    compile_command = f"javac -cp .;{javassist_jar};{json_jar} -d {class_output_directory} {java_file_path}"
    subprocess.run(compile_command, shell=True, check=True)

# Function to run the compiled Java program
def scan_folder_with_java(folder_path, javassist_jar_path):
    java_classpath = f".;{javassist_jar_path};{documents_folder}/json.jar;{documents_folder}"
    run_command = ["java", "-cp", java_classpath, "ModScanner", folder_path]
    result = subprocess.run(run_command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Java process failed: {result.stderr}")

    results_file_path = os.path.join(os.getcwd(), "scan_results.json")
    with open(results_file_path, 'r') as results_file:
        json_output = results_file.read().strip()
        try:
            return json.loads(json_output)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return []

def scan_mod_task(folder_path, javassist_jar_path):
    try:
        results = scan_folder_with_java(folder_path, javassist_jar_path)
        return results
    except Exception as e:
        logging.error(f"Error in scan_mod_task: {e}")
        return []

def check_incompatibilities(mods):
    incompatibilities = {}
    mixin_conflicts = {}
    class_conflicts = {}

    for mod_info in mods:
        mod_id = mod_info.get('mod_id')
        if mod_id:
            for other_mod_info in mods:
                other_mod_id = other_mod_info.get('mod_id')
                if mod_id != other_mod_id:
                    if mod_info.get('mod_id') == other_mod_info.get('mod_id'):
                        incompatibilities.setdefault(mod_id, []).append(other_mod_id)

                    classes1 = mod_info.get('classes', [])
                    classes2 = other_mod_info.get('classes', [])
                    for cls in classes1:
                        if cls in classes2:
                            if compare_class_details(mod_info, other_mod_info, cls):
                                class_conflicts.setdefault(mod_id, []).append(other_mod_id)

                    bc1 = mod_info.get('bytecode', {})
                    bc2 = other_mod_info.get('bytecode', {})
                    if bc1 and bc2:
                        for file_name, bytecode1 in bc1.items():
                            bytecode2 = bc2.get(file_name)
                            if bytecode2 and compare_bytecode(bytecode1, bytecode2):
                                mixin_conflicts.setdefault(mod_id, []).append(other_mod_id)

    class_conflicts = {k: list(set(v)) for k, v in class_conflicts.items()}
    mixin_conflicts = {k: list(set(v)) for k, v in mixin_conflicts.items()}

    return incompatibilities, class_conflicts, mixin_conflicts

def compare_class_details(mod_info1, mod_info2, class_name):
    methods1 = set(mod_info1['methods'][class_name])
    methods2 = set(mod_info2['methods'][class_name])
    fields1 = set(mod_info1['fields'][class_name])
    fields2 = set(mod_info2['fields'][class_name])
    cfg1 = mod_info1['cfg'][class_name]
    cfg2 = mod_info2['cfg'][class_name]

    return methods1 == methods2 and fields1 == fields2 and cfg1 == cfg2

def compare_bytecode(bc1, bc2):
    try:
        decoded_bc1 = base64.b64decode(bc1.encode('utf-8'))
        decoded_bc2 = base64.b64decode(bc2.encode('utf-8'))
        if compare_control_flow(decoded_bc1, decoded_bc2):
            return True
    except Exception as e:
        logging.error(f"Error comparing bytecode: {e}")
    return False

def compare_control_flow(cfg1, cfg2):
    return cfg1 == cfg2

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

def write_conflicts_to_text_file(file_path, incompatibilities, class_conflicts, mixin_conflicts):
    with open(file_path, 'w') as file:
        file.write("Mod Conflict Summary:\n\n")

        if incompatibilities:
            file.write("Incompatibilities:\n")
            for mod, conflicts in incompatibilities.items():
                if conflicts:
                    file.write(f"{mod} conflicts with {', '.join(conflicts)}\n")

        if class_conflicts:
            file.write("\nClass Conflicts:\n")
            for mod, conflicts in class_conflicts.items():
                if conflicts:
                    file.write(f"{mod} conflicts with {', '.join(conflicts)}\n")

        if mixin_conflicts:
            file.write("\nMixin Conflicts:\n")
            for mod, conflicts in mixin_conflicts.items():
                if conflicts:
                    file.write(f"{mod} conflicts with {', '.join(conflicts)}\n")

class ModIncompatibilityGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mod Incompatibility Manager")
        self.geometry("400x300")

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

        self.cancel_button = tk.Button(self, text="Cancel", command=self.cancel_action, bg='red', fg='white')
        self.cancel_button.pack(pady=5)

        self.choose_mc_dir_button = tk.Button(self, text="Choose Minecraft Directory", command=self.choose_mc_directory)
        self.choose_mc_dir_button.pack(pady=5)

        self.choose_javassist_dir_button = tk.Button(self, text="Choose Javassist Directory",
                                                     command=self.choose_javassist_directory)
        self.choose_javassist_dir_button.pack(pady=5)

        self.mc_directory = None
        self.javassist_jar_path = None
        self.mods = {}
        self.queue = Queue()

        create_and_compile_java()

    def add_mod(self):
        mod_path = filedialog.askopenfilename(title="Select Mod File", filetypes=[("JAR Files", "*.jar")])
        if mod_path:
            threading.Thread(target=self.run_scan_mod, args=(mod_path,)).start()

    def run_scan_mod(self, mod_path):
        result = scan_mod_task(mod_path, self.javassist_jar_path)
        if result:
            self.queue.put(result)
        self.after(100, self.check_queue)

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
        if not self.mc_directory:
            messagebox.showerror("Error", "Minecraft directory not selected.")
            return
        if not self.javassist_jar_path:
            messagebox.showerror("Error", "Javassist JAR file not selected.")
            return

        self.mods = {}
        thread = threading.Thread(target=self.run_scan_mods)
        thread.start()
        self.after(100, self.check_queue)

    def run_scan_mods(self):
        results = scan_mod_task(self.mc_directory, self.javassist_jar_path)
        for result in results:
            if 'mod_id' in result:
                self.mods[result['mod_id']] = result
            else:
                logging.error(f"Mod metadata missing 'mod_id': {result}")
        self.after(100, self.check_queue)

    def check_queue(self):
        try:
            while True:
                result = self.queue.get_nowait()
                if 'mod_id' in result:
                    self.mods[result['mod_id']] = result
                else:
                    logging.error(f"Mod metadata missing 'mod_id': {result}")
        except Empty:
            pass

        if threading.active_count() > 1:
            self.after(100, self.check_queue)
        else:
            self.check_results()

    def check_results(self):
        incompatibilities, class_conflicts, mixin_conflicts = check_incompatibilities(list(self.mods.values()))
        save_incompatibilities_to_file(json_file_path)
        write_conflicts_to_text_file(conflicts_txt_file_path, incompatibilities, class_conflicts, mixin_conflicts)
        messagebox.showinfo("Scan Complete", "Mod scan complete. Conflicts have been written to mod_conflicts.txt.")

    def cancel_action(self):
        self.destroy()

    def choose_mc_directory(self):
        self.mc_directory = filedialog.askdirectory(title="Select Minecraft Mod Directory")
        if self.mc_directory:
            messagebox.showinfo("Selected Directory", f"Minecraft Mod Directory: {self.mc_directory}")

    def choose_javassist_directory(self):
        self.javassist_jar_path = filedialog.askopenfilename(title="Select Javassist JAR File",
                                                             filetypes=[("JAR Files", "*.jar")])
        if self.javassist_jar_path:
            messagebox.showinfo("Selected JAR", f"Javassist JAR Path: {self.javassist_jar_path}")

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

            mod_info = scan_mod_task(temp_dir, documents_folder + '/javassist.jar')
            self.assertIsInstance(mod_info, list)
            self.assertGreater(len(mod_info), 0)
            self.assertIn('mod_id', mod_info[0])

    def test_check_incompatibilities(self):
        mod_info1 = {
            'mod_id': 'mod1',
            'classes': ['file1.class'],
            'methods': {'file1.class': ['public void method1()']},
            'fields': {'file1.class': ['private int field1']},
            'bytecode': {'file1.class': base64.b64encode(b'\x00\x01\x02').decode('utf-8')},
            'cfg': {'file1.class': {0: [1], 1: [2], 2: []}}
        }
        mod_info2 = {
            'mod_id': 'mod2',
            'classes': ['file1.class'],
            'methods': {'file1.class': ['public void method1()']},
            'fields': {'file1.class': ['private int field1']},
            'bytecode': {'file1.class': base64.b64encode(b'\x00\x01\x02').decode('utf-8')},
            'cfg': {'file1.class': {0: [1], 1: [2], 2: []}}
        }
        mods = [mod_info1, mod_info2]
        incompatibilities, class_conflicts, mixin_conflicts = check_incompatibilities(mods)
        self.assertIn('mod1', mixin_conflicts)
        self.assertIn('mod2', mixin_conflicts['mod1'])
        self.assertIn('mod1', class_conflicts)
        self.assertIn('mod2', class_conflicts['mod1'])

    def test_choose_mc_directory(self):
        gui = ModIncompatibilityGUI()
        with mock.patch('tkinter.filedialog.askdirectory', return_value='/fake/minecraft/directory'):
            gui.choose_mc_directory()
            self.assertEqual(gui.mc_directory, '/fake/minecraft/directory')

    def test_choose_javassist_directory(self):
        gui = ModIncompatibilityGUI()
        with mock.patch('tkinter.filedialog.askopenfilename', return_value='/fake/javassist.jar'):
            gui.choose_javassist_directory()
            self.assertEqual(gui.javassist_jar_path, '/fake/javassist.jar')

def background_save():
    while True:
        save_incompatibilities_to_file(json_file_path)
        time.sleep(300)

def run_tests():
    unittest.main(exit=False)

def run_gui():
    app = ModIncompatibilityGUI()
    app.mainloop()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        threading.Thread(target=background_save, daemon=True).start()
        run_gui()
