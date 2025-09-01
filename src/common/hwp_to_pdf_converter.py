import os
import subprocess
import glob
import datetime

raw_docs_dir = os.path.join(os.getcwd() + "/backend" + "/raw_docs")
print(raw_docs_dir)
# want to find all subdirectories hwp file with glob
hwp_files = glob.glob(raw_docs_dir + "/**/*.hwp", recursive=True)
print(hwp_files)

error_list = []
now = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")

for hwp in hwp_files:
    try:
        output_file = hwp.replace(".hwp", ".html").replace("raw_docs", "docs")
        print("processing ", hwp.split("/")[-1], "->", output_file.split("/")[-1])
        result = subprocess.run(["hwp5html", hwp, "--output", output_file])
        if result.returncode != 0:
            print(f"Error while Processing {hwp}")
            error_list.append(hwp)
    except Exception:
        error_list.append(hwp)

with open(f"error_list_{now}.txt", "w") as f:
    f.write("\n".join(error_list))
