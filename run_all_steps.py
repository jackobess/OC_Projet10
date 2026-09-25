# run_all_steps.py         => python run_all_steps.py [--data-url URL]

import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--data-url", default=None)
url = parser.parse_args().data_url

subprocess.run(["python", "step1_prepare_data.py"] + (["--data-url", url] if url else []), check=True)
subprocess.run(["python", "step2_indexer.py"], check=True)
subprocess.run(["python", "step3_excel_to_db.py"], check=True)
