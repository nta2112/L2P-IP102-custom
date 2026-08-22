import json

with open(r'D:\Sau_Benh_object\retrieval-img\l2p\l2p-custom-ip102.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Hotfix cell
hotfix_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "hotfix_cell",
    "metadata": {
        "execution": {},
        "papermill": {"duration": None, "end_time": None, "exception": None, "start_time": None, "status": "pending"},
        "tags": []
    },
    "outputs": [],
    "source": [
        "# ==== HOTFIX: Patch ip102_eval.py de fix KeyError Recall@1_unseen_energy ====\n",
        "import sys\n",
        "eval_path = '/kaggle/working/L2P-for-IP102/libml/ip102_eval.py'\n",
        "with open(eval_path, 'r') as f:\n",
        "    content = f.read()\n",
        "\n",
        "# Patch 1: Them init Recall@1_unseen_energy vao evaluate_task\n",
        "if \"res['Recall@1_unseen_energy'] = None\" not in content:\n",
        "    content = content.replace(\n",
        "        \"res['Recall@1_seen'] = rec_seen\",\n",
        "        \"res['Recall@1_seen'] = rec_seen\\n  res['Recall@1_unseen_energy'] = None  # Hotfix KeyError\"\n",
        "    )\n",
        "    print('Patch 1: Added Recall@1_unseen_energy init')\n",
        "\n",
        "# Patch 2: Dam bao write_results khong loi khi key missing\n",
        "eval_path = '/kaggle/working/L2P-for-IP102/libml/ip102_eval.py'\n",
        "with open(eval_path, 'r') as f:\n",
        "    content = f.read()\n",
        "\n",
        "if \"row.get(k, None)\" not in content:\n",
        "    content = content.replace(\n",
        "        \"writer.writerow({k: _fmt(row[k]) for k in RESULTS_HEADER})\",\n",
        "        \"writer.writerow({k: _fmt(row.get(k, None)) for k in RESULTS_HEADER})\"\n",
        "    )\n",
        "    with open('/kaggle/working/L2P-for-IP102/libml/ip102_eval.py', 'w') as f:\n",
        "        f.write(content)\n",
        "    print('Patch 2: write_results dung row.get()')\n",
        "\n",
        "# Verify patch\n",
        "with open('/kaggle/working/L2P-for-IP102/libml/ip102_eval.py', 'r') as f:\n",
        "    content = f.read()\n",
        "print('Recall@1_unseen_energy init:', \"res['Recall@1_unseen_energy'] = None\" in content)\n",
        "print('row.get() in write_results:', 'row.get(' in content)\n",
        "print('HOTFIX DONE - Chay cell 5 tiep tuc!')"
    ],
    "cell_type": "code",
    "execution_count": None,
    "id": "hotfix_cell",
    "metadata": {
        "execution": {},
        "papermill": {"duration": None, "end_time": None, "exception": None, "start_time": None, "status": "pending"},
        "tags": []
    },
    "outputs": []
}

# Load notebook
with open(r'D:\Sau_Benh_object\retrieval-img\l2p\l2p-custom-ip102.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert hotfix cell before the run_train cell (index 5)
nb['cells'].insert(5, hotfix_cell)

# Save
with open(r'D:\Sau_Benh_object\retrieval-img\l2p\l2p-custom-ip102.ipynb', 'w', encoding='utf-8', newline='') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook updated with hotfix cell')