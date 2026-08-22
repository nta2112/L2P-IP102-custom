import json

# Build notebook step by step to avoid complex nesting issues
nb = {
    "cells": [],
    "metadata": {
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [{"sourceId": 19051537, "sourceType": "datasetVersion"}],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13"
        },
        "papermill": {
            "default_parameters": {},
            "duration": 65.963576,
            "end_time": "2026-08-21T04:11:35.7988+00:00",
            "environment_variables": {},
            "exception": True,
            "input_path": "__notebook__.ipynb",
            "output_path": "__notebook__.ipynb",
            "parameters": {},
            "start_time": "2026-08-21T04:10:29.835224+00:00",
            "version": "2.7.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": []
}

# Cell 0: Markdown
nb["cells"].append({
    "cell_type": "markdown",
    "id": "9aabdaf7",
    "metadata": {"papermill": {"duration": 0.002389, "end_time": "2026-08-21T04:10:32.305267+00:00", "exception": False, "start_time": "2026-08-21T04:10:32.302878+00:00", "status": "completed"}, "tags": []},
    "source": [
        "# L2P for IP102 (Learning to Prompt, CVPR2022)\n",
        "Incremental learning + retrieval (R@1/5/10, mAP) + open-world (AUROC, FPR95) + lifelong (plasticity/forgetting/overall) metrics on the IP102 pest dataset.\n",
        "Dataset: **1 Input duy nhat** (chua train.json/val.json/test.json + thu muc anh JPEGImages). Code duoc clone tu GitHub qua env IP102_CODE_REPO, neu khong co thi tim trong /kaggle/input."
    ]
})

# Cell 1: Code - Clone repo
cell1 = {
    "cell_type": "code",
    "execution_count": 1,
    "id": "c649511d",
    "metadata": {
        "execution": {"iopub.execute_input": "2026-08-21T04:10:32.310268Z", "iopub.status.busy": "2026-08-21T04:10:32.309526Z", "iopub.status.idle": "2026-08-21T04:10:32.885993Z", "shell.execute_reply": "2026-08-21T04:10:32.884993Z"},
        "papermill": {"duration": 0.580687, "end_time": "2026-08-21T04:10:32.887627+00:00", "exception": False, "start_time": "2026-08-21T04:10:32.30694+00:00", "status": "completed"},
        "tags": []
    },
    "outputs": [
        {"name": "stdout", "output_type": "stream", "text": ["git clone https://github.com/nta2112/L2P-IP102-custom\n"]},
        {"name": "stderr", "output_type": "stream", "text": ["Cloning into '/kaggle/working/L2P-for-IP102'...\n"]},
        {"name": "stdout", "output_type": "stream", "text": ["CODE_DIR = /kaggle/working/L2P-for-IP102\n"]}
    ],
    "source": [
        "# ==== 1. Lay code tu GitHub (env IP102_CODE_REPO) hoac /kaggle/input ====\n",
        "import os, sys, glob, subprocess\n",
        "\n",
        "CODE_DIR = None\n",
        "os.environ['IP102_CODE_REPO'] = \"https://github.com/nta2112/L2P-IP102-custom\"\n",
        "\n",
        "target = '/kaggle/working/L2P-for-IP102'\n",
        "if os.environ.get('IP102_CODE_REPO'):\n",
        "    repo = os.environ['IP102_CODE_REPO']\n",
        "    if os.path.isdir(target):\n",
        "        print('Pulling latest code...')\n",
        "        subprocess.run(['git', '-C', target, 'pull'], check=True)\n",
        "    else:\n",
        "        repo = os.environ['IP102_CODE_REPO']\n",
        "        print('git clone', repo)\n",
        "        subprocess.run(['git', 'clone', repo, target], check=True)\n",
        "    CODE_DIR = target\n",
        "else:\n",
        "    for base in ('/kaggle/input', '/kaggle/working'):\n",
        "        found = sorted(glob.glob(os.path.join(base, '**', 'main_ip102.py'), recursive=True))\n",
        "        if found:\n",
        "            CODE_DIR = os.path.dirname(found[0])\n",
        "            break\n",
        "    if not CODE_DIR:\n",
        "        raise RuntimeError('Khong tim thay code. Dat env IP102_CODE_REPO hoac day code vao /kaggle/input.')\n",
        "\n",
        "sys.path.insert(0, CODE_DIR)\n",
        "print('CODE_DIR =', CODE_DIR)"
    ]
})
# Add to cells list after creating
print("Cell 1 created")