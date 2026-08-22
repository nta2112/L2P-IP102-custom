import json

with open(r'D:\Sau_Benh_object\retrieval-img\l2p\l2p-custom-ip102.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 1 source (index 1)
cell1_source = [
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
    "        # Da clone roi -> PULL code moi nhat\n",
    "        print('Pulling latest code...')\n",
    "        subprocess.run(['git', '-C', target, 'pull'], check=True)\n",
    "    else:\n",
    "        # Chua clone -> clone moi\n",
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

nb['cells'][1]['source'] = cell1_source

with open(r'D:\Sau_Benh_object\retrieval-img\l2p\l2p-custom-ip102.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Updated l2p-custom-ip102.ipynb cell 1')