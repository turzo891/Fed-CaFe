python3 run.py











Scaffold
python3 run.py

Use the generated script (not your root one)
Switch it to python3 and make it executable:
sed -i 's/\bpython\b/python3/g' fedgen/run_matrix.sh
chmod +x fedgen/run_matrix.sh


Deps
pip3 install pyyaml

Run matrix
./fedgen/run_matrix.sh

Inspect outputs
ls runs | head
column -t -s, results/summary.csv | head


still want a single-job run:
python3 fedgen/runner.py --config fedgen/configs/dcgan64_noniid_heavy.yaml --algo fedcafe --seed 0



2nd step 
# from repo root
python3 run.py
sed -i 's/\bpython\b/python3/g' fedgen/run_matrix.sh
chmod +x fedgen/run_matrix.sh
pip3 install torch torchvision numpy pyyaml
./fedgen/run_matrix.sh

