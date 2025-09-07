# 1) Fewer samples per client and fewer batches
sed -i 's/num_samples=1000/num_samples=200/' fedgen/runner.py
sed -i 's/batches >= 100/batches >= 10/' fedgen/runner.py

# 2) Lighter configs across the board
for f in fedgen/configs/*.yaml; do
  sed -i 's/^rounds: .*/rounds: 2/' "$f"
  sed -i 's/^clients: .*/clients: 4/' "$f"
  sed -i 's/^clients_per_round: .*/clients_per_round: 2/' "$f"
  sed -i 's/^batch_size: .*/batch_size: 32/' "$f"
done

# 3) Sanity check
python3 -m py_compile fedgen/runner.py

