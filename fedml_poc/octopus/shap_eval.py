# shap_eval.py  (run from /home/spoof/projects/fedml_poc/octopus)
import os, numpy as np, torch, shap, matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from fed_gan_poc_5clients import Generator, SmallCls, NZ

OUT_DIR = "./out"
CKPT = os.path.join(OUT_DIR, "fed_gan_ckpt_5c.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_BG, BATCH_EVAL = 32, 64  # background/eval sizes

os.makedirs(OUT_DIR, exist_ok=True)
print(f"[INFO] device={DEVICE}, ckpt={CKPT}")

# --- load generator ---
ckpt = torch.load(CKPT, map_location=DEVICE)
G = Generator().to(DEVICE); G.load_state_dict(ckpt["G"]); G.eval()

# --- quick train classifier (1 epoch) ---
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
ldr = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
C = SmallCls().to(DEVICE)
opt = torch.optim.Adam(C.parameters(), lr=1e-3)
C.train()
for _ in range(1):
    for x, y in ldr:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = F.cross_entropy(C(x), y); loss.backward(); opt.step()
C.eval()
print("[INFO] classifier trained")

# --- generate samples to explain ---
with torch.no_grad():
    z = torch.randn(max(BATCH_BG, BATCH_EVAL), NZ, 1, 1, device=DEVICE)
    X = G(z).detach()  # [-1,1], shape [N,1,28,28]

X_bg = X[:BATCH_BG].to(DEVICE)
X_ev = X[:BATCH_EVAL].to(DEVICE)

# --- SHAP DeepExplainer (PyTorch path) ---
explainer = shap.DeepExplainer(C, X_bg)
sv = explainer.shap_values(X_ev, check_additivity=False)  # list of 10 numpy arrays [N,1,28,28]
print("[INFO] SHAP computed")

# --- helpers ---
def nchw_to_nhwc(a): return np.transpose(a, (0, 2, 3, 1))

# inputs for plotting (to [0,1])
X_np = X_ev.detach().cpu().numpy()              # [N,1,28,28]
X_disp = (nchw_to_nhwc(X_np) + 1.0) / 2.0       # [N,28,28,1]

# summary for class 0
SV0 = nchw_to_nhwc(sv[0])                       # numpy
n = min(SV0.shape[0], X_disp.shape[0])
shap.image_plot([SV0[:n]], X_disp[:n], show=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=200); plt.close()

# local for sample 0 using predicted class
with torch.no_grad():
    pred0 = C(X_ev[:1]).argmax(1).item()
SV_pred = nchw_to_nhwc(sv[pred0][:1])
X0 = X_disp[:1]
shap.image_plot([SV_pred], X0, show=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "shap_local_0.png"), dpi=200); plt.close()

print("[DONE]", os.path.join(OUT_DIR, "shap_summary.png"))
print("[DONE]", os.path.join(OUT_DIR, "shap_local_0.png"))
