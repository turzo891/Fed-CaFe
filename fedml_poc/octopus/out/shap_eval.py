# shap_eval.py
import torch, shap
from torchvision.utils import save_image
from fed_gan_poc_5clients import Generator, SmallCls, NZ
device="cuda" if torch.cuda.is_available() else "cpu"
ckpt=torch.load("/home/spoof/projects/fedml_poc/octopus/out/fed_gan_ckpt_5c.pt", map_location=device)
G=Generator().to(device); G.load_state_dict(ckpt["G"]); G.eval()
C=SmallCls().to(device); C.eval()  # retrain or load if you saved; else re-train once

# sample 200 images
with torch.no_grad():
    z=torch.randn(200,NZ,1,1,device=device); X=G(z).detach()  # [-1,1]
Xn=(X+1)/2*2-1  # keep normalized same as training

# SHAP GradientExplainer on logits for predicted class
def f(x): return C(x).detach()
background=Xn[:32].to(device)
explainer=shap.GradientExplainer(f, background)
shap_values=explainer.shap_values(Xn[:64].to(device))  # small batch for speed

# Save a summary plot for the predicted class (class 0 for example)
shap.image_plot(shap_values[0].detach().cpu().numpy(), Xn[:64].cpu().numpy(), show=False)
import matplotlib.pyplot as plt; plt.tight_layout(); plt.savefig("/home/spoof/projects/fedml_poc/octopus/out/shap_summary.jpg")
