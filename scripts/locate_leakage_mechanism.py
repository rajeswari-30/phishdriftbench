import qrcode, numpy as np, random, string
from sklearn.ensemble import RandomForestClassifier

def mat(url, version=13):
    q = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=1, border=0)
    q.add_data(url); q.make(fit=False)
    return np.array(q.get_matrix(), dtype=np.uint8)

random.seed(2)
rnd=lambda n:"https://"+"".join(random.choice(string.ascii_lowercase) for _ in range(n))+".com"

X,y=[],[]
for _ in range(400):
    X.append(mat(rnd(random.randint(20,45))).ravel()); y.append(0)
    X.append(mat(rnd(random.randint(70,110))).ravel()); y.append(1)
X=np.array(X); y=np.array(y)

rf=RandomForestClassifier(n_estimators=300,random_state=0).fit(X,y)
imp=rf.feature_importances_.reshape(69,69)

# Where does the signal live? QR data is written bottom-right -> upward in 2-module columns.
print("Top-20 most important module coordinates (row, col):")
idx=np.dstack(np.unravel_index(np.argsort(imp.ravel())[::-1][:20],(69,69)))[0]
for r,c in idx: print(f"   row={r:2d} col={c:2d}  imp={imp[r,c]:.4f}")

print("\nImportance mass by region (data is placed from bottom-right upward):")
print(f"   bottom-right quadrant (r>34,c>34): {imp[35:,35:].sum():.3f}")
print(f"   top-left quadrant     (r<34,c<34): {imp[:34,:34].sum():.3f}")

# Direct check: does a *variable-version* dataset (version chosen by fit) destroy or amplify it?
def mat_fit(url):
    q=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L,box_size=1,border=0)
    q.add_data(url); q.make(fit=True)
    m=np.array(q.get_matrix(),dtype=np.uint8)
    out=np.zeros((69,69),dtype=np.uint8); out[:m.shape[0],:m.shape[1]]=m   # pad to common size
    return out.ravel()
X2=[];y2=[]
for _ in range(400):
    X2.append(mat_fit(rnd(random.randint(20,45)))); y2.append(0)
    X2.append(mat_fit(rnd(random.randint(70,110)))); y2.append(1)
from sklearn.model_selection import cross_val_score
print("\nAUC when QR version is allowed to vary with length (as in real-world scans):",
      round(cross_val_score(RandomForestClassifier(n_estimators=200,random_state=0),
                            np.array(X2),np.array(y2),cv=5,scoring="roc_auc").mean(),4))
