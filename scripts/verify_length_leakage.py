import qrcode, numpy as np, random, string
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

def matrix(url, version=13):
    q = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_L,
                      box_size=1, border=0)
    q.add_data(url); q.make(fit=False)
    return np.array(q.get_matrix(), dtype=np.uint8).ravel()

random.seed(1)
def rnd(n):
    return "https://" + "".join(random.choice(string.ascii_lowercase+string.digits+"-/.") for _ in range(n)) + ".com"

# Two classes that differ ONLY in URL length distribution (a pure artifact, no phishing semantics)
X, y = [], []
for _ in range(600):
    X.append(matrix(rnd(random.randint(20, 45))));  y.append(0)   # "benign"  = shorter
    X.append(matrix(rnd(random.randint(70, 110)))); y.append(1)   # "phishing"= longer
X = np.array(X); y = np.array(y)
print("data:", X.shape)

for name, clf in [("LogReg", LogisticRegression(max_iter=2000)),
                  ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=0))]:
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"  {name:13s} AUC = {auc.mean():.4f}  (+/- {auc.std():.4f})")
