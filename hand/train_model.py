import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Đọc dữ liệu từ file CSV
data = pd.read_csv('hand_face_landmarks.csv', header=None, on_bad_lines='skip')

# 2. Tách nhãn và đặc trưng
#data = data.dropna()
X = data.iloc[:, 1:].values  # Tọa độ landmark
y = data.iloc[:, 0].values   # Nhãn (label)

# 3. Encode nhãn nếu là chuỗi
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Chuẩn hóa dữ liệu đặc trưng
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 6. Huấn luyện mô hình SVM
clf = SVC(kernel='rbf', C=1.0, random_state=42)
clf.fit(X_train, y_train)

# 7. Đánh giá mô hình
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Lưu mô hình, encoder và scaler
joblib.dump(clf, 'gesture_model_svm.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
