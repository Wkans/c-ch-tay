import joblib

# Tải mô hình từ file gesture_model.pkl
model = joblib.load('gesture_model.pkl')

# Kiểm tra loại đối tượng
print("Loại đối tượng:", type(model))

# Kiểm tra các thuộc tính quan trọng của mô hình Random Forest
print("Số lượng cây quyết định (n_estimators):", model.n_estimators)
print("Số đặc trưng đầu vào (n_features_in_):", model.n_features_in_)
print("Các lớp nhãn (classes_):", model.classes_)
print("Tham số của mô hình:", model.get_params())