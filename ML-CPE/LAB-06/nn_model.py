from sklearn.neural_network import MLPClassifier

def build_nn_model(hidden_layers=(16,), max_iter=100):
    """สร้าง Neural Network (MLP) สำหรับจำแนกข้อมูล"""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        activation='relu',
        solver='adam',
        random_state=42
    )
    return model