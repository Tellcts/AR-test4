from pathlib import Path

import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_hog_features(image):
    """提取单张图像的HOG特征"""
    # 确保图像是灰度格式
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # HOG特征提取参数设置
    hog = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    # 计算HOG特征
    features = hog.compute(gray.astype(np.uint8))
    return features.flatten()


def extract_sift_features(image):
    """提取单张图像的SIFT特征"""
    # 确保图像是灰度格式
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 创建SIFT对象并检测关键点和描述符
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 如果没有检测到关键点，则返回零向量
    if descriptors is None:
        return np.zeros(128)  # SIFT默认描述符长度为128

    # 聚合所有描述符（例如取平均值）
    # 对于CIFAR-10图像，我们可能需要固定数量的特征
    # 这里简单地取前几个或平均所有描述符
    if descriptors.shape[0] >= 1:
        # 平均所有描述符
        sift_features = np.mean(descriptors, axis=0)
    else:
        sift_features = np.zeros(128)

    return sift_features


def load_cifar10_data():
    """加载CIFAR-10数据集"""

    data_root = Path(__file__).parent.parent / "data"
    dataset = data_root / "cifar-10-batches-py"
    need_to_download = not dataset.exists()

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=need_to_download, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=need_to_download, transform=transform
    )

    # 转换为numpy数组
    train_images = []
    train_labels = []
    for img, label in trainset:
        img_np = (
            np.array(img.permute(1, 2, 0)) * 255
        )  # 转换为HxWxC格式并恢复到0-255范围
        img_np = img_np.astype(np.uint8)
        train_images.append(img_np)
        train_labels.append(label)

    test_images = []
    test_labels = []
    for img, label in testset:
        img_np = np.array(img.permute(1, 2, 0)) * 255
        img_np = img_np.astype(np.uint8)
        test_images.append(img_np)
        test_labels.append(label)

    return (
        np.array(train_images),
        np.array(train_labels),
        np.array(test_images),
        np.array(test_labels),
    )


def extract_features_for_dataset(images, feature_type="hog"):
    """为整个数据集提取特征"""
    features = []
    for i, img in enumerate(images):
        if feature_type.lower() == "hog":
            feat = extract_hog_features(img)
        elif feature_type.lower() == "sift":
            feat = extract_sift_features(img)
        else:
            raise ValueError("feature_type must be 'hog' or 'sift'")

        features.append(feat)

        if (i + 1) % 5000 == 0:
            print(
                f"Processed {i + 1}/{len(images)} images for {feature_type.upper()} features"
            )

    return np.array(features)


def cifar10_main():
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10_data()

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 特征提取类型
    feature_types = ["hog", "sift"]

    for feat_type in feature_types:
        print(f"\n--- Processing {feat_type.upper()} features ---")

        # 提取训练和测试特征
        print(f"Extracting {feat_type.upper()} features for training set...")
        X_train_features = extract_features_for_dataset(X_train, feat_type)

        print(f"Extracting {feat_type.upper()} features for test set...")
        X_test_features = extract_features_for_dataset(X_test, feat_type)

        # 标准化特征
        scaler = StandardScaler()
        X_train_features_scaled = scaler.fit_transform(X_train_features)
        X_test_features_scaled = scaler.transform(X_test_features)

        # 使用PCA降维以加快SVM训练速度
        pca = PCA(n_components=0.95)  # 保留95%的方差
        X_train_features_pca = pca.fit_transform(X_train_features_scaled)
        X_test_features_pca = pca.transform(X_test_features_scaled)

        print(
            f"After PCA, feature dimensions reduced to: {X_train_features_pca.shape[1]}"
        )

        # 训练SVM分类器
        print("Training SVM classifier...")
        svm_classifier = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        svm_classifier.fit(X_train_features_pca, y_train)

        # 预测和评估
        print("Making predictions...")
        y_pred = svm_classifier.predict(X_test_features_pca)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{feat_type.upper()} + SVM Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    cifar10_main()
