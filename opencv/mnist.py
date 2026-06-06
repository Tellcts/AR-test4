from pathlib import Path

import cv2
import numpy as np

# 从本地数据集中加载MNIST
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import datasets


def load_mnist_data():
    """加载MNIST数据集"""
    # path = Path("./data/MNIST/raw")
    data_root = Path(__file__).parent.parent / "data"
    dataset_path = data_root / "MNIST" / "raw"
    need_to_download = not dataset_path.exists()

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=need_to_download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=need_to_download, transform=transform
    )

    # 转换为numpy格式
    train_images = []
    train_labels = []
    for img, label in train_dataset:
        train_images.append(img.numpy().squeeze())  # 移除通道维度
        train_labels.append(label)

    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img.numpy().squeeze())
        test_labels.append(label)

    return (
        np.array(train_images),
        np.array(train_labels),
        np.array(test_images),
        np.array(test_labels),
    )


def extract_hog_features(images):
    """提取HOG特征"""
    # 设置HOG参数
    win_size = (28, 28)
    block_size = (14, 14)
    block_stride = (7, 7)
    cell_size = (7, 7)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1.0
    histogram_norm_type = 0
    l2_hys_threshold = 2.0000000000000001e-01
    gamma_correction = 0
    nlevels = 64
    signed_gradients = True

    hog = cv2.HOGDescriptor(
        win_size,
        block_size,
        block_stride,
        cell_size,
        nbins,
        deriv_aperture,
        win_sigma,
        histogram_norm_type,
        l2_hys_threshold,
        gamma_correction,
        nlevels,
        signed_gradients,
    )

    features = []
    for i, img in enumerate(images):
        # 确保图像是正确的大小和类型
        img_resized = cv2.resize(img.astype(np.float32), (28, 28))
        # 将图像归一化到0-255范围并转换为uint8类型
        img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_normalized.astype(np.uint8)

        # 提取HOG特征
        feature = hog.compute(img_uint8)
        features.append(feature.flatten())

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(images)} images for HOG features")

    return np.array(features)


def extract_sift_features(images):
    """提取SIFT特征"""
    sift = cv2.SIFT_create()
    features = []

    for i, img in enumerate(images):
        # 确保图像是正确的类型，归一化到0-255范围并转换为uint8
        img_normalized = cv2.normalize(
            img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
        )
        img_uint8 = img_normalized.astype(np.uint8)

        # 检测关键点和描述符
        kp, desc = sift.detectAndCompute(img_uint8, None)

        if desc is not None:
            # 如果有多个关键点，取平均值或前几个关键点
            if desc.shape[0] >= 128:
                # 取前128个描述符的平均值
                desc_subset = desc[:128]
                feature = np.mean(desc_subset, axis=0)
            else:
                # 如果关键点不足，使用所有可用的关键点并平均
                feature = np.mean(desc, axis=0)
        else:
            # 如果没有检测到关键点，返回零向量（长度为128，因为SIFT描述符是128维）
            feature = np.zeros(128)

        features.append(feature)

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(images)} images for SIFT features")

    return np.array(features)


def evaluate_feature_method(X_train, y_train, X_test, y_test, method_name):
    """评估特定特征提取方法的性能"""
    print(f"正在使用{method_name}特征训练SVM...")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 为了减少计算时间，我们可以使用部分数据训练SVM
    sample_size = min(10000, len(X_train))  # 限制训练样本数量
    indices = np.random.choice(len(X_train), size=sample_size, replace=False)

    X_train_sample = X_train_scaled[indices]
    y_train_sample = y_train[indices]

    # 训练SVM分类器
    svm_classifier = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_classifier.fit(X_train_sample, y_train_sample)

    # 在测试集上预测
    y_pred = svm_classifier.predict(X_test_scaled)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{method_name}特征的SVM分类准确率: {accuracy * 100:.2f}%")

    return accuracy, svm_classifier, scaler


def mnist_main():
    print("加载MNIST数据集...")
    X_train, y_train, X_test, y_test = load_mnist_data()

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")

    # 提取HOG特征
    print("\n提取HOG特征...")
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    print(f"HOG特征维度 - 训练集: {X_train_hog.shape}, 测试集: {X_test_hog.shape}")

    # 评估HOG特征
    hog_accuracy, hog_svm, hog_scaler = evaluate_feature_method(
        X_train_hog, y_train, X_test_hog, y_test, "HOG"
    )

    # 提取SIFT特征
    print("\n提取SIFT特征...")
    X_train_sift = extract_sift_features(X_train)
    X_test_sift = extract_sift_features(X_test)

    print(f"SIFT特征维度 - 训练集: {X_train_sift.shape}, 测试集: {X_test_sift.shape}")

    # 评估SIFT特征
    sift_accuracy, sift_svm, sift_scaler = evaluate_feature_method(
        X_train_sift, y_train, X_test_sift, y_test, "SIFT"
    )

    # 总结结果
    print("\n" + "=" * 50)
    print("MNIST数据集分类结果总结:")
    print(f"HOG特征 + SVM 准确率: {hog_accuracy * 100:.2f}%")
    print(f"SIFT特征 + SVM 准确率: {sift_accuracy * 100:.2f}%")

    if hog_accuracy > sift_accuracy:
        print("HOG特征在MNIST数据集上表现更好")
    elif sift_accuracy > hog_accuracy:
        print("SIFT特征在MNIST数据集上表现更好")
    else:
        print("两种特征方法表现相当")


if __name__ == "__main__":
    mnist_main()
