# LAB 4: DECISION TREE & RANDOM FOREST

## 1. Giới thiệu dự án

**Họ và Tên:** Bùi Đức Gia Huy | **MSSV:** 24520650

Dự án này thực hiện cài đặt và đánh giá hai thuật toán học máy: **Decision Tree** và **Random Forest** trên tập dữ liệu **Wine Quality**. Mục tiêu cốt lõi là so sánh hiệu suất giữa việc tự hiện thực thuật toán bằng **NumPy** (From Scratch) và sử dụng thư viện chuyên dụng **Scikit-learn**.

Link dataset: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## 2. Cấu trúc thư mục

Dự án được tổ chức theo cấu trúc module như sau, nhằm tối ưu hóa việc tái sử dụng mã nguồn.

```text
DS102_LAB4/
├── data/                      # (Được bỏ qua trên Git)
│   ├── processed/             # Dữ liệu sau khi làm sạch và chuẩn hóa
│   └── raw/
│       └── wine+quality/      # Chứa dataset gốc (winequality-red.csv, winequality-white.csv)
├── src/                       # Các module xử lý logic cốt lõi
│   ├── data_loader.py         # Module đọc dữ liệu (dùng pathlib & pandas)
│   ├── data_preprocessing.py  # Tiền xử lý: Gộp data, Train/Test Split & Z-score Scaling
│   ├── DecisionTree.py        # Cài đặt class DecisionTree bằng NumPy
│   ├── RandomForest.py        # Cài đặt class RandomForest bằng NumPy
│   ├── evaluate.py            # Hàm tính toán metrics (Accuracy, Precision, Recall, F1)
│   ├── Assignment1.py         # Script chạy Assignment 1 (Decision Tree NumPy)
│   ├── Assignment2.py         # Script chạy Assignment 2 (Random Forest NumPy)
│   └── Assignment3.py         # Script chạy Assignment 3 (Scikit-learn Library)
├── .gitignore                 # Cấu hình loại trừ các file cache và dữ liệu nặng
├── assignments.ipynb          # Notebook thử nghiệm và báo cáo
├── requirements.txt           # Danh sách các thư viện phụ thuộc của dự án
└── README.md                  # Báo cáo tổng hợp dự án

```

---

## 3. Nội dung thực hiện

### Assignment 1: Decision Tree (NumPy)
* Hiện thực thuật toán **Decision Tree** bằng thuần NumPy.
* Sử dụng tiêu chí **Gini Impurity** để tính toán độ tinh khiết và tìm điểm phân hoạch tối ưu (Best Split).
* Xây dựng cây theo cơ chế đệ quy và dừng lại dựa trên điều kiện `max_depth` và `min_samples_split`.

### Assignment 2: Random Forest (NumPy)
* Hiện thực kỹ thuật học tập hợp **Bagging** (Bootstrap Aggregating).
* Mỗi cây được huấn luyện trên một tập mẫu **Bootstrap** và một tập con các đặc trưng ngẫu nhiên (**Feature Randomness** với tỷ lệ $\sqrt{\text{n\_features}}$).
* Kết hợp dự đoán từ 100 cây con thông qua cơ chế biểu quyết số đông (**Majority Voting**).

### Assignment 3: Machine Learning Library
* Sử dụng thư viện **Scikit-learn** để huấn luyện mô hình.
* Sử dụng `DecisionTreeClassifier` và `RandomForestClassifier` với các tham số tương đương để tạo cơ sở đối chiếu (Baseline) công bằng với phiên bản tự code.

---

## 4. Kết quả thực nghiệm

Kết quả dưới đây được ghi nhận sau khi thực hiện chia tập dữ liệu ngẫu nhiên (Test size = 20%) và chuẩn hóa Z-score độc lập trên từng đặc trưng:

| Assignment | Mô hình | Tham số cài đặt | F1-Score (Weighted) | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Assig 1** | DT (NumPy) | `max_depth: 10`, `min_samples_split: 2` | 0.5447 | 0.5519 |
| **Assig 2** | RF (NumPy) | `n_estimators: 100`, `max_depth: 10` | 0.5792 | 0.6050 |
| **Assig 3** | DT (Library) | `max_depth: 10`, `min_samples_split: 2` | 0.5447 | 0.5527 |
| **Assig 3** | RF (Library) | `n_estimators: 100`, `max_depth: 10` | 0.5958 | 0.6197 |

---

## 5. Phân tích & Nhận xét

### 5.1. So sánh Decision Tree và Random Forest
* Kết quả cho thấy **Random Forest** vượt trội hơn hẳn Decision Tree đơn lẻ trên cả hai phương diện tự code (NumPy) và dùng Thư viện. F1-score tăng từ **0.54** lên **0.57** (NumPy) và đạt gần **0.60** (Library).
* Việc kết hợp 100 cây độc lập giúp mô hình giảm **Variance** (phương sai) và hạn chế đáng kể hiện tượng Overfitting.

### 5.2. Hiệu quả của bản cài đặt NumPy vs Thư viện chuyên dụng
* Các mô hình từ **Scikit-learn** đạt hiệu suất cao hơn một chút (khoảng 1-2%). Điều này là nhờ các kỹ thuật tối ưu hóa thuật toán sâu, cơ chế xử lý giá trị ngưỡng (threshold) hiệu quả, và lõi tính toán bằng ngôn ngữ C.
* Tuy nhiên, phiên bản **tự code bằng NumPy** bám rất sát kết quả của thư viện, chứng minh logic hiện thực các thuật toán cốt lõi (Gini, Split, Bootstrapping, Voting) đã được xây dựng hoàn toàn chính xác và chuẩn mực.

### 5.3. Đặc thù của tập dữ liệu
* Tập dữ liệu Wine Quality có sự chênh lệch (imbalanced) rất lớn giữa 10 nhãn phân lớp (class 5, 6 chiếm đa số). Do đó, điểm số F1 dao động ở mức ~0.60 là một kết quả phản ánh thực tế độ khó của bài toán khi không áp dụng các kỹ thuật tái lấy mẫu (Resampling) như SMOTE.

> *Báo cáo được thực hiện cho bài Lab 4 môn DS102 - Học máy thống kê.*