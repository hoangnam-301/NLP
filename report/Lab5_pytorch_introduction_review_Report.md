# BÁO CÁO THỰC HÀNH LAB 5: NHẬP MÔN PYTORCH

## 1. Giải thích các bước thực hiện

Quy trình thực hiện bài Lab được tiến hành qua **3 giai đoạn chính**:

### Giai đoạn 1: Làm chủ Tensor (Đơn vị dữ liệu cơ bản)
- Thực hiện chuyển đổi dữ liệu từ **List** và **NumPy** sang định dạng **Tensor** của PyTorch.
- Đây là bước quan trọng để dữ liệu có thể được tính toán hiệu quả, đặc biệt khi chạy trên **GPU**.
- Tiến hành các phép toán đại số tuyến tính cơ bản:
  - Nhân ma trận
  - Indexing (truy xuất phần tử)
  - Thay đổi hình dạng dữ liệu (**reshape**)
- Các thao tác này giúp dữ liệu phù hợp với kiến trúc của mạng nơ-ron.

### Giai đoạn 2: Cơ chế Tự động tính đạo hàm (Autograd)
- Sử dụng thuộc tính `requires_grad=True` để PyTorch theo dõi lịch sử các phép toán trên Tensor.
- Khi gọi hàm `.backward()`, PyTorch tự động xây dựng và duyệt **đồ thị tính toán** để tính Gradient.
- Cơ chế này giúp loại bỏ việc tính đạo hàm thủ công, rất quan trọng trong huấn luyện mạng nơ-ron sâu.

### Giai đoạn 3: Thiết kế Mô hình (Neural Network Construction)
- Sử dụng thư viện `torch.nn` để xây dựng mô hình học sâu.
- Kết hợp các thành phần:
  - **Embedding Layer**: Mã hóa từ ngữ thành vector số.
  - **Linear Layer**: Biến đổi tuyến tính.
  - **ReLU**: Hàm kích hoạt phi tuyến.
- Tất cả được đóng gói trong một lớp kế thừa từ `nn.Module` để tạo thành mô hình hoàn chỉnh.

---

## 2. Hướng dẫn chạy code

### 2.1. Môi trường
Cài đặt PyTorch bằng lệnh:
```bash
pip install torch
```

### 2.2. Thực thi
- Sao chép mã nguồn vào Jupyter Notebook hoặc file Python (`.py`).

### 2.3. Kiểm tra luồng dữ liệu
- **Phần 1:** Kiểm tra việc khởi tạo Tensor và các phép toán ma trận, quan sát kết quả nhân ma trận bằng toán tử `@`.
- **Phần 2:** Theo dõi giá trị Gradient được lưu trong thuộc tính `.grad`.
- **Phần 3:** Khởi tạo mô hình và truyền `LongTensor` (ID từ) qua mạng để nhận kết quả dự báo.

---

## 3. Phân tích kết quả thực hiện

### 3.1. Thao tác trên Tensor

Với dữ liệu:
\[
x = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\]

Phép nhân:
\[
x x^T = \begin{bmatrix} 5 & 11 \\ 11 & 25 \end{bmatrix}
\]

→ PyTorch thực hiện đúng các quy tắc đại số tuyến tính.

**Reshape:** Tensor `(4,4)` → `(16,1)` trong khi vẫn giữ nguyên 16 phần tử, phù hợp trước khi đưa vào Fully Connected Layer.

### 3.2. Tính toán Đạo hàm (Autograd)

Hàm số:
\[
z = 3y^2, \quad y = x + 2
\]

Với `x = 1`:
- `y = 3`
- `z = 27`
- `x.grad = 18.0`

Theo lý thuyết:
\[
\frac{dz}{dx} = 6y = 18
\]

→ Autograd hoạt động chính xác.

### 3.3. Kiến trúc Mô hình Neural Network

- **Embedding:** Đầu vào 4 từ → đầu ra `(4,3)`
- **Output:** `torch.Size([1, 4, 2])`
  - 1: batch size  
  - 4: số từ  
  - 2: số lớp phân loại  

Sự xuất hiện của `grad_fn` chứng tỏ mô hình sẵn sàng cho quá trình huấn luyện.

---

**Trả lời câu hỏi:**  
Khi gọi z.backward() lần thứ hai, chương trình sẽ phát sinh lỗi RuntimeError vì theo mặc định, PyTorch tự động giải phóng và xóa bỏ toàn bộ đồ thị tính toán ngay sau khi hoàn thành việc tính đạo hàm ở lần gọi đầu tiên nhằm tối ưu hóa bộ nhớ hệ thống. Do đồ thị các phép toán trung gian đã bị hủy, PyTorch không còn dữ liệu để thực hiện lại quy trình lan truyền ngược, trừ khi bạn chỉ định tham số retain_graph=True trong lần gọi đầu. Ngoài ra, nếu đồ thị được giữ lại để chạy lần hai, các giá trị đạo hàm mới sẽ bị cộng dồn vào kết quả của lần tính trước đó chứ không tự động ghi đè hay làm mới.
