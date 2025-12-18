# BÁO CÁO LAB X – TÌM HIỂU TỔNG QUAN BÀI TOÁN TEXT-TO-SPEECH (TTS)

---

## 1. Bối cảnh và động cơ nghiên cứu

Trong bối cảnh công nghệ phát triển nhanh chóng, **khả năng tự học và tự nghiên cứu** trở thành kỹ năng cốt lõi đối với sinh viên ngành Công nghệ thông tin và Trí tuệ nhân tạo. Với sự hỗ trợ của Internet, các hệ thống tìm kiếm, công cụ AI/Agent và kho tài nguyên học thuật mở, việc tiếp cận tri thức chuyên sâu trở nên dễ dàng hơn bao giờ hết.

Trong nội dung học bổ sung của tuần 12, sinh viên được tiếp cận bài toán **Text-To-Speech (TTS)** – một bài toán nền tảng trong lĩnh vực xử lý tiếng nói (Speech Processing). Mục tiêu của bài tập này là đóng vai trò như một **nhà nghiên cứu**, tìm hiểu:
- Bức tranh tổng quan của bài toán TTS
- Các hướng nghiên cứu và triển khai hiện tại
- Ưu điểm, hạn chế và kịch bản sử dụng của từng hướng tiếp cận

---

## 2. Bức tranh toàn cảnh về Text-To-Speech

Bài toán **Text-To-Speech (TTS)** nhằm mục tiêu chuyển đổi văn bản đầu vào thành tín hiệu giọng nói có ngữ điệu tự nhiên, dễ nghe và giống con người. Quá trình phát triển của TTS có thể được chia thành **ba cấp độ (Level)** chính.

### 2.1. Level 1 – TTS dựa trên luật (Rule-based / Concatenative TTS)

**Mô tả:**  
Đây là thế hệ TTS đầu tiên, dựa trên các **luật ngôn ngữ học**, quy tắc âm tiết và từ điển phát âm. Âm thanh thường được ghép nối từ các đơn vị âm thanh có sẵn.

**Ưu điểm:**
- Tốc độ xử lý rất nhanh
- Tiêu tốn ít tài nguyên tính toán
- Dễ triển khai cho nhiều ngôn ngữ khác nhau

**Nhược điểm:**
- Giọng nói thiếu tự nhiên
- Ngữ điệu cứng nhắc, khó biểu đạt cảm xúc
- Khó mở rộng chất lượng âm thanh

**Phù hợp với:**
- Thiết bị nhúng, hệ thống tài nguyên thấp
- Ứng dụng yêu cầu tốc độ cao nhưng không cần tính tự nhiên cao

---

### 2.2. Level 2 – TTS dựa trên Deep Learning (Single-speaker / Fine-tuned TTS)

**Mô tả:**  
Các mô hình Deep Learning (Tacotron, FastSpeech, WaveNet, HiFi-GAN, …) được sử dụng để học ánh xạ trực tiếp từ văn bản sang đặc trưng âm thanh hoặc waveform. Nhiều nghiên cứu xây dựng **pipeline cá nhân hóa**, trong đó mỗi người dùng có thể ghi âm dữ liệu của chính mình để tinh chỉnh mô hình.

**Ưu điểm:**
- Giọng nói tự nhiên, mượt mà
- Có thể cá nhân hóa theo từng người dùng
- Chi phí suy luận (inference) thấp hơn so với Level 3

**Nhược điểm:**
- Yêu cầu dữ liệu ghi âm chất lượng cho mỗi giọng nói
- Khó mở rộng đa ngôn ngữ
- Cần công đoạn huấn luyện riêng cho từng người

**Phù hợp với:**
- Trợ lý ảo cá nhân
- Ứng dụng đọc sách, đọc báo
- Hệ thống TTS cá nhân hóa

---

### 2.3. Level 3 – Few-shot / Zero-shot TTS (Voice Cloning)

**Mô tả:**  
Các mô hình hiện đại cho phép tạo giọng nói mới chỉ với **vài giây âm thanh mẫu**. Mô hình học đặc trưng giọng nói (speaker embedding) và tổng hợp giọng nói mới mang đặc trưng đó.

**Ưu điểm:**
- Rất linh hoạt, không cần huấn luyện lại cho từng người
- Có thể mở rộng đa ngôn ngữ
- Trải nghiệm người dùng tốt (chỉ cần vài giây ghi âm)

**Nhược điểm:**
- Mô hình phức tạp
- Tiêu tốn nhiều tài nguyên tính toán
- Rủi ro cao về đạo đức (deepfake)

**Phù hợp với:**
- Sản phẩm thương mại quy mô lớn
- Nền tảng AI đa người dùng
- Ứng dụng sáng tạo nội dung

---

## 3. Các thách thức chung trong nghiên cứu TTS

Các nghiên cứu hiện đại về TTS tập trung giải quyết đồng thời nhiều mục tiêu:

- **Hiệu suất cao:** tốc độ suy luận nhanh như Level 1
- **Tiết kiệm tài nguyên:** phù hợp với thiết bị hạn chế phần cứng
- **Tính tự nhiên:** giọng nói gần giống con người
- **Đa ngôn ngữ:** hỗ trợ nhiều ngôn ngữ và phương ngữ
- **Biểu đạt cảm xúc:** kiểm soát cảm xúc, ngữ điệu
- **Giảm công sức người dùng:** ít hoặc không cần dữ liệu huấn luyện

---

## 4. Các pipeline nghiên cứu nhằm tối ưu ưu/nhược điểm

Nhiều hướng nghiên cứu đã được đề xuất để cân bằng giữa chất lượng và tài nguyên:

- **Tách pipeline:** Text → Phoneme → Spectrogram → Vocoder
- **Non-autoregressive models:** (FastSpeech) giúp tăng tốc inference
- **Knowledge distillation:** giảm kích thước mô hình
- **Speaker embedding + Adapter:** cá nhân hóa mà không cần huấn luyện lại toàn bộ
- **Multilingual pretraining:** huấn luyện trên nhiều ngôn ngữ để tăng khả năng tổng quát

Các pipeline này giúp tận dụng ưu điểm của từng level và giảm thiểu nhược điểm cố hữu.

---

## 5. Đạo đức nghiên cứu và vấn đề xã hội

Sự phát triển mạnh mẽ của TTS, đặc biệt là Level 3, đặt ra nhiều thách thức đạo đức:

- Nguy cơ **deepfake giọng nói**
- Phát tán thông tin sai lệch
- Xâm phạm quyền riêng tư

**Giải pháp được đề xuất:**
- Nhúng **watermark** vào âm thanh do AI tạo ra
- Cơ chế xác thực nguồn gốc âm thanh
- Quy định pháp lý và đạo đức trong nghiên cứu và triển khai

---

## 6. Kết luận

Qua bài nghiên cứu tổng quan này, có thể thấy rằng bài toán **Text-To-Speech** đã và đang phát triển mạnh mẽ qua nhiều thế hệ mô hình. Mỗi hướng tiếp cận (Level 1, 2, 3) đều có ưu và nhược điểm riêng, phù hợp với từng kịch bản ứng dụng và nguồn lực khác nhau.

Xu hướng hiện tại của nghiên cứu TTS là hướng tới các mô hình:
- Tự nhiên hơn
- Đa ngôn ngữ hơn
- Ít tốn tài nguyên hơn
- An toàn và có trách nhiệm hơn

Đây là một lĩnh vực giàu tiềm năng nghiên cứu và ứng dụng trong tương lai.
