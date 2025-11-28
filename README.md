# Football Analysis- Computer Vision


## Video demo
[![Demo Video](https://i.ytimg.com/an_webp/tcK_ONtxWRw/mqdefault_6s.webp?du=3000&sqp=CNC7pMkG&rs=AOn4CLBKllpMUgKUez4GyIC6G9ESsXe7KA)](https://youtu.be/tcK_ONtxWRw)

## Lưu ý quan trọng: 
### Chuẩn bị môi trường
Chuẩn bị > 9 GB dung lượng cho việc tải tất cả thư viện trong requirement.txt và source code.  
Nếu không muốn tải thư viện, bạn(thầy) có thể chạy thử dự án qua Google Colab (các file để trong thư mục Colab_Notebook). Tuy nhiên, các file notebook này không sinh ra video hoàn chỉnh, chỉ là bản test thuật toán của tôi (em/Phúc trong file football_ai.ipynb).  

Để chạy mã nguồn này, bạn(thầy) cần lập 1 tài khoản [Roboflow](https://roboflow.com/), lấy Private_API_KEY của tài khoản ([tại đây](https://app.roboflow.com/mycv-ybicf/settings/api)) và thay vào file .env (tự tạo) theo dạng 'ROBOFLOW_API_KEY=VcXIxxxxxxxjjqp' để pull mô hình nhóm đã 'deploy' lên Roboflow về. Đối với các file Football_object_detector.ipynb và train_pitch_keypoint_detector.ipynb thì lưu API_KEY vào phần Secrets, bật notebook access.

### Lỗi có thể gặp
Mã nguồn được triển khai trên Linux, nếu gặp lỗi liên quan đến sinh video khi chạy trên các hệ điều hành khác, rất có thể là do việc sinh file mp4 nhiều luồng 1 lúc không thích ứng với hệ điều hành của bạn (thầy), khi đó thay vì chạy file main, hãy chạy các file tách từng phần riêng tôi (em) đã chuẩn bị sẵn gồm các file track.py, voronoi.py, blended_voronoi.py, ball_track.py, gameplay.py

### Chuẩn bị thời gian
Tốc độ thực thi trên hđh Linux với CPU i5 13420H, GPU : Nvidia RTX 3060Ti 6GB và 16 GB RAM là 55 phút cho video 30s. Hãy cân nhắc dừng chương trình (bằng Ctrl+C trên terminal) khi sinh được 4-5s video sau đó kiểm tra vid có được ghi đúng như mong đợi không để tránh mất thời gian, nếu không hãy thực hiện theo hướng dẫn bên trên. Khi dừng sớm, ảnh tracking bóng sẽ không được tạo, bạn (thầy) có thể tạo bằng việc chạy file ball_track.py, tất nhiên file này cũng đang break sau 5 frame để tránh mất thời gian, bạn(thầy) có thế tìm đến dòng 77, 78 để xóa điều kiện dừng này.


## Chú thích Module

Colab_Notebook: Chứa mã nguồn dạng notebook, không sinh video, chỉ test thuật toán và kết quả triển khai mô hình.
Trìnhbay: chứa slides, báo cáo PDF của môn học  
input: video input cho demo của dự án  
output: các video output  
src: toàn bộ mã nguồn của dự án  
 - lib: module hỗ trợ chia team vào 2 đội (team.py), vẽ các bản đồ 2D sân bóng (annotators.py), thông số của sân bóng 2D (configs.py), 1 số hàm khác (func.py), chiếu homography (view.py)
 - main.py: phần thực hiện các logic chính của chương trình

## Hướng dẫn thực thi

- Cài đặt thư viện phụ thuộc : pip install -r requirements.txt
- Chạy phần chính của chương trình: python main.py (nếu dùng python3: python3 main.py)
- Nếu gặp lỗi khi chạy file main, có thể chạy riêng từng module: python3 track.py, python3 voronoi.py, python3 blended_voronoi.py, python3 ball_track.py, python3 gameplay.py.
