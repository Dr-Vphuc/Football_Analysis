# Football Analysis- Computer Vision

Lưu ý: Chuẩn bị > 9 GB dung lượng cho việc tải tất cả thư viện trong requirement.txt và source code.  

Mã nguồn được triển khai trên Linux, nếu gặp lỗi liên quan đến sinh video khi chạy trên các hệ điều hành khác, rất có thể là do việc sinh file mp4 nhiều luồng 1 lúc không thích ứng với hệ điều hành của bạn (thầy), khi đó thay vì chạy file main, hãy chạy các file tách từng phần riêng tôi (em) đã chuẩn bị sẵn gồm các file track.py, voronoi.py, blended_voronoi.py, ball_track.py, gameplay.py

Cân nhắc khi chạy file main : Hãy dừng chương trình (bằng Ctrl+C trên terminal) khi sinh được 4-5s sau đó kiểm tra vid có được ghi đúng như mong đợi không để tránh mất thời gian, nếu không hãy thực hiện theo hướng dẫn bên trên.

[![Demo Video](ss)](https://youtu.be/tcK_ONtxWRw)  
[![Demo Video](https://i.ytimg.com/an_webp/tcK_ONtxWRw/mqdefault_6s.webp?du=3000&sqp=CLy9ockG&rs=AOn4CLD0Y0gU8PLgHh6LlLi6niPmQQ6BWw)](https://youtu.be/tcK_ONtxWRw)


## Chú thích Module

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
