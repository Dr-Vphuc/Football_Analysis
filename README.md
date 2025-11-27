# Football Analysis- Computer Vision

Lưu ý: Chuẩn bị > 9 GB dung lượng cho việc tải tất cả thư viện trong requirement.txt và source code.  

Mã nguồn được triển khai trên Linux, nếu gặp lỗi liên quan đến sinh video khi chạy trên các hệ điều hành khác, rất có thể là do việc sinh file mp4 nhiều luồng 1 lúc, tìm và fĩx phần này trong file main.py. Gợi ý của tôi, có thể đổi định dạng video thành MJPG,... Nếu vẫn không được, tách từng luồng xuất file ra riêng.  

Tuy nhiên cân nhắc chỉnh sửa mã nguồn cho hợp lí vì lúc này việc tính toán phải thực hiện lại từ đầu với mỗi 1 loại video, dự đoán có thể đến tầm 4 tiếng với input video 30s (vì thời gian thực thi của code hiện tại tầm khoảng 1 tiếng cho video 30s). 

Hướng dẫn sửa : tách module ở phần vòng for dòng 124, sau khi xuất video xong phải release() writer lại kể kết thúc luồng của nó.

[![Demo Video](ss)](https://youtu.be/tcK_ONtxWRw)  
[![Demo Video](https://i.ytimg.com/an_webp/tcK_ONtxWRw/mqdefault_6s.webp?du=3000&sqp=CLy9ockG&rs=AOn4CLD0Y0gU8PLgHh6LlLi6niPmQQ6BWw)](https://youtu.be/tcK_ONtxWRw)


## Chú thích Module

Trìnhbay: chứa slides, báo cáo PDF của môn học  
input: video input cho demo của dự án  
output: các video output  
src: toàn bộ mã nguồn của dự án  
 - lib: module hỗ trợ chia team vào 2 đội (team.py), vẽ các bản đồ 2D sân bóng (annotators.py), thông số của sân bóng 2D (configs.py), 1 số hàm khác (func.py), chiếu homography (view.py)
 - main.py: phần thực hiện các logic chính của chương trình
