import os

# Đường dẫn đến thư mục gốc chứa ảnh
image_root = r"D:\\Course\\3RD\\AI PROJECT\\dataset\\factory_dataset\images"

# Đếm số lượng ảnh .jpg
image_count = 0
for root, dirs, files in os.walk(image_root):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_count += 1

print(f"✅ Tổng số ảnh .jpg trong thư mục '{image_root}': {image_count}")
