import cv2
import numpy as np

# 读取图像
image_path = '/home/yanwenli/light-weight-refinenet/test/data/1/scene.npy'  # 更新为您的图像路径
data = np.load(image_path, allow_pickle=True).item()

# 假设图像数据存储在字典的 'image' 键中
image = data.get('rgb')

if image is None:
    print(f"Failed to load image {image_path}")
else:
    # 检查图像是否为三通道，如果不是则转换为三通道
    if len(image.shape) == 2:  # 如果是灰度图像
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:  # 如果是单通道图像
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edged = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Edged Image', edged)
    cv2.imshow('Image with Contours', image_with_contours)

    # 保存结果
    output_path = '/home/yanwenli/light-weight-refinenet/test/out_put_image/image.png'  # 更新为您的输出路径
    cv2.imwrite(output_path, image_with_contours)
    print(f"Segmented image saved as {output_path}")

    # 等待按键并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
