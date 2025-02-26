import gradio as gr
import requests
import cv2
import numpy as np
import os
from tempfile import NamedTemporaryFile

# ======================= 配置部分 =======================
# ----------------- 文本脱敏配置 -----------------
TEXT_API_URL = "http://localhost:8081/api/text"

# ----------------- 图像脱敏配置 -----------------
FACE_API_URL = "http://localhost:8000/api/v1/recognition/recognize"
FACE_API_KEY = "96f83d97-025f-4df4-bc5b-3881411f36ad"
TEMP_DIR = os.path.join(os.getcwd(), "temp_files")       # 临时文件目录
os.makedirs(TEMP_DIR, exist_ok=True)                     # 确保目录存在

# ====================== 文本处理模块 ======================
def text_deidentify(text):
    """调用文本脱敏服务"""
    try:
        response = requests.post(
            TEXT_API_URL,
            json={"text": text},
            timeout=10
        )
        return response.json().get("result", "处理失败") if response.ok else "服务响应异常"
    except Exception as e:
        return f"服务调用失败: {str(e)}"

# ====================== 图像处理模块 ======================
def safe_image_write(image, path):
    """安全写入图像文件"""
    try:
        ret, buf = cv2.imencode('.jpg', image)
        if ret:
            with open(path, 'wb') as f:
                f.write(buf.tobytes())
            return True
        return False
    except Exception as e:
        print(f"图像写入失败: {str(e)}")
        return False

def get_face_landmarks(image):
    """获取人脸关键点（安全版本）"""
    temp_path = os.path.join(TEMP_DIR, f"temp_{os.getpid()}.jpg")
    if not safe_image_write(image, temp_path):
        return []
    
    try:
        with open(temp_path, 'rb') as f:
            response = requests.post(
                FACE_API_URL,
                headers={"x-api-key": FACE_API_KEY},
                params={"face_plugins": "landmarks"},
                files={"file": f},
                timeout=15
            )
        return response.json().get('result', []) if response.ok else []
    except Exception as e:
        print(f"人脸检测API错误: {str(e)}")
        return []
    finally:
        try:
            os.remove(temp_path)
        except PermissionError:
            print(f"临时文件清理失败: {temp_path}")

def anonymize_image(image, blur_type, width_ratio, height_ratio, kernel_size, sigma):
    """图像脱敏核心逻辑"""
    # 参数验证
    kernel_size = max(3, int(kernel_size // 2 * 2 + 1))  # 确保奇数
    sigma = max(1, min(sigma, 100))
    
    detection_results = get_face_landmarks(image)
    if not detection_results:
        return image, "未检测到人脸"
    
    processed = image.copy()
    face_count = 0
    
    for face in detection_results:
        face_count += 1
        box = face.get("box", {})
        if not box:
            continue
        
        x_min = max(0, box.get("x_min", 0))
        y_min = max(0, box.get("y_min", 0))
        x_max = min(processed.shape[1], box.get("x_max", 0))
        y_max = min(processed.shape[0], box.get("y_max", 0))
        
        if blur_type == "face":
            # 全脸模糊
            face_roi = processed[y_min:y_max, x_min:x_max]
            h, w = face_roi.shape[:2]
            ksize = (min(w//2 * 2+1, 99), min(h//2 * 2+1, 99))
            processed[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(face_roi, ksize, sigma)
        elif blur_type == "eyes":
            # 眼部模糊
            landmarks = face.get("landmarks", [])
            if len(landmarks) >= 2:
                left_eye = np.array(landmarks[0])
                right_eye = np.array(landmarks[1])
                eye_distance = np.linalg.norm(right_eye - left_eye)
                
                for eye in [left_eye, right_eye]:
                    x1 = int(eye[0] - eye_distance * width_ratio / 2)
                    y1 = int(eye[1] - eye_distance * height_ratio / 2)
                    x2 = int(eye[0] + eye_distance * width_ratio / 2)
                    y2 = int(eye[1] + eye_distance * height_ratio / 2)
                    
                    # 边界检查
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(processed.shape[1], x2), min(processed.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        eye_roi = processed[y1:y2, x1:x2]
                        ksize = (min((x2-x1)//2 * 2+1,99), min((y2-y1)//2 * 2+1,99))
                        processed[y1:y2, x1:x2] = cv2.GaussianBlur(eye_roi, ksize, sigma)
    
    return processed, f"处理完成：检测到 {face_count} 张人脸"

# ====================== Gradio界面 ======================
with gr.Blocks(title="数据脱敏系统", css=".gradio-container {max-width: 1200px}") as app:
    gr.Markdown("# 🔐 隐私数据脱敏平台")
    
    with gr.Tabs():
        # ----------------- 文本脱敏标签页 -----------------
        with gr.TabItem("文本脱敏"):
            with gr.Row():
                text_input = gr.Textbox(
                    label="输入文本", 
                    lines=7, 
                    placeholder="请输入需要脱敏的文本..."
                )
                text_output = gr.Textbox(label="处理结果", lines=7, interactive=False)
            
            # 添加独立示例组件
            gr.Examples(
                examples=[
                    ["用户邮箱：test@example.com，电话：13800138000"],
                    ["地址：北京市朝阳区建国门外大街1号"],
                    ["身份证号码:330281201401239319,出生日期:2014-01-23,性别:男,年龄:11,出生地:浙江省 宁波市 余姚市"]
                ],
                inputs=text_input,
                label="点击示例快速输入",
                examples_per_page=3
            )
            
            text_btn = gr.Button("执行脱敏", variant="primary")

        # ----------------- 图像脱敏标签页 -----------------
        with gr.TabItem("图像脱敏"):
            gr.Markdown("### 人脸隐私保护")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="上传图片", type="numpy")
                    with gr.Accordion("高级参数", open=False):
                        with gr.Group():
                            blur_type = gr.Dropdown(
                                ["face", "eyes"],
                                value="eyes",
                                label="脱敏方式"
                            )
                            width_ratio = gr.Slider(
                                0.1, 2.0, 1.5,
                                step=0.1,
                                label="眼部区域宽度比例"
                            )
                            height_ratio = gr.Slider(
                                0.1, 2.0, 0.6,
                                step=0.1,
                                label="眼部区域高度比例"
                            )
                            kernel_size = gr.Slider(
                                3, 99, 35,
                                step=2,
                                label="模糊核大小（奇数）"
                            )
                            sigma = gr.Slider(
                                1, 100, 20,
                                label="模糊强度"
                            )
                    img_btn = gr.Button("开始处理", variant="primary")
                
                with gr.Column():
                    img_output = gr.Image(label="处理结果", interactive=False)
                    info_output = gr.Textbox(label="处理信息", interactive=False)
            
            gr.Examples(
                examples=[os.path.join("examples", f) for f in ["1.jpg", "2.jpg","3.jpg"]],
                inputs=img_input,
                label="🖼️ 示例图片"
            )

    # ==================== 事件绑定 ====================
    text_btn.click(
        fn=text_deidentify,
        inputs=text_input,
        outputs=text_output
    )
    
    img_btn.click(
        fn=anonymize_image,
        inputs=[img_input, blur_type, width_ratio, height_ratio, kernel_size, sigma],
        outputs=[img_output, info_output]
    )

# ==================== 启动应用 ====================
if __name__ == "__main__":
    # 权限检查
    if not os.access(TEMP_DIR, os.W_OK):
        raise PermissionError(f"临时目录不可写: {TEMP_DIR}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

def text_deidentify(text):
    """调用Go文本脱敏服务"""
    try:
        response = requests.post(
            TEXT_API_URL,
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["result"]
        return f"错误：服务返回状态码 {response.status_code}"
    except Exception as e:
        return f"API调用失败：{str(e)}"

def validate_kernel_size(ks):
    """确保核大小为正奇数"""
    return (max(1, ks[0] // 2 * 2 + 1), max(1, ks[1] // 2 * 2 + 1))

def get_face_landmarks(image):
    """获取人脸关键点"""
    with NamedTemporaryFile(suffix=".jpg") as temp:
        cv2.imwrite(temp.name, image)
        try:
            response = requests.post(
                FACE_API_URL,
                headers={"x-api-key": FACE_API_KEY},
                params={"face_plugins": "landmarks"},
                files={"file": open(temp.name, "rb")},
                timeout=15
            )
            return response.json().get('result', []) if response.ok else []
        except Exception as e:
            print(f"人脸检测失败: {str(e)}")
            return []

def image_anonymize(image, blur_type, width_ratio, height_ratio, kernel_size, sigma):
    """图像脱敏处理"""
    # 参数校验
    kernel_size = validate_kernel_size((kernel_size, kernel_size))
    sigma = max(1, min(sigma, 100))
    
    # 获取检测结果
    detection_results = get_face_landmarks(image)
    if not detection_results:
        return image, "未检测到人脸"
    
    processed = image.copy()
    faces_count = 0
    
    for face in detection_results:
        faces_count += 1
        box = face["box"]
        x_min, y_min = max(0, box["x_min"]), max(0, box["y_min"])
        x_max, y_max = min(image.shape[1], box["x_max"]), min(image.shape[0], box["y_max"])

        if blur_type == "face":
            # 全脸模糊
            face_roi = processed[y_min:y_max, x_min:x_max]
            h, w = face_roi.shape[:2]
            ksize = (min(w//2 * 2+1, 99), min(h//2 * 2+1, 99))
            processed[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(face_roi, ksize, sigma)
        elif blur_type == "eyes" and "landmarks" in face:
            # 眼部模糊
            landmarks = face["landmarks"]
            if len(landmarks) >= 2:
                left_eye = np.array(landmarks[0])
                right_eye = np.array(landmarks[1])
                eye_distance = np.linalg.norm(right_eye - left_eye)
                
                for eye in [left_eye, right_eye]:
                    x1 = int(eye[0] - eye_distance * width_ratio / 2)
                    y1 = int(eye[1] - eye_distance * height_ratio / 2)
                    x2 = int(eye[0] + eye_distance * width_ratio / 2)
                    y1, y2 = max(0,y1), min(processed.shape[0],y2)
                    x1, x2 = max(0,x1), min(processed.shape[1],x2)
                    
                    if x2 > x1 and y2 > y1:
                        eye_roi = processed[y1:y2, x1:x2]
                        ksize = (min((x2-x1)//2 * 2+1,99), min((y2-y1)//2 * 2+1,99))
                        processed[y1:y2, x1:x2] = cv2.GaussianBlur(eye_roi, ksize, sigma)
    
    return processed, f"处理完成：检测到 {faces_count} 张人脸"

# ----------------- Gradio界面 -----------------
with gr.Blocks(title="综合脱敏系统", css=".gradio-container {max-width: 1200px !important}") as app:
    gr.Markdown("# 🔒 敏感数据脱敏工作台")
    
    with gr.Tabs():
        with gr.TabItem("文本脱敏"):
            with gr.Row():
                text_input = gr.Textbox(label="输入文本", lines=7, placeholder="请输入需要脱敏的文本...")
                text_output = gr.Textbox(label="处理结果", lines=7, interactive=False)
            text_btn = gr.Button("执行脱敏", variant="primary")
            examples=[
                ["用户邮箱：test@example.com，电话：13800138000"],
                ["地址：北京市朝阳区建国门外大街1号"]
            ]
        
        with gr.TabItem("图像脱敏"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="上传图片", type="numpy")
                    with gr.Accordion("高级参数", open=False):
                        blur_type = gr.Dropdown(
                            ["face", "eyes"], value="eyes", label="脱敏方式",
                            info="选择全脸模糊或眼部模糊"
                        )
                        width_ratio = gr.Slider(0.1, 2.0, 0.6, step=0.1, label="眼部区域宽度比例")
                        height_ratio = gr.Slider(0.1, 2.0, 0.4, step=0.1, label="眼部区域高度比例")
                        kernel_size = gr.Slider(3, 99, 35, step=2, label="模糊核大小（奇数）")
                        sigma = gr.Slider(1, 100, 30, label="模糊强度")
                with gr.Column():
                    img_output = gr.Image(label="处理结果", interactive=False)
                    info_output = gr.Textbox(label="处理信息", interactive=False)
            img_btn = gr.Button("开始处理", variant="primary")

    # 事件处理
    text_btn.click(
        fn=text_deidentify,
        inputs=text_input,
        outputs=text_output
    )
    
    img_btn.click(
        fn=image_anonymize,
        inputs=[img_input, blur_type, width_ratio, height_ratio, kernel_size, sigma],
        outputs=[img_output, info_output]
    )

# ----------------- 启动应用 -----------------
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True
    )
