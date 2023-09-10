import base64
import importlib
from io import BytesIO
from pathlib import Path
import PIL
import cv2
import numpy as np
from rich import print
import gradio as gr
from PIL.Image import Image

from modules.processing import StableDiffusionProcessingImg2Img

from modules import scripts, images, extensions


def tobase64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())


class ControlNetExt:
    """
    获取controlnet的扩展
    """

    def __init__(self) -> None:
        self.cn_base_path = ""
        for extension in extensions.active():
            if not extension.enabled:
                continue
            # For cases like sd-webui-controlnet-master
            if "sd-webui-controlnet" in extension.name:
                controlnet_exists = True
                controlnet_path = Path(extension.path)
                self.cn_base_path = ".".join(controlnet_path.parts[-2:])
                break

    def init_controlnet(self):
        import_path = self.cn_base_path + ".scripts.external_code"
        self.external_cn = importlib.import_module(import_path, "external_code")


class ZoomOutScript(scripts.Script):
    # 对图片进行放大扩充的脚本

    def __init__(self):
        super().__init__()
        self.cn = ControlNetExt()
        self.cn.init_controlnet()

    def title(self):
        return "Zoom Out"

    def show(self, is_img2img):
        if is_img2img:
            return scripts.AlwaysVisible
        else:
            return False

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False, elem_id="zo_main_accordion"):
            with gr.Row():
                with gr.Column(scale=6):
                    zo_enable = gr.Checkbox(
                        label="Enable ZoomOut",
                        value=False,
                        visible=True,
                        elem_id="zo_enable",
                    )

            with gr.Row():
                # 设置上下左右中五个方向的选项
                with gr.Column(scale=6):
                    zo_direction = gr.Radio(
                        label="Direction",
                        choices=[
                            "Center",
                            "Left",
                            "Right",
                            "Up",
                            "Down",
                        ],
                        value="Center",
                        inline=True,
                        visible=True,
                        elem_id="zo_direction",
                    )

            with gr.Row():
                with gr.Column(scale=6):
                    zo_scale = gr.Slider(
                        label="Scale",
                        minimum=1.0,
                        maximum=4.0,
                        step=0.1,
                        value=2.0,
                        visible=True,
                        elem_id="zo_scale",
                    )

        components = [zo_enable, zo_scale, zo_direction]

        return components

    def before_process(self, p: StableDiffusionProcessingImg2Img, *args):
        """
        预处理
        """
        zo_enable, zo_scale, zo_direction, *more = args

        if not zo_enable:
            return

        # 处理图片缩放
        resize_mode = p.resize_mode
        w, h = p.width, p.height
        img = p.init_images[0]
        img = images.resize_image(resize_mode, img, w, h)
        if zo_direction == "Center":
            img, mask, (w, h) = self.zoomout_upscaler_images(img, zo_scale)

        elif zo_direction == "Left":
            img, mask, (w, h) = self.zoomout_move(img, 0, zo_scale)

        elif zo_direction == "Right":
            img, mask, (w, h) = self.zoomout_move(img, 1, zo_scale)

        elif zo_direction == "Up":
            img, mask, (w, h) = self.zoomout_move(img, 2, zo_scale)

        elif zo_direction == "Down":
            img, mask, (w, h) = self.zoomout_move(img, 3, zo_scale)

        p.init_images[0] = img
        p.image_mask = mask
        p.width = w
        p.height = h

        # 处理图片蒙版属性
        p.mask_blur_x = 1
        p.mask_blur_y = 1
        p.inpaint_full_res_padding = 32  # 预留像素

        p.inpainting_mask_invert = 0  # 蒙版模式 0=重绘蒙版内容
        p.inpainting_fill = 0  # 蒙版蒙住的内容 0=填充
        p.inpaint_full_res = 1  # 重绘区域 0 = 全图 1=仅蒙版

        # 添加controlnet inpaint
        cn_units = []
        cn_inpaint = self.cn.external_cn.ControlNetUnit(
            model="control_v11p_sd15_inpaint",
            weight=1.0,
            pixel_perfect=True,
            module="inpaint_only+lama",
            control_mode=self.cn.external_cn.ControlMode.CONTROL,
            guidance_start=0.0,
            guidance_end=1.0,
            resize_mode=resize_mode,
        )
        cn_units.append(cn_inpaint)
        # cn_reference = self.cn.external_cn.ControlNetUnit(
        #     weight=1.0,
        #     pixel_perfect=True,
        #     module="reference_only",
        #     control_mode=self.cn.external_cn.ControlMode.BALANCED,
        #     guidance_start=0.0,
        #     guidance_end=1.0,
        #     threshold_a=0.5,
        #     resize_mode=resize_mode,
        #     processor_res=max(w, h),
        # )
        # cn_units.append(cn_reference)

        self.cn.external_cn.update_cn_script_in_processing(p, cn_units)

    def zoomout_upscaler_images(self, img: Image, scale=2.0):
        """
        对图片进行放大扩充

        """
        w, h = img.size

        background = (0, 0, 0)
        front = (255, 255, 255)

        # 创建一张相同大小的画布, 然后把原图按照比例缩小,放到画布中间
        canvas = PIL.Image.new("RGB", (w, h), (255, 255, 255))
        img = img.resize((int(w / scale), int(h / scale)))
        canvas.paste(img, (int((w - img.size[0]) / 2), int((h - img.size[1]) / 2)))

        # 制作一张蒙版,把放大的区域填充为黑色
        mask = PIL.Image.new("RGB", (w, h), background)
        # 计算上方和下方的黑色区域的高度
        top = int((h - img.size[1]) / 2)
        bottom = h - img.size[1] - top
        # 计算左右两边的黑色区域的宽度
        left = int((w - img.size[0]) / 2)
        right = w - img.size[0] - left
        # 把黑色区域填充到蒙版上
        mask.paste(front, (0, 0, w, top + 10))
        mask.paste(front, (0, h - bottom - 10, w, h))
        mask.paste(front, (0, 0, left + 10, h))
        mask.paste(front, (w - right - 10, 0, w, h))

        return canvas, mask, (w, h)

    def zoomout_move(self, img: Image, direction=0, crop=0.5):
        """
        对图片进行扩充

        direction: int 0=左 1=右 2=上 3=下
        crop: float 裁剪比例,0.5表示裁剪一半
        """

        w, h = img.size
        original_w, original_h = w, h

        background = (255, 255, 255)
        front = (0, 0, 0)

        # 裁剪图片
        if direction == 0:
            # 左移
            box = (int(w * (1 - crop)), 0, w, h)
            point = (0, 0)
        elif direction == 1:
            # 右移
            box = (0, 0, int(w * crop), h)
            point = (w - int(w * crop), 0)
        elif direction == 2:
            # 上移
            box = (0, int(h * crop), w, h)
            point = (0, 0)
        elif direction == 3:
            # 下移
            box = (0, 0, w, int(h * crop))
            point = (0, h - int(h * crop))

        # 创建一张画布,画布基于移动的方向,需要额外增加一些像素
        if direction == 0 or direction == 1:
            w = int(w * (1 + crop))
        elif direction == 2 or direction == 3:
            h = int(h * (1 + crop))
        canvas = PIL.Image.new("RGB", (w, h), (255, 255, 255))
        # 把裁剪后的图片贴到画布上
        img = img.crop(box)
        canvas.paste(img, point)

        # 制作一张蒙版
        background = (255, 255, 255, 255)
        front = (0, 0, 0, 100)
        mask = PIL.Image.new("RGBA", (w, h), background)

        # 计算蒙版的区域,并且向内缩小10个像素
        reserve = 10

        # TODO 只修改了left
        if direction == 0:
            mask_box = (0, 0, int(original_w * crop) - reserve, h)
        elif direction == 1:
            mask_box = (w - int(w * crop) + reserve, 0, w, h)
        elif direction == 2:
            mask_box = (0, 0, w, int(h * crop) - reserve)
        elif direction == 3:
            mask_box = (0, h - int(h * crop) + reserve, w, h)

        # 把黑色区域填充到蒙版上
        mask.paste(front, mask_box)


        return canvas, mask, (w, h)
