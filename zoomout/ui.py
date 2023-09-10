import gradio as gr


def zoui(title, is_img2img):
    with gr.Accordion(title, open=False, elem_id="zo_main_accordion"):
        with gr.Row():
            with gr.Column(scale=6):
                zo_enable = gr.Checkbox(
                    label="Enable ZoomOut",
                    value=False,
                    visible=True,
                    elem_id="zo_enable",
                )

        with gr.Row():
            with gr.Column(scale=6):
                zo_scale = gr.Slider(
                    label="Scale",
                    min_value=1.0,
                    max_value=4.0,
                    step=0.1,
                    value=2.0,
                    visible=True,
                    elem_id="zo_scale",
                )
    components = [zo_enable, zo_scale]

    return components
