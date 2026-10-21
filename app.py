import gradio as gr
import torch
from diffusers import DiffusionPipeline
import imageio

# Load text-to-video model (first time takes 2-3 mins)
model_id = "damo-vilab/modelscope-text-to-video-ms-1.7b"

pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    variant="fp16"
  )
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def text_to_video(prompt):
    if not prompt.strip():
        return None
    # Generate 24 frames
    result = pipe(prompt, num_frames=24, num_inference_steps=25)
    frames = result.frames
    
    # Save video
    output_path = "output.mp4"
    imageio.mimsave(output_path, frames, fps=8)
    return output_path
  # Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Text to Video Generator")
    gr.Markdown("Type any text and get AI video")
    
    prompt = gr.Textbox(label="Enter your text", placeholder="e.g., a cat dancing in snow, mountains of Muzaffarabad")
    video = gr.Video(label="Generated Video")
    btn = gr.Button("Generate Video")
    
    btn.click(fn=text_to_video, inputs=prompt, outputs=video)

demo.launch(share=True)
