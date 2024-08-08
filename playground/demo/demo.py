
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import gradio as gr
from openai import OpenAI, AzureOpenAI
import pdb
# import time
import cv2

# import package to process pdf


# import copy
import torch
# import random
import numpy as np

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import TextStreamer
from llava.file_utils import (
    is_valid_video_filename, sample_frames, load_image, 
    convert_doc_to_images, extract_text_from_doc, 
    encode_image_gpt, process_video_gpt, extract_pdf_images_gpt,
    PDF_PROMPT)

class InferenceDemo(object):
    def __init__(self,args, model_path,tokenizer, model, image_processor, context_len, model_in_use) -> None:
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = tokenizer, model, image_processor, context_len

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        elif 'qwen' in model_name.lower():
            conv_mode = "qwen_1_5"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
        self.conv_mode = conv_mode
        self.model_in_use = model_in_use
        self.conversation = conv_templates[args.conv_mode].copy()
        self.gpt_conversation = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.last_history_point = 0
        self.num_frames = args.num_frames
        self.seconds_per_frame = args.seconds_per_frame


def clear_history(history):
    our_chatbot.conversation = conv_templates[our_chatbot.conv_mode].copy()
    our_chatbot.gpt_conversation = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    our_chatbot.last_history_point = 0
    return None

def change_model(model_name):
    our_chatbot.model_in_use = model_name
    print(f"Model changed to {model_name}")
    clear_history(None)
    print(f"Clear history")
    return None

def clear_response(history):
    for index_conv in range(1, len(history)):
        # loop until get a text response from our model.
        conv = history[-index_conv]
        if not (conv[0] is None):
            break
    question = history[-index_conv][0]
    history = history[:-index_conv]
    return history, question

def print_like_dislike(x):
    print(x.index, x.value, x.liked)



def add_message(history, message):
    # history=[]
    global our_chatbot
        
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    if our_chatbot.model_in_use == "LLaVA-Interleave-qwen-7B":          
        text=history[-1][0]
        images_this_term=[]
        text_this_term=''
        num_new_images = 0
        for i,message in enumerate(history[:-1]):
            if type(message[0]) is tuple:
                images_this_term.append(message[0][0])   
        # for message in history[-i-1:]:
        #     images_this_term.append(message[0][0])

        # assert len(images_this_term)>0, "must have an image"
        # image_files = (args.image_file).split(',')
        # image = [load_image(f) for f in images_this_term if f]
        image_list=[]
        pdf_text = []
        for f in images_this_term:
            if is_valid_video_filename(f):
                image_list+=sample_frames(f, our_chatbot.num_frames)
                num_new_images+=our_chatbot.num_frames
            elif f.endswith(".pdf"):
                pdf_images = convert_doc_to_images(f)
                image_list += pdf_images
                pdf_text.append(extract_text_from_doc(f))
                num_new_images += len(pdf_images)
            else:
                image_list.append(load_image(f))
                num_new_images+=1
        image_tensor = [our_chatbot.image_processor.preprocess(f, return_tensors="pt")["pixel_values"][0].half().to(our_chatbot.model.device) for f in image_list]

        if len(image_tensor) == 0:
            image_tensor = [torch.zeros(3, 384, 384).half().to(our_chatbot.model.device)]
            num_new_images = 1
        
        image_tensor = torch.stack(image_tensor)
        image_token = DEFAULT_IMAGE_TOKEN * num_new_images if num_new_images >= 1 else ""
        # if our_chatbot.model.config.mm_use_im_start_end:
        #     inp = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN + "\n" + inp
        # else:
        inp= text
        if len(pdf_text) > 0:
            inp = '\n'.join(pdf_text) + "\n" + PDF_PROMPT + "\n" + inp
        inp = image_token+ "\n" + inp
        our_chatbot.conversation.append_message(our_chatbot.conversation.roles[0], inp)
        # image = None
        our_chatbot.conversation.append_message(our_chatbot.conversation.roles[1], None)
        prompt = our_chatbot.conversation.get_prompt()

        input_ids = tokenizer_image_token(prompt, our_chatbot.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(our_chatbot.model.device)
        stop_str = our_chatbot.conversation.sep if our_chatbot.conversation.sep_style != SeparatorStyle.TWO else our_chatbot.conversation.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, our_chatbot.tokenizer, input_ids)
        streamer = TextStreamer(our_chatbot.tokenizer, skip_prompt=True, skip_special_tokens=True)
        # import pdb;pdb.set_trace()
        with torch.inference_mode():
            output_ids = our_chatbot.model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, max_new_tokens=1024, streamer=streamer, use_cache=False, stopping_criteria=[stopping_criteria])
        outputs = our_chatbot.tokenizer.decode(output_ids[0]).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        our_chatbot.conversation.messages[-1][-1] = outputs
    
        history[-1]=[text,outputs]
    elif our_chatbot.model_in_use.startswith("gpt"):
        text=history[-1][0]
        images_this_term=[]
        num_new_images = 0
        for i,message in enumerate(history[our_chatbot.last_history_point:-1]):
            if type(message[0]) is tuple:
                images_this_term.append(message[0][0])
        if client is None:
            print("Model not found")
        else:
            user_content = []  
            for f in images_this_term:
                if is_valid_video_filename(f):
                    base64Frames = process_video_gpt(f, seconds_per_frame=our_chatbot.seconds_per_frame)
                    user_content.append({"type": "text", "content": "These are the frames from the video."})
                    user_content.extend([*map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)])
                elif f.endswith(".pdf"):
                    pdf_images = extract_pdf_images_gpt(f)
                    pdf_text = extract_text_from_doc(f)
                    user_content.extend([*map(lambda x: {"type": "image_url", 
                                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, pdf_images)])
                    user_content.append({"type": "text", "content": pdf_text + "\n" + PDF_PROMPT})
                else:
                    user_content.append({"type": "image_url", "image_url":{"url":f'data:image/jpg;base64,{encode_image_gpt(f)}', "detail": "low"} })
            user_content.append({"type": "text", "content": text})
            our_chatbot.gpt_conversation.append({"role": "user", "content": user_content})
            
            response = client.chat.completions.create(
                    model=our_chatbot.model_in_use,
                    temperature=0,
                    messages=our_chatbot.gpt_conversation,
                    max_tokens=300,
                    top_p=0.1
            )
            our_chatbot.last_history_point = len(history)
            our_chatbot.gpt_conversation.append({"role": "assistant", "content": response.choices[0].message.content})
            history[-1]=[text,response.choices[0].message.content]
            print(history[-1][-1])
    else:
        print("Model not found")
    
    return history
txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter.",
    container=False,
)
with gr.Blocks() as demo:
    # Informations
    title_markdown = ("""
        # Assignment: AI Research Scientist
        [[Blog]](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)  [[Code]](https://github.com/LLaVA-VL/LLaVA-NeXT) [[Model]](https://huggingface.co/lmms-lab/llava-next-interleave-7b)
    """)
    tos_markdown = ("""
    ### TODO!. Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
    Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """)
    learn_more_markdown = ("""
    ### TODO!. License
    The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
    """)
    models = [
        "LLaVA-Interleave-qwen-7B",
        "gpt-4o"
    ]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gr.Markdown(title_markdown)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image", "video", "pdf"], placeholder="Enter message or upload file...", show_label=False)

    with gr.Row():
        model_select_dropdown = gr.Dropdown(models, value=models[0], label="Model", interactive=True)
        #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=True)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)
    clear_btn.click(fn=clear_history, inputs=[chatbot], outputs=[chatbot], api_name="clear_all")
    model_select_dropdown.change(change_model, inputs=[model_select_dropdown], outputs=[chatbot], api_name="change_model")
    with gr.Column():
        gr.Examples(examples=[
            [{"files": [f"{cur_dir}/examples/code1.jpeg",f"{cur_dir}/examples/code2.jpeg"], "text": "Please pay attention to the movement of the object from the first image to the second image, then write a HTML code to show this movement."}],
            [{"files": [f"{cur_dir}/examples/shub.jpg",f"{cur_dir}/examples/shuc.jpg",f"{cur_dir}/examples/shud.jpg"], "text": "what is fun about the images?"}],
            [{"files": [f"{cur_dir}/examples/tokyo_people.mp4"], "text": "Please describe the given video."}],
            [{"files": [f"{cur_dir}/examples/FYPPG_Contest2023_Rundown.pdf"], "text": "Please describe the content in this pdf."}],
            # [{"files": [f"playground/demo/examples/lion1_.mp4",f"playground/demo/examples/lion2_.mp4"], "text": "The input contains two videos, the first half is the first video and the second half is the second video. What is the difference between the two videos?"}],
            


            
        ], inputs=[chat_input], label="Compare images: ")

demo.queue()

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default="6123", type=str)
    argparser.add_argument("--model_path", default="", type=str)
    # argparser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    argparser.add_argument("--model-base", type=str, default=None)
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument("--conv-mode", type=str, default=None)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample from video. It is used for llava model only")
    argparser.add_argument("--seconds_per_frame", type=int, default=2, help="Number of seconds per frame for video processing. It is used for gpt model only")
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--gpt_api_key", type=str, default=None)
    argparser.add_argument("--gpt_api_base", type=str, default=None)
    
    args = argparser.parse_args()
    model_path = args.model_path
    filt_invalid="cut"
    model_name = get_model_name_from_path(args.model_path)
    model_in_use = "LLaVA-Interleave-qwen-7B"
    if model_path is None:
        model_in_use = "gpt-4o"
        tokenizer, model, image_processor, context_len = None, None, None, None
    else:
        model_in_use = "LLaVA-Interleave-qwen-7B"
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    
    if model_in_use == "gpt-4o" and args.gpt_api_key is None:
        raise ValueError("Please provide gpt api key when local model is not provided.")
    client = None
    if args.gpt_api_key is not None:
        print("Got api. Create OpenAI client...")
        if args.gpt_api_base is not None:
            if "azure" in args.gpt_api_base:
                client = AzureOpenAI(
                        api_key = args.gpt_api_key,
                        azure_endpoint = args.gpt_api_base,
                        api_version = "2024-06-01"
                    )
            else:
                raise ValueError("Other api base is not implemented.")
        else:
            client = OpenAI(api_key=args.gpt_api)
    our_chatbot = InferenceDemo(args, model_path, tokenizer, model, image_processor, context_len, model_in_use)
    # import pdb;pdb.set_trace()
    try:
        demo.launch(server_name=args.server_name, server_port=int(args.port),share=True)
    except Exception as e:
        args.port=int(args.port)+1
        print(f"Port {args.port} is occupied, try port {args.port}")
        demo.launch(server_name=args.server_name, server_port=int(args.port),share=True)
