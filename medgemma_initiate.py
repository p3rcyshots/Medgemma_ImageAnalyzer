import argparse
import os
import sys
import signal
import time
from io import BytesIO
import warnings  # Import the warnings library

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForImageTextToText
import pygame
from tqdm import tqdm

# --- FEATURE: Suppress specific, harmless warnings from the transformers library ---
# This keeps the terminal output clean during generation.
warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id` to `eos_token_id`.*", category=UserWarning)

# --- ANSI escape codes for colors ---
BRIGHT_BLUE = "\033[94m"
RESET_COLOR = "\033[0m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"

# --- List of Medical Specialties ---
MEDICAL_SPECIALISTS = [
    "Allergist", "Anesthesiologist", "Cardiologist", "Dermatologist", 
    "Endocrinologist", "Gastroenterologist", "Gynecologist", "Hematologist", 
    "Infectious Disease Specialist", "Internist", "Nephrologist", "Neurologist", 
    "Obstetrician", "Oncologist", "Ophthalmologist", "Orthopedic Surgeon", 
    "Otolaryngologist (ENT Specialist)", "Pediatrician", "Plastic Surgeon", 
    "Psychiatrist", "Pulmonologist", "Radiologist", "Rheumatologist", 
    "Urologist", "Veterinarian"
]

# --- 1. Load the MedGemma Model and Processor ---
def load_medgemma():
    """Loads the model and processor and reports the device used."""
    print(f"{YELLOW}Loading MedGemma 4B Multimodal model...{RESET_COLOR}")
    print(f"{YELLOW}This may take a moment and requires ~8-9GB of VRAM...{RESET_COLOR}")
    try:
        model_id = "google/medgemma-4b-it"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"{GREEN}Model successfully loaded onto GPU: {gpu_name}{RESET_COLOR}")
        else:
            print(f"{YELLOW}Model loaded onto CPU. Analysis will be very slow.{RESET_COLOR}")

        return model, processor
    except Exception as e:
        print(f"{RED}FATAL ERROR during MedGemma model loading: {e}{RESET_COLOR}")
        exit()

# --- 2. The Analysis Functions ---
def perform_image_analysis(text_query, image_path):
    """Runs the analysis for a single image and returns the text."""
    try:
        pil_image = Image.open(image_path)
        messages = [{"role": "user", "content": [{"type": "text", "text": text_query}, {"type": "image", "image": pil_image}]}]
        inputs = medgemma_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(medgemma_model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = medgemma_model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.95)
            generation = generation[0][input_len:]
        
        return medgemma_processor.decode(generation, skip_special_tokens=True)
    except UnidentifiedImageError:
        return f"Error: Cannot identify image file '{os.path.basename(image_path)}'. It may be corrupted or in an unsupported format."
    except Exception as e:
        return f"An unexpected error occurred during image analysis: {str(e)}"

def perform_text_analysis(text_query):
    """Runs analysis for a general text-only query."""
    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": text_query}]}]
        inputs = medgemma_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(medgemma_model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = medgemma_model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.95)
            generation = generation[0][input_len:]
            
        return medgemma_processor.decode(generation, skip_special_tokens=True)
    except Exception as e:
        return f"An unexpected error occurred during text analysis: {str(e)}"

# --- 3. Main Application Logic ---
def main():
    def graceful_exit(sig, frame):
        print("\n\nCTRL+C detected. Shutting down.")
        try:
            pygame.mixer.Sound('stop.ogg').play()
            time.sleep(2)
        except Exception as e:
            print(f"{YELLOW}Could not play stop sound: {e}{RESET_COLOR}")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, graceful_exit)

    parser = argparse.ArgumentParser(
        description="Conversational Medical AI using MedGemma.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d", "--directory",
        default="Image",
        help="Path to the directory for image analysis.\nDefaults to a folder named 'Image' in the current directory."
    )
    args = parser.parse_args()
    image_directory = args.directory

    try:
        pygame.mixer.init()
        pygame.mixer.Sound('start.ogg').play()
    except Exception as e:
        print(f"{YELLOW}Could not play start sound: {e}{RESET_COLOR}")

    global medgemma_model, medgemma_processor
    medgemma_model, medgemma_processor = load_medgemma()
    
    print("\n" + "="*60)
    print("MedGemma Conversational AI - Interactive Specialist Mode")
    print("="*60)
    print("Ask a general medical question, or type 'analyze images' to start image analysis.")
    print(f"Image analysis will target the '{image_directory}' folder.")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        prompt = input(f"{GREEN}You: {RESET_COLOR}")

        if prompt.lower() in ['exit', 'quit']:
            graceful_exit(None, None)

        elif 'analyze images' in prompt.lower():
            print(f"MedGemma AI: {YELLOW}Please select an expert perspective for the analysis:{RESET_COLOR}")
            
            for i, specialist in enumerate(MEDICAL_SPECIALISTS):
                print(f"  {i+1}. {specialist}")
            
            try:
                choice_prompt = f"\nEnter the number of the desired specialist (1-{len(MEDICAL_SPECIALISTS)}): "
                choice_str = input(choice_prompt)
                choice_idx = int(choice_str) - 1

                if not 0 <= choice_idx < len(MEDICAL_SPECIALISTS):
                    raise ValueError("Selection out of range.")
                
                selected_specialist = MEDICAL_SPECIALISTS[choice_idx]
                print(f"MedGemma AI: {YELLOW}Analyzing images from the perspective of an expert {selected_specialist}...{RESET_COLOR}")

                if not os.path.isdir(image_directory):
                    raise FileNotFoundError(f"The specified directory does not exist: {image_directory}")

                supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
                image_files_to_analyze = [
                    os.path.join(image_directory, f) 
                    for f in os.listdir(image_directory) 
                    if f.lower().endswith(supported_extensions)
                ]

                if not image_files_to_analyze:
                    raise FileNotFoundError(f"No supported image files found in '{image_directory}'.")

                print(f"MedGemma AI: {YELLOW}Found {len(image_files_to_analyze)} images to analyze.{RESET_COLOR}")
                
                image_query = f"Analyze this medical image in detail as an expert {selected_specialist}."
                
                with tqdm(image_files_to_analyze, desc="Overall Progress", unit="image", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                    for file_path in pbar:
                        file_name = os.path.basename(file_path)
                        
                        pbar.set_description(f"Analyzing {file_name}")
                        
                        analysis_text = perform_image_analysis(image_query, file_path)
                        
                        # Use a newline before printing the analysis to avoid tqdm interference
                        pbar.write(f"\n--- Analysis by {selected_specialist} for: {file_name} ---\n")
                        pbar.write(f"{BRIGHT_BLUE}{analysis_text}{RESET_COLOR}\n")
                
                print(f"\n{GREEN}==================================={RESET_COLOR}")
                print(f"{GREEN}All image analyses complete.{RESET_COLOR}")
                print(f"{GREEN}==================================={RESET_COLOR}\n")

            except ValueError:
                print(f"MedGemma AI: {RED}Invalid input. Please enter a number between 1 and {len(MEDICAL_SPECIALISTS)}.{RESET_COLOR}")
            except FileNotFoundError as e:
                print(f"MedGemma AI: {RED}Error: {e}{RESET_COLOR}")
            except Exception as e:
                print(f"MedGemma AI: {RED}A critical error occurred during image analysis: {e}{RESET_COLOR}")
            
            print("Returning to conversational mode.\n")

        else:
            print(f"MedGemma AI: {YELLOW}Thinking...{RESET_COLOR}")
            final_text = perform_text_analysis(prompt)
            print(f"MedGemma AI: {BRIGHT_BLUE}{final_text}{RESET_COLOR}\n")

if __name__ == '__main__':
    main()