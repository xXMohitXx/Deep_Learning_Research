### Install necessary dependencies
import torch
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
from PIL import Image
import streamlit as st

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

### Load ESA Hubble dataset from Kaggle
dataset = load_dataset("subhamshome/esa-hubble-images-3-classes", split="train")

# Create a folder to store training images
os.makedirs("hubble_images", exist_ok=True)

def save_dataset_images(dataset, save_path="hubble_images"):
    for i, data in enumerate(dataset):
        img = Image.open(data["image"]).convert("RGB")
        img.save(os.path.join(save_path, f"image_{i}.jpg"))
    print("Dataset images saved!")

save_dataset_images(dataset)

### Fine-tune Stable Diffusion with DreamBooth
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

class HubbleDataset(Dataset):
    def __init__(self, folder):
        self.image_paths = [os.path.join(folder, img) for img in os.listdir(folder)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize((512, 512))
        return torchvision.transforms.ToTensor()(image)

dataset = HubbleDataset("hubble_images")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define optimizer
optimizer = Adam(pipeline.unet.parameters(), lr=5e-6)

# Train loop (basic fine-tuning)
for epoch in range(5):
    for i, images in enumerate(dataloader):
        images = images.to(device)
        loss = pipeline.unet(images).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

# Save fine-tuned model
pipeline.save_pretrained("fine-tuned-hubble")

### Streamlit App for Image Generation
def generate_space_image(prompt):
    pipeline = StableDiffusionPipeline.from_pretrained("fine-tuned-hubble").to(device)
    image = pipeline(prompt).images[0]
    image.save("generated_image.png")
    return image

st.title("AI-Powered Space Image Generator")
st.write("Enter a space-related description, and AI will generate a unique image for you!")

user_prompt = st.text_input("Describe your space image (e.g., 'A blue nebula in deep space'):")
if st.button("Generate Image"):
    if user_prompt:
        generated_image = generate_space_image(user_prompt)
        st.image(generated_image, caption="Generated Image", use_column_width=True)
        with open("generated_image.png", "rb") as file:
            st.download_button("Download Image", file, "generated_image.png", "image/png")
    else:
        st.warning("Please enter a description to generate an image.")
