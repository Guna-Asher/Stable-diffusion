# Importing necessary libraries
import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
import os
import uuid

class ImageGeneratorApp:
    def __init__(self, model_path, device):
        # Initialize the tkinter app
        self.app = tk.Tk()
        self.app.geometry("532x632")
        self.app.title("Image Generator")
        ctk.set_appearance_mode("dark")

        # Load the model
        self.pipe = self.load_model(model_path, device)

        # Create the widgets
        self.create_widgets()

    def load_model(self, model_path, device):
        # Load the model and move it to the device
        pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp32", torch_dtype=torch.float32)
        return pipe.to(device)

    def create_widgets(self):
        # Create the input field
        self.prompt = ctk.CTkEntry(self.app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
        self.prompt.place(x=10, y=10)

        # Create the label for the image
        self.lmain = ctk.CTkLabel(self.app, height=512, width=512)
        self.lmain.place(x=10, y=110)

        # Create the generate button
        self.trigger = ctk.CTkButton(self.app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue",
                            command=self.generate)
        self.trigger.configure(text="Generate")
        self.trigger.place(x=206, y=60)

    def generate(self):
        with torch.no_grad():
            # Generate the image using the model
            result = self.pipe(self.prompt.get(), guidance_scale=8.5)

            # Process the result and update the image
            img = self.process_result(result)
            self.update_image(img)

            # Save the image
            self.save_image(img)

    def process_result(self, result):
        # Convert the 'images' list to a list of numpy arrays
        images_array = [np.array(img) for img in result['images']]

        # Convert the list of numpy arrays to a single numpy array
        images_tensor = torch.Tensor(np.array(images_array))

        # Convert the tensor back to a numpy array and normalize it
        img = images_tensor.cpu().numpy().squeeze() / np.max(images_tensor.cpu().numpy().squeeze())

        # Convert the numpy array to a PIL Image
        return Image.fromarray(np.uint8(img * 255), 'RGB')

    def update_image(self, img):
        # Convert the PIL Image to a Tkinter PhotoImage
        self.app.photo = ImageTk.PhotoImage(img)

        # Update the label with the new image
        self.lmain.configure(image=self.app.photo)

    def save_image(self, img):
        # Get the path to the Downloads directory
        downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')

        # Generate a unique filename using UUID
        unique_filename = str(uuid.uuid4())

        # Save the image to the file
        img.save(os.path.join(downloads_dir, unique_filename + '.png'))

    def run(self):
        # Start the tkinter event loop
        self.app.mainloop()

if __name__ == "__main__":
    # Path to the model and device to run the model on
    model_id = "C:/Users/scl/stable-diffusion-v1-5"
    device = "cpu"          # cpu or cuda

    # Create an instance of the ImageGeneratorApp class and run it
    app = ImageGeneratorApp(model_id, device)
    app.run()
