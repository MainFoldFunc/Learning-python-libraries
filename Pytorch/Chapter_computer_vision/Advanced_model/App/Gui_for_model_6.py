import tkinter as tk
import customtkinter as ctk

class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Model for MNIST visualization")
        self.geometry("1920x1080")

        # Top Label
        self.nameLabel = ctk.CTkLabel(
            self,
            text="You can draw here",
            font=("Arial", 32)  # Adjust font size here
        )
        self.nameLabel.pack(
            side="top",  # Place at the top
            pady=20      # Add vertical padding
        )

        # Drawing Canvas in the middle
        self.canvas = tk.Canvas(
            self,
            width=500, height=500,  # Size of the drawing area
            bg="gray",              # Background color set to gray
            cursor="cross"          # Cursor when hovering over canvas
        )
        self.canvas.place(
            relx=0.5,  # Center horizontally
            rely=0.5,  # Center vertically
            anchor="center"  # Anchor at the center
        )

        # Bind mouse events to enable drawing
        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        """Callback function to draw on the canvas."""
        x, y = event.x, event.y
        r = 3  # Radius of the circle
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")

def main():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("green")

    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()

