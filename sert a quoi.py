from tkinter import *
from tkinter.filedialog import askopenfilename
import pert
import os


def create_gui():
    root = Tk()
    root.title('PERT & GANTT Chart Generator')
    root.geometry("400x300")

    # Style configuration
    root.configure(bg="#f0f0f0")

    # Frame for content
    main_frame = Frame(root, bg="#f0f0f0", padx=20, pady=20)
    main_frame.pack(expand=True, fill=BOTH)

    # Title label
    title_label = Label(
        main_frame,
        text="Project Management Tools",
        font=("Arial", 16, "bold"),
        bg="#f0f0f0"
    )
    title_label.pack(pady=(0, 20))

    # Instructions
    instructions = Label(
        main_frame,
        text="Import a CSV file with your project tasks.\nFormat: Task, Duration, Prerequisites",
        font=("Arial", 10),
        bg="#f0f0f0",
        justify=LEFT
    )
    instructions.pack(pady=(0, 15), anchor=W)

    # Status display
    status_frame = Frame(main_frame, bg="#f0f0f0")
    status_frame.pack(fill=X, pady=(0, 15))

    status_label = Label(
        status_frame,
        text="Status:",
        font=("Arial", 10),
        bg="#f0f0f0",
        anchor=W
    )
    status_label.pack(side=LEFT)

    status_value = Label(
        status_frame,
        text="Ready",
        font=("Arial", 10),
        bg="#f0f0f0",
        fg="blue",
        anchor=W
    )
    status_value.pack(side=LEFT, padx=(5, 0))

    # Import button
    import_button = Button(
        main_frame,
        text="Import CSV File",
        font=("Arial", 12),
        command=lambda: on_import(root, status_value),
        bg="#4CAF50",
        fg="white",
        padx=20,
        pady=10,
        relief=RAISED
    )
    import_button.pack(pady=(10, 0))

    # Exit button
    exit_button = Button(
        main_frame,
        text="Exit",
        font=("Arial", 12),
        command=root.destroy,
        bg="#f44336",
        fg="white",
        padx=20,
        pady=5
    )
    exit_button.pack(pady=(10, 0))

    # Set window close handler
    root.protocol('WM_DELETE_WINDOW', root.destroy)

    return root


def on_import(root, status_label):
    filename = askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not filename:
        return  # User cancelled

    # Update status
    status_label.config(text="Processing...", fg="orange")
    root.update()

    # Run PERT analysis
    status = pert.main(filename)

    if status == -1:
        status_label.config(text="Error processing file", fg="red")
    else:
        status_label.config(text="Success! Charts generated", fg="green")

        # Show info about generated files
        pert_path = os.path.abspath("pert.png")
        gantt_path = os.path.abspath("gantt.png")

        info_window = Toplevel(root)
        info_window.title("Charts Generated")
        info_window.geometry("400x150")

        info_text = f"PERT Chart: {pert_path}\nGantt Chart: {gantt_path}"

        Label(
            info_window,
            text="Charts successfully generated at:",
            font=("Arial", 12, "bold"),
            pady=10
        ).pack()

        Label(
            info_window,
            text=info_text,
            font=("Arial", 10),
            justify=LEFT,
            padx=20,
            pady=10
        ).pack()

        Button(
            info_window,
            text="OK",
            command=info_window.destroy,
            font=("Arial", 10),
            padx=20
        ).pack(pady=10)


if __name__ == "__main__":
    root = create_gui()
    root.mainloop()