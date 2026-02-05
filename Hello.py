import tkinter as tk

# Class 1: Student

class Student:
    def __init__(self, name, student_id):
        self.__name = name          # Encapsulation
        self.__student_id = student_id

    def get_details(self):
        return f"Name: {self.__name}, ID: {self.__student_id}"


# Class 2: StudentManager

class StudentManager:
    def __init__(self):
        self.__students = []        # Encapsulation

    def add_student(self, student):
        self.__students.append(student)

    def get_students(self):
        return self.__students


# Class 3: StudentApp (GUI)

class StudentApp:
    def __init__(self):
        self.manager = StudentManager()   # Class Interaction

        self.window = tk.Tk()
        self.window.title("Student Management System")
        self.window.geometry("400x300")

        tk.Label(self.window, text="Student Name").pack()
        self.name_entry = tk.Entry(self.window)
        self.name_entry.pack()

        tk.Label(self.window, text="Student ID").pack()
        self.id_entry = tk.Entry(self.window)
        self.id_entry.pack()

        tk.Button(self.window, text="Add Student", command=self.add_student).pack()

        self.listbox = tk.Listbox(self.window, width=50)
        self.listbox.pack()

        self.window.mainloop()

    def add_student(self):
        name = self.name_entry.get()
        student_id = self.id_entry.get()

        student = Student(name, student_id)    # Using Student class
        self.manager.add_student(student)      # Using StudentManager class

        self.update_list()

        self.name_entry.delete(0, tk.END)
        self.id_entry.delete(0, tk.END)

    def update_list(self):
        self.listbox.delete(0, tk.END)
        for student in self.manager.get_students():
            self.listbox.insert(tk.END, student.get_details())


StudentApp()
