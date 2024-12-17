
# 📄 Script for Converting Magnetic Moments for VASP

**Author:** Israel C. Ribeiro  
**Inspired by:** João H. Mazo  

---

## 📋 Description  
Python script to extract magnetic moments from the final ionic step of **OUTCAR** and generate the **MAGMOM** format compatible with VASP.

---

## 🔧 Features  
- Automatic reading of the number of ions (**NIONS**) in the **OUTCAR**.  
- Extraction of the final magnetic moments.  
- Generation of the **MAGMOM** file in the correct format.

---

## 🚀 How to Use  
1. Save the script as `magmom_input.py`.  
2. Place the `OUTCAR` file in the same directory.  
3. Run in the terminal:  
   ```bash
   python3 magmom_input.py
   ```
4. The `MAGMOM` file will be generated.

---

## 📧 Contact  
For questions or suggestions: [israelribeiroc7@gmail.com](mailto:israelribeiroc7@gmail.com).

---
