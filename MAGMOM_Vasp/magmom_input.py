# Script for transforming the last ionic step magnetic moments to the MAGMOM flag format for VASP
# Made by Israel C. Ribeiro inspire by Joao Mazo's bash script
# Exclusive use by QTNANO group and collaborators, please do not share without proper authorization!
# Special Thanks to Joao Mazo for helping with the initial steps
# If any bug or difficulty is found, please e-mail me: israelribeiroc7@gmail.com

import re

outcar_file = "OUTCAR"

def extract_nions(outcar_file):
    with open(outcar_file, 'r') as file:
        for line in file:
            if "NIONS" in line:
                return int(re.search(r'\d+', line[::-1]).group()[::-1])

def extract_magmoms(outcar_file, nions):
    magmom_list = []
    magnetization_block = False
    with open(outcar_file, 'r') as file:
        for line in file:
            if "magnetization (x)" in line:
                magnetization_block = True
            if magnetization_block:
                if len(magmom_list) < nions:
                    if re.match(r'\s*\d+', line):
                        magmom_list.append(line.split()[4])
                else:
                    break
    return magmom_list

def main():
    nions = extract_nions(outcar_file)
    magmoms = extract_magmoms(outcar_file, nions)

    # Escreve os momentos magnéticos no formato necessário para o arquivo MAGMOM
    with open("MAGMOM", 'w') as magmom_file:
        magmom_file.write(" ".join(magmoms) + "\n")

if __name__ == "__main__":
    main()

