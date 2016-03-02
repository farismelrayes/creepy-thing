#!usr/bin/python
# Filename: duplicatefinder.py

"""
Duplicate image finder
Finds duplicate images and stores them in a subfolder for sorting/deletion
"""

from PIL import Image
from tqdm import tqdm
import glob
import os
import shutil

# Creates a hash of the image file
def dhash(image, hash_size = 8):
    # Shrink the image
    image = image.convert('L').resize((hash_size+1, hash_size), Image.ANTIALIAS)

    pixels = list(image.getdata())

    # Compare adjacent pixels
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            difference += [pixel_left > pixel_right]

    # Convert array to a hash
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0

    # Send back hash
    return ''.join(hex_string)

print("\nDUPLICATE IMAGE FINDER")

p = os.getcwd()+"/duplicates/"
createfolder = False
if not os.path.isdir(p):
    print("Creating duplicates folder...")
    os.mkdir(p)
    createfolder = True

print("Hashing files...")
hashes = {}

for filetype in ["*.jpg","*.jpeg","*.png"]:
    for infile in tqdm(glob.glob(filetype)):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        hs = dhash(im, 8)
        hashes[infile] = hs
        im.close()

print("Comparing hashes...")
samehash = []

for key1 in hashes:
    for key2 in hashes:
        if key1 != key2:
            if hashes[key1] == hashes[key2]:
                compare = [key1, key2]
                compare.sort()
                if compare not in samehash:
                    samehash += [compare]

print("\nRESULTS:")

for i in samehash:
    im1 = Image.open(i[0])
    im2 = Image.open(i[1])
    im1size = (im1.size[0]+im1.size[1])/2
    im2size = (im2.size[0]+im2.size[1])/2
    im1.close()
    im2.close()
    if (im1size > im2size):
        print(i[0] + " > " + i[1])
        print("Removing " + i[1])
        try:
            shutil.move(i[1],"duplicates/"+i[1])
        except: pass
    elif (im2size > im1size):
        print(i[0] + " < " + i[1])
        print("Removing " + i[0])
        try:
            shutil.move(i[0],"duplicates/"+i[0])
        except: pass
    else:
        print(i[0] + " = " + i[1])
        print("Removing " + i[1])
        try:
            shutil.move(i[0],"duplicates/"+i[1])
        except: pass

if len(samehash) == 0:
    print("No duplicates found.")
    if (createfolder == True):
        print("Removing duplicates folder...")
        try:
            os.rmdir(p)
        except Exception as ex:
            print(ex)
