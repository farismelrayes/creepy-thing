import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
from time import sleep

def update_screen():
    ImageGrab.grab(bbox=(1198, 163, 1682, 1024)).save('test.png')

    img_rgb = cv2.imread('test.png')
    return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


def mouse_to_image(gray, temp, xoff=0, yoff=0):
    #w, h = temp.shape[::-1]
    res = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        pyautogui.moveTo(game_x + pt[0] + xoff, game_y + pt[1] + yoff)
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

if __name__ == '__main__':

    game_x = 1198
    game_y = 163

    img_gray = update_screen()
    mouse_to_image(img_gray, cv2.imread('playnow.png', 0))
    pyautogui.click()

    sleep(.2)

    img_gray = update_screen()
    mouse_to_image(img_gray, cv2.imread('play.png', 0), 32, 32)
    pyautogui.click()

    #cv2.imwrite('res.png', img_rgb)