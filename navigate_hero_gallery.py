import pyautogui
import pydirectinput
import pytesseract
import PIL.ImageOps
from PIL import Image, ImageChops
import time
import numpy as np
import pickle

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def click_hero_gallery():
    heroes = (100,750,470-100,900-750)
    '''
    while True:
        im = pyautogui.screenshot(region=heroes)
        heroes_string = pytesseract.image_to_string(add_margin(PIL.ImageOps.invert(im),400,400,400,400,"white"), config='--psm 4')
        print(heroes_string)
        if "heroes" in heroes_string.lower():
            pyautogui.click(heroes[0]+heroes[2]//2,heroes[1]+heroes[3]//2)
            pyautogui.moveTo(0,0)
            return
    '''
    time.sleep(5)
    pyautogui.click(heroes[0]+heroes[2]//2,heroes[1]+heroes[3]//2)
    pyautogui.moveTo(0,0)
tank = ["dva", "doomfist", "junkerqueen", "mauga", "orisa", "ramattra", "reinhardt", "roadhog", "sigma", "winston", "wreckingball", "zarya"]
damage = ["ashe", "bastion", "cassidy", "echo", "genji", "hanzo", "junkrat", "mei", "pharah", "reaper", "sojourn", "soldier76", "sombra", "symmetra", "torbjorn", "tracer", "venture", "widowmaker"]
support = ["ana", "baptiste", "brigitte", "illari", "kiriko", "lifeweaver", "lucio", "mercy", "moira", "zenyatta"]
'''
def click_tank(hero):
    im = pyautogui.screenshot()
    tank_start = (185,710)
    tank_shape = (4,3)
    name_box = (250,95)
    start = tank_start
    hero_index = tank.index(hero)
    index = 0
    for j in range(tank_shape[1]):
        for i in range(tank_shape[0]):
            x,y = start[0]+name_box[0]*i,start[1]+315*j
            im.crop((x,y,x+name_box[0],y+name_box[1])).show()
            if index == hero_index:
                pyautogui.click(x+name_box[0]//2,y+name_box[1]//2)
            hero_string = pytesseract.image_to_string(add_margin(im.crop((x,y,x+name_box[0],y+name_box[1])),400,400,400,400,"white"), config='--psm 4')
            print(hero_string)
            if len(hero_string):
                index += 1
'''
def click_tank(hero):
    im = pyautogui.screenshot()
    start = (185,710)
    shape = (4,3)
    name_box = (250,95)
    hero_index = tank.index(hero)
    i = hero_index % shape[0]
    j = hero_index // shape[0]
    x,y = start[0]+name_box[0]*i,start[1]+315*j
    pyautogui.moveTo(x+name_box[0]//2,y+name_box[1]//2)
    time.sleep(0.5)
    pyautogui.click()
def click_damage(hero):
    im = pyautogui.screenshot()
    start = (1295,710)
    shape = (5,4)
    name_box = (250,95)
    hero_index = damage.index(hero)
    i = hero_index % shape[0]
    j = hero_index // shape[0]
    if hero_index >= 15:
        i += 1
    x,y = start[0]+name_box[0]*i,start[1]+315*j
    pyautogui.moveTo(x+name_box[0]//2,y+name_box[1]//2)
    time.sleep(0.5)
    pyautogui.click()
def click_support(hero):
    im = pyautogui.screenshot()
    start = (2655,710)
    shape = (4,3)
    name_box = (250,95)
    hero_index = support.index(hero)
    i = hero_index % shape[0]
    j = hero_index // shape[0]
    x,y = start[0]+name_box[0]*i,start[1]+315*j
    pyautogui.moveTo(x+name_box[0]//2,y+name_box[1]//2)
    time.sleep(0.5)
    pyautogui.click()
def click_hero(hero):
    if hero in tank:
        click_tank(hero)
    elif hero in damage:
        click_damage(hero)
    elif hero in support:
        click_support(hero)
def click_skins():
    pyautogui.moveTo(605,605)
    time.sleep(1)
    pyautogui.keyDown('space')
    time.sleep(0.5)
    pyautogui.keyUp('space')
'''
def get_manhattan_hull(boxes):
    return (min(x[0] for x in boxes),min(x[1] for x in boxes),max(x[2] for x in boxes),max(x[3] for x in boxes))
def get_bounding_box(diff):
    visited = set()
    queue = [(diff.size[0]-16,diff.size[1]-16,diff.size[0]+16,diff.size[1]+16)]
    center_brightness = np.average(np.array(diff.crop(queue[0])))
    cache = dict()
    print(center_brightness)
    while len(queue):
        current = queue.pop()
        visited.add(current)
        nexts = []
        nexts.append((current[0]-1,current[1],current[2]-1,current[3]))
        nexts.append((current[0]+1,current[1],current[2]+1,current[3]))
        nexts.append((current[0],current[1]-1,current[2],current[3]-1))
        nexts.append((current[0],current[1]+1,current[2],current[3]+1))
        for x in nexts:
            if x in cache:
                brightness = cache[x]
            else:
                brightness = np.average(np.array(diff.crop(x)))
                cache[x] = brightness
            if brightness > center_brightness/2 and x not in visited:
                queue.append(x)
        print(get_manhattan_hull(visited))
    return get_manhattan_hull(visited)
'''
def spin_hero():
    bounding_box = (1100,0,2900-1100,2000-0)
    #prev_im = pyautogui.screenshot(region=bounding_box)
    ims = []
    for i in range(50):
        pyautogui.moveTo(1300,100)
        pyautogui.mouseDown()
        pyautogui.move(500,0,1)
        pyautogui.mouseUp()
        im = pyautogui.screenshot()
        ims.append(im)
    return ims
if __name__ == "__main__":
    pyautogui.FAILSAFE = False 
    click_hero_gallery()
    time.sleep(2)
    #for hero in tank + damage + support:
    for hero in support[8:10]:
        click_hero(hero)
        time.sleep(1)
        click_skins()
        time.sleep(1)
        #pyautogui.press("up")
        ims = []
        for i in range(25):
            ims.extend(spin_hero())
            pyautogui.press("down")
        pickle.dump(ims, open('Y:\\pickles\\{0}.p'.format(hero),'wb'))
        pyautogui.press('esc')
        time.sleep(1)
        pyautogui.press('esc')
        time.sleep(1)
