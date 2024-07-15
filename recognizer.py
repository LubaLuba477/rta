import cv2
import numpy as np
import mss
import time


# Function to capture screen
def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if region is None else region
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Function to perform template matching
def match_card_template(screen_img, template_img):
    gray_screen = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)    
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    # Edges = cv2.Canny(gray_template, 50, 150)

    # Res is 2D array each element is a reps each result of the coeff correlation method
    res = cv2.matchTemplate(gray_template, gray_screen, cv2.TM_CCOEFF_NORMED)
    return res


template_filenames =  [
    '2c.png', '3c.png', '4c.png', '5c.png', '6c.png', '7c.png', '8c.png', '9c.png', 'Tc.png', 'Jc.png', 'Qc.png', 'Kc.png', 'Ac.png',
    '2d.png', '3d.png', '4d.png', '5d.png', '6d.png', '7d.png', '8d.png', '9d.png', 'Td.png', 'Jd.png', 'Qd.png', 'Kd.png', 'Ad.png',
    '2h.png', '3h.png', '4h.png', '5h.png', '6h.png', '7h.png', '8h.png', '9h.png', 'Th.png', 'Jh.png', 'Qh.png', 'Kh.png', 'Ah.png',
    '2s.png', '3s.png', '4s.png', '5s.png', '6s.png', '7s.png', '8s.png', '9s.png', 'Ts.png', 'Js.png', 'Qs.png', 'Ks.png', 'As.png']

flag_filenames = ['noflop.png', 'noflop2.png', 'noflop3.png', 'noflop4.png','noflop5.png']

flag_table = ['table_flag2.png', 'table_flag3.png', 'table_flag1.png', 'table_flag4.png', 'table_flag5.png', 'table_flag6.png', 'table_flag7.png']


def new_card(screen_img, card_templates, treshold, detected_cards):
    for card_name, template_img in card_templates.items():
        res = match_card_template(screen_img, template_img)            
        loc = np.where(res >= treshold)
        
        if np.any(loc[0]):  # Check if any matches are found
            if card_name not in detected_cards:
                # print(card_name)
                detected_cards.add(card_name)            
                return card_name            
    
    return None  # No new card detected


def is_flag_on(screen_img, flags, treshold):    
    for flag, template_img in flags.items():
        res = match_card_template(screen_img, template_img)   
        loc = np.where(res >= treshold)             
        if loc[0].size > 0:  # Check if there are any matches       
            return True            
    return False


# Load all template images
card_templates = {}
for filename in template_filenames:
    template_path = f'cards_cropped/{filename}'
    template_img = cv2.imread(template_path)
    print(filename)
    if template_img is None:
        print(f"Error: Could not load template image {template_path}")
    else:
        card_templates[filename.split('.')[0]] = template_img

flop_flags = {}
for filename in flag_filenames:
    template_path = f'flag_img/{filename}'
    template_img = cv2.imread(template_path)
    if template_img is None:
        print(f"Error: Could not load template image {template_path}")
    else:
        flop_flags[filename.split('.')[0]] = template_img
        

table_flags = {}
for filename in flag_table:
    template_path = f'flag_table/{filename}'
    template_img = cv2.imread(template_path)
    if template_img is None:
        print(f"Error: Could not load template image {template_path}")
    else:
        table_flags[filename.split('.')[0]] = template_img
        

# Define region of screen capture 
# region = {"top": 100, "left": 100, "width": 800, "height": 600}
region = None

# List to store matches
matched_cards = []
detected_cards = set()

# Main loop for real-time matching
while True:
    # Capture screen
    screen_img = capture_screen(region)
    # Checking if table is open 
    table_found = is_flag_on(screen_img, table_flags, 0.95)  
    print("Table found", table_found)
    # screening the board
    preflop_state = is_flag_on(screen_img, flop_flags, 0.85)
    print("preflop state", preflop_state)
    
    print(new_card(screen_img, card_templates, 0.5, detected_cards))
    
    # sceening the cards
    for card_name, template_img in card_templates.items():
        res = match_card_template(screen_img, template_img)
        threshold = 0.87        
        loc = np.where(res >= threshold)
        
        # a 2D array egyik oszlopat adja vissza, a mar kiszurtbol         
        for pt in zip(*loc[::-1]):
            
            # print(f"Matched template: {card_name}")
            if card_name not in matched_cards:
                matched_cards.append(card_name)      
            print(f"Matched template: {matched_cards}")            
            matched_cards = []
            break

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


