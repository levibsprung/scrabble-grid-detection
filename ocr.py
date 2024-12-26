# Scrabble solver

from operator import inv
from re import X
import cv2
import numpy as np
import imutils
from solver import *
from PIL import Image
from os import listdir
from os.path import isfile, join
from config import *
from slid import pSLID, SLID, slid_tendency #== step 1
#from paddleocr import PaddleOCR,draw_ocr

import bisect
import random
import math
import heapq
import statistics
import string
import copy
import datetime
import time
import base64
import io

def get_surrounding_boxes(x,y,dist=1):
    sb = []
    if x>0:
        sb.append((x-dist,y))
    if x<14:
        sb.append((x+dist,y))
    if y>0:
        sb.append((x,y-dist))
    if y<14:
        sb.append((x,y+1))
    return sb


def mse(img1, median_color):
    h, w, channel = img1.shape
 
    blank_img = np.ones((h, w, channel), dtype='uint8')
    blank_img[:] = median_color
    
    blank_img_hsv = cv2.cvtColor(blank_img, cv2.COLOR_BGR2HSV)
    diff = cv2.absdiff(img1, blank_img_hsv)
    hue_only = diff[:,:,0]
    saturation_only = diff[:,:,1]
    value_only = diff[:,:,2]
    hue_err = np.sum((hue_only / 180) **2)
    saturation_err = np.sum((saturation_only/255)**2)
    value_err = np.sum((value_only/255)**2)
 
    #if white or black
    #if (median_color[1] < 20 and median_color[2] > 230) or median_color[2] < 40:
    #    print("white or black")
    #    mse = hue_err * 0.1 + saturation_err * 0.1 + value_err * 0.1
    #else :
    #    print("color")
    mse = hue_err * 0.8 + saturation_err * 0.1+ value_err * 0.1
 
    mse = mse/(float(h*w))
    return mse

def ranger():
    yield from range(1, 5)
    yield from range(11, 16)

def get_perspective(img, location, height = 720, width = 720):
    """Takes an image and location os interested region.
        And return the only the selected region with a perspective transformation"""

    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[width, 0], [width, height], [0, 0], [0, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def get_InvPerspective(img, masked_num, location, height = 720, width = 720):
    """Takes original image as input"""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def find_board(img):

    height, width, channels = img.shape 

    img = img[int(height * 0.08 ): int(height * 0.92), :]
    #cv2.imshow("cropped", img)
    height, width, channels = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Find red squares
    kernel = np.ones((3,3),np.uint8)

    # Lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # Upper mask (170-180)
    lower_red = np.array([168, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # Join the masks
    red_mask = mask0 | mask1
    
    eroded = cv2.erode(red_mask,kernel,iterations = 8)
    red_dilated = cv2.dilate(eroded,kernel,iterations = 9)
    red_dilated = cv2.erode(red_dilated,kernel,iterations = 5)
    #cv2.imshow("red dilated", red_dilated)

    keypoints = cv2.findContours(red_dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours  = imutils.grab_contours(keypoints)

    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    #cv2.imshow("Contour", newimg)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    newimg3 = img.copy()
    newimg2 = img.copy()

    red_dots = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour,True), True)
        area = cv2.contourArea(approx)
        #print(area)
        if area < 1500 or area > 4500:
            continue
        
        for point in approx:
            red_dots.append((int(point[0][0]),int(point[0][1])))
        cv2.drawContours(newimg3, contour, -1, (0, 255, 0), 3)
        cv2.drawContours(newimg2, approx, -1, (0, 255, 0), 3)
    
    # Find lines 
	# --- 1 step --- find all possible lines ----------------
    segments = pSLID(img)
    raw_lines = SLID(img, segments)
    lines = slid_tendency(raw_lines)   
    lines_img = img.copy() 
    for line in lines:
        #print(line[0][0])
        cv2.line(lines_img,(line[0][0],line[0][1]),(line[1][0],line[1][1]),(255,0,0),2)
    #cv2.imshow("lines_img", lines_img)

    hor = []
    ver = []
    horver_img = img.copy()
    for line in lines:
        #print(line)
        x1,y1,x2,y2 = line[0][0],line[0][1], line[1][0],line[1][1]
        if y2 != y1 and abs((x2-x1)/(y2-y1))<0.2:
            cv2.line(horver_img,(x1,y1),(x2,y2),(255,0,0),2)
            ver.append(line)
        if x2 != x1 and abs((y2-y1)/(x2-x1))<0.2:
            cv2.line(horver_img,(x1,y1),(x2,y2),(255,0,0),2)
            hor.append(line)

    hml = ((1,height/2),(100,height/2))
    vml = ((width/2,1),(width/2,100))

    ver = sorted(ver, key=lambda line: line_intersection(line, hml)[0])
    ver_left = ver[:4]
    ver_right = ver[-4:]
    hor = sorted(hor, key=lambda line: line_intersection(line, vml)[1])
    hor_top = hor[:4]
    hor_bottom = hor[-4:]

    h_ref_pts = []
    v_ref_pts = []
    for v in ver:
        h_ref_pts.append(line_intersection(v, hml)[0])
    for h in hor:
        v_ref_pts.append(line_intersection(h, vml)[1])

    lattice_pts = []
    for v in ver:
        for h in hor:
            lattice_pts.append(line_intersection(v,h))
    lattice_pts.sort()
    print(len(lattice_pts))

    for line in ver_left + ver_right :
        x1,y1,x2,y2 = line[0][0],line[0][1], line[1][0],line[1][1]
        cv2.line(horver_img,(x1,y1),(x2,y2),(0,0,255),2)
    for line in hor_top + hor_bottom:
        x1,y1,x2,y2 = line[0][0],line[0][1], line[1][0],line[1][1]
        cv2.line(horver_img,(x1,y1),(x2,y2),(0,255,0),2)
    candidates = []

    for vl in ver_left:
        for vr in ver_right:
            for ht in hor_top:
                for hb in hor_bottom:
                    # get four corners
                    tl = line_intersection(vl,ht)
                    tr = line_intersection(vr,ht)
                    bl = line_intersection(vl,hb)
                    br = line_intersection(vr,hb)
                    t = math.dist(tl,tr)
                    b = math.dist(bl,br)
                    l = math.dist(tl,bl)
                    r = math.dist(tr,br)

                    if not (l > 0 and r > 0  and t > 0 and b > 0):
                        continue

                    if t/l > 1.05 or l/t > 1.15 or l/r > 1.1 or r/l > 1.1 or (t+b)/(r+l) < 0.9:
                        continue

                    score = 0
                    diff_list = []
                    # Draw the point
                    for x in range(0,16):
                        for y in range(0,16):
                            pts2 = np.float32([tr,tl,bl,br])
                            pts1 = np.float32([[720, 0], [0, 0], [0, 720], [720, 720]])
                            matrix=cv2.getPerspectiveTransform(pts1,pts2)

                            p = (x * 48, y * 48)
                            px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
                            py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
                            p_after = (int(px), int(py))
                            #cv2.circle(split,p_after, 8, (255,0,255), -1)
                            lattice_pts_x = [i[0] for i in lattice_pts]
                            left = bisect.bisect_left(lattice_pts_x,p_after[0]-1)
                            right = bisect.bisect_right(lattice_pts_x,p_after[0]+1)
                            #print(p_after,left,right)
                            #print(lattice_pts_x)

                            #cv2.waitKey(0)
                            for lat in lattice_pts[left:right]:
                                if  (lat[1] + 1) >= p_after[1] >= (lat[1] - 1):
                                    score += 1
                                    break

                    print(score)
                    reject = False

                    #print(t,b,l,r, sd)
                    cnt = np.array([tl,tr,br,bl], dtype=np.int32)
                    print(cnt)
                    if not any(cv2.pointPolygonTest(cnt, rc, False) < 0 for rc in red_dots):
                        score += 100
                    if reject == False:
                        area = cv2.contourArea(cnt)
                        heapq.heappush(candidates, (1/score, (tr,tl,bl,br)))
    
    top_candidates = heapq.nsmallest(3, candidates)
    print(top_candidates)

    #cv2.waitKey(0)
    predicted_board_list = []
    if len(candidates) != 0:
        for score_and_corners in top_candidates:
            r = random.randrange(255)
            g = random.randrange(255)
            b = random.randrange(255)
            corners = score_and_corners[1]
            split = horver_img.copy()
            #print("score")
            #print(score_and_corners[1])
            #cv2.waitKey(0)
            for x in range(0,16):
                for y in range(0,16):
                    pts2 = np.float32([corners])
                    pts1 = np.float32([[720, 0], [0, 0], [0, 720], [720, 720]])
                    matrix=cv2.getPerspectiveTransform(pts1,pts2)
                    p = (x * 48, y * 48)
                    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
                    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
                    p_after = (int(px), int(py))
                    cv2.circle(split,p_after, 8, (255,0,255), -1)
            #cv2.imshow("split", split)
            for corner in score_and_corners[1]:
                #print(a)
                cv2.circle(horver_img,corner, 8, (r,g,b), -1)
            #cv2.imshow("horver_img", horver_img)
            print(corners[0][0] + 2)
            print(corners)
            # improve the chance of getting bottom row and right most column 
            corners2 = ((corners[0][0] + 4, corners[0][1] - 2),(corners[1][0], corners[1][1] - 2),(corners[2][0], corners[2][1]),(corners[3][0] + 4,corners[3][1]))
 
            perspective = get_perspective(img, corners2)
            #cv2.imshow("perspective", perspective)
            
            predicted_board = predict_board(perspective, True)
            predicted_board_list.append((predicted_board, corners2))
            predicted_board = predict_board(perspective, False)
            predicted_board_list.append((predicted_board, corners2))
            #ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
            #result = ocr.ocr(gray, cls=True)
            #for idx in range(len(result)):
            #    res = result[idx]
            #    print(res)

            #from PIL import Image
            #result = result[0]
            #image = Image.open(orig_file).convert('RGB')
            #boxes = [line[0] for line in result]
            #txts = [line[1][0] for line in result]
            #scores = [line[1][1] for line in result]
            #im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
            #im_show = Image.fromarray(im_show)
            #im_show.save('result.jpg')
            print("score")
            print(1/score_and_corners[0])
            #cv2.waitKey(0)
   
    #result = get_perspective(img, four_corners)
    ##cv2.imshow("perspective", result) 
    #cv2.waitKey(0)
    return predicted_board_list


# split the board into 81 individual images
def predict_board(board, flip):
    
    ##cv2.imshow("board",board)
    print("Predict board starts")
    boxes = []
    rows = np.vsplit(board,15)
    for r in (rows):
        cols = np.hsplit(r,15)
        boxes.append(cols)

    predicted_board = []
    tile_color_dict = {}
    sample_loc_dict = {}
    for rn in range(0,15):
        row_prediction = []
        for cn in range(0,15):
            box = boxes[rn][cn]
            color_box = box.copy()
            box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            _, box = cv2.threshold(box,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            if flip:
                box = cv2.bitwise_not(box)
            for x in range(8):
                cv2.floodFill(box,None,(x,0),0)
                cv2.floodFill(box,None,(0,x),0)
                #cv2.floodFill(box,None,(47-x,47),0)
                #cv2.floodFill(box,None,(47,47-x),0)
                cv2.floodFill(box,None,(47,x),0)
                cv2.floodFill(box,None,(x,47),0)
                cv2.floodFill(box,None,(0,47-x),0)
                cv2.floodFill(box,None,(47-x,0),0)
            ref_letter_dir = "ref_tiles/"
            ref_letter_paths = [f for f in listdir(ref_letter_dir) if (isfile(join(ref_letter_dir, f)) and f.find(".jpg") != -1 )]
            match_method = cv2.TM_SQDIFF 
            score_queue = []
            for ref_letter_path in ref_letter_paths:
                min_score = 100000000
                ref_letter_img = cv2.imread(ref_letter_dir + ref_letter_path, cv2.IMREAD_GRAYSCALE)
                #img_display = img_display.astype('uint8')
                ##cv2.imshow("ref", img_display)
                #print(img_display.ndim)
                #print(box.ndim)
                #cv2.waitKey(0)
                print(box.shape)
                print(ref_letter_img.shape)
                result = cv2.matchTemplate(box, ref_letter_img, match_method)
                #print(result)
                #cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
                _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
                print(ref_letter_path[0])
                print(_minVal)
                #print(_maxVal)
                print(minLoc)
                print(ref_letter_img.shape[1])
                _minVal = int(_minVal)
                matchLoc = minLoc
                min_score = min(min_score,_minVal)
                heapq.heappush(score_queue, (min_score, ref_letter_path, matchLoc))

                #box2 = box.copy()
                #cv2.rectangle(box2, matchLoc, (matchLoc[0] + ref_letter_img.shape[1], matchLoc[1] + ref_letter_img.shape[0]), (255,0,0), 2, 8, 0 )
                #cv2.circle(color_box, (matchLoc[0] + ref_letter_img.shape[1] -2, matchLoc[1] + 1), 1, (255,255,0), 2)
                #cv2.rectangle(result, matchLoc, (matchLoc[0] + img_A.shape[0], matchLoc[1] + img_A.shape[1]), (255,0,0), 2, 8, 0 )
                ##cv2.imshow("color_box", color_box)
                ##cv2.imshow("result", result)
                
            print(score_queue)
            prediction = score_queue[0][1][0]
            print("prediction: " + prediction)
            ml = score_queue[0][2]
            mlx = ml[0] + 8
            mly = ml[1] + ref_letter_img.shape[0] - 6
            # Get tile color
            if prediction in list(string.ascii_uppercase):
                bc = color_box[mly][mlx]
                color_box_copy = color_box.copy()
                cv2.rectangle(color_box_copy, ml, (ml[0] + ref_letter_img.shape[1], ml[1] + ref_letter_img.shape[0]), (255,0,0), 2, 8, 0 )
                cv2.circle(color_box_copy, (mlx, mly), 1, (255,255,0), 2)
                #cv2.imshow("color_box", color_box_copy)
                #cv2.imshow("box", box)
                #print(box)

                #cv2.waitKey(0)
                print(bc)
                tile_color_dict[(rn,cn)] = bc
                sample_loc_dict[(rn,cn)] = (mlx, mly)
       
            row_prediction.append(prediction)

            #boxes.append(a)
        print(row_prediction)
        predicted_board.append(row_prediction)
    
    print(tile_color_dict)

    #detect false positive of recognizing a tile
    #median_color_all = np.median(np.array(list(tile_color_dict.values())),axis = 0)
    #print(median_color_all)
    #cv2.waitKey(0)
    #for tile_coord in tile_color_dict:
    #    h, s, v = tile_color_dict[tile_coord]
    #    print(tile_coord)
    #    print(h,s,v)
    #    err = abs(median_color_all[0]-h)/180 * 0.8 + abs(median_color_all[1]-h)/255 * 0.1 + abs(median_color_all[2]-v)/255 * 0.1
    #    print(err)
    #    #cv2.waitKey(0)
    #    if err > 0.3:
    #        print("wrong detection")
    #        cv2.waitKey(0)
    #        predicted_board[tile_coord[0]][tile_coord[1]] = "+"
        

    blank_rank = []
    for i in range(0,15):
        for j in range(0,15):
            if predicted_board[i][j] not in list(string.ascii_uppercase):
                sb = get_surrounding_boxes(i,j)
                surrounding_letters = []
                for k in sb:
                    if predicted_board[k[0]][k[1]] in list(string.ascii_uppercase):
                        surrounding_letters.append(k)
                if not surrounding_letters:
                    continue
                surrounding_letter_colors = []
                for q in surrounding_letters:
                    letter_color = tile_color_dict[(q[0],q[1])]
                    print(letter_color)
                    print(q)
                    surrounding_letter_colors.append(letter_color)
                    #cv2.waitKey(0)
                blank_color_nparray = np.array(surrounding_letter_colors)
                median_color = np.median(blank_color_nparray,axis = 0)
                print(median_color)

                box = boxes[i][j]
                box = box[9:39,9:39,:]
                box_hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
                diff = mse(box_hsv, median_color)
                #cv2.imshow("test box", box)
                print((i,j))
                print(diff)
                #cv2.waitKey(0)
                blank_rank.append((diff, (i, j)))
    blank_rank.sort()
    print(blank_rank)
    print("blank positions")

    if len(blank_rank) >= 2:
        print(blank_rank[0])
        print(blank_rank[1])
        if blank_rank[0][0] < 0.01: 
            predicted_board[blank_rank[0][1][0]][blank_rank[0][1][1]] = '?'
        if blank_rank[1][0] < 0.01:
            predicted_board[blank_rank[1][1][0]][blank_rank[1][1][1]] = '.'
    
    for row in predicted_board:
        row_text = ""
        for col in row:
            row_text += col
        print(row_text)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(boxes)
    print("Predict board ends")
    #cv2.waitKey(0)
    return predicted_board

def displayLetters(img, numbers, color=(0, 255, 0)):
    """Displays 225 letters in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1]/15)
    H = int(img.shape[0]/15)
    for i in range (15):
        for j in range (15):
            if numbers[(j*15)+i] !=0:
                cv2.putText(img, str(numbers[(j*15)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

def evaluate_predicted_board(pb):
    print(pb)
    board = pb[0]
    valid_shape = check_valid_shape(board)
    island_count = numIslands(board)
    eval = 0
    words = []
    x = 0
    for x in range(0,15):
        y = 0
        word = ""
        while y < 15:
            if is_tile(board[x][y]):
                word += board[x][y]
            else :
                if len(word) >= 2:
                    words.append(word)
                word = ""
            y += 1
            if y == 15 and len(word) >= 2:
                words.append(word)

    y = 0
    for y in range(0,15):
        x = 0
        word = ""
        while x < 15:
            if is_tile(board[x][y]):
                word += board[x][y]
            else :
                if len(word) >= 2:
                    words.append(word)
                word = ""
            x += 1
            if x == 15 and len(word) >= 2:
                words.append(word)
    print(words)

    invalid_words = []
    words_with_first_blank = []
    words_with_second_blank = []
    for w in words:
        if not any(char in ['?','.'] for char in w):
            if w not in wordlist:
                invalid_words.append(w)
        else :
            if '?' in w:
                words_with_first_blank.append(w)
            if '.' in w:
                words_with_second_blank.append(w)
    
    print(words_with_first_blank)
    print(words_with_second_blank)
    first_blank = ''
    second_blank = ''

    blank_solution = False

    if len(words_with_second_blank) > 0:
        for b1 in list(string.ascii_uppercase):
            for b2 in list(string.ascii_uppercase):
                words_with_blank_replaced = []
                #print(words_with_first_blank + words_with_second_blank)
                for wwb in words_with_first_blank + words_with_second_blank:
                    wwb_temp = wwb.replace('?',b1).replace('.',b2)
                    words_with_blank_replaced.append(wwb_temp)
                #print("replaced words")
                #print(b1)
                #print(b2)
                #print(words_with_blank_replaced)

                if all(wwbr in wordlist for wwbr in words_with_blank_replaced):
                    first_blank = b1
                    second_blank = b2
                    blank_solution = True
    else:
        for b1 in list(string.ascii_uppercase):
            words_with_blank_replaced = []
            for wwb in words_with_first_blank:
                wwb_temp = wwb.replace('?',b1)
                words_with_blank_replaced.append(wwb_temp)
            if all(wwbr in wordlist for wwbr in words_with_blank_replaced):
                first_blank = b1
                blank_solution = True

    print(invalid_words)
    print(first_blank)
    print(second_blank)

    if blank_solution:
        for x in range(0,15):
            for y in range(0,15):
                if board[x][y] == '?':
                    board[x][y] = first_blank.lower()
                if board[x][y] == '.':
                    board[x][y] = second_blank.lower()
    print(board)

    
    #cv2.waitKey()
    return invalid_words, board, valid_shape, island_count

def check_valid_shape(board):
    return is_tile(board[7][7])
    #if numIslands(grid = board) > 1:
    #    return False

def numIslands(board):
    grid = copy.deepcopy(board)
    if not grid:
        return 0
    num_islands = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if is_tile(grid[i][j]):
                dfs(grid, i, j)
                num_islands += 1
    return num_islands

def dfs(grid, r, c):
    if (
        r < 0
        or c < 0
        or r >= len(grid)
        or c >= len(grid[0])
        or not is_tile(grid[r][c])
    ):
        return
    grid[r][c] = " "
    dfs(grid, r - 1, c)
    dfs(grid, r + 1, c)
    dfs(grid, r, c - 1)
    dfs(grid, r, c + 1)

def is_tile(s):
    return s in list(string.ascii_uppercase) + ["?"] + ["."]

def solveByFilename(filename):
    img = cv2.imread(filename)
    return solve(img)

def solve(img):
    ct = datetime.datetime.now()
    file_path = join("/tmp", str(ct))
    f = open(file_path, "a")
    #orig_file = rel_base_path + file
    
    
    #img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #img_test = Image.open(file)
    #img = np.array(img_test)
    #pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.1/bin/tesseract'
    # extract board from input image
    predicted_board_list = find_board(img)
    eval = []
    for pb in predicted_board_list:
        invalid_words, board, valid_shape, island_count = evaluate_predicted_board(pb)
        if valid_shape and (island_count < 3):
            eval.append((len(invalid_words), invalid_words, board, valid_shape, island_count))
    eval.sort()
    # best eval
    best_eval_dict = {}
    if len(eval) > 0:
        best_eval = eval[0]
        print("no. of invalid words")
        print(str(best_eval[0]))
        print(str(best_eval[1]))
        print(str(best_eval[3]))
        print(str(best_eval[4]))
        for row in best_eval[2]:
            row_text = ""
            for col in row:
                row_text += col
            print(row_text, file=f)
        best_eval_dict["invalid_words"] = best_eval[1]
        best_eval_dict["best_board"] = best_eval[2]
    else :
        print("no solutions")
        best_eval_dict = { "best_board": ""}
    #print(filename + " ends",file=f)
    #cv2.waitKey(0)
    print("summary", file=f)
    print(str(best_eval_dict),file=f)
    f.close()
    #cv2.destroyAllWindows()

    return best_eval_dict

def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    img_b64dec = base64.b64decode(image_bytes)
    img_byteIO = io.BytesIO(img_b64dec)
    #image = Image.open(img_byteIO)
    img = cv2.imdecode(np.frombuffer(img_byteIO.read(), np.uint8), 1)
    return solve(img)
# load dictionary
#wordlist =  set(open('csw19.txt').read().split())
#print(wordlist)

# Read image
wordlist =  set(open('csw19.txt').read().split())
rel_base_path = "samples/temp/"
#mypath = "/Users/carson/Downloads/sudoku-solver-python/" + rel_base_path
#files = [f for f in listdir(rel_base_path) if isfile(join(rel_base_path, f))]
#random.shuffle(files)
files = []
#files = ["f92583e9-178a-45a0-a2e3-d54890eef51a.jpg"]
# oblique angle failed onlyfiles = ["c51347c6-8dc5-446a-ab62-7a3f9f8da538.jpg"]
#/Users/carson/Downloads/sudoku-solver-python/Photos-001/temp/5ea4add7-2cd6-4615-86d6-6f40a20c1c8d.jpg
#/Users/carson/Downloads/sudoku-solver-python/Photos-001/temp/8922a3cf-64dc-4b54-ae62-392f725cf41e.jpg
#83ae0c39-f1e2-433a-a95f-2bea1c2601d6.jpg angle issue
#onlyfiles = ["9ed97c67-d7a2-4444-9014-a361cc166af0.jpg"]
#checking missing light tiles 84cad9df-6439-4c26-a69e-5834ef97e591.jpg
#white tiles Photos-001/temp/fb4bb776-78ea-4d16-afbf-c5cd36b4de9e.jpg
#glare detects letters /Users/carson/Downloads/sudoku-solver-python/Photos-001/temp/0bcc9d8d-4a7e-497f-a7b4-b5627ff72624.jpg

ct = datetime.datetime.now()
file_path = join("/tmp", str(ct))
try:
   
     for file in files:
         f = open(file_path, "a")
         orig_file = rel_base_path + file
         print(orig_file + " starts",file=f)
         img = cv2.imread(rel_base_path + file) 
         predicted_board_list = find_board(img)
         eval = []
         for pb in predicted_board_list:
             invalid_words, board, valid_shape, island_count = evaluate_predicted_board(pb)
             if valid_shape and (island_count <= 2):
                 eval.append((len(invalid_words), invalid_words, board, valid_shape, island_count))
         eval.sort()
         # best eval
         if len(eval) > 0:
             best_eval = eval[0]
             print("no. of invalid words")
             print(str(best_eval[0]),file=f)
             print(str(best_eval[1]),file=f)
             print(str(best_eval[3]),file=f)
             print(str(best_eval[4]),file=f)
             for row in best_eval[2]:
                 row_text = ""
                 for col in row:
                     row_text += col
                 print(row_text, file=f)
         else :
             print("no solutions")
         print(rel_base_path + file + " ends",file=f)
         #cv2.waitKey(0)
         f.close()
         #gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
         #print(gray.shape)
         #rois = split_boxes(gray)
     ##cv2.imshow("Input image", img)
     ##cv2.imshow("Board", board)
   
except Exception as e:
     print(repr(e))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
