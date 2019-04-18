# imgFunctions are functions to assist with the preprocessing of images
import configparser
import numpy as np
import cv2
import sys

debug = False
saveSteps = True
np.set_printoptions(threshold=sys.maxsize)
config = configparser.ConfigParser()
config.read('cardbot.config')
orientation = int(config['TABLE']['orientation_to_dealer'])
upper_B = int(config['TABLE']['table_color_upper_B'])
upper_G = int(config['TABLE']['table_color_upper_G'])
upper_R = int(config['TABLE']['table_color_upper_R'])
lower_B = int(config['TABLE']['table_color_lower_B'])
lower_G = int(config['TABLE']['table_color_lower_G'])
lower_R = int(config['TABLE']['table_color_lower_R'])
upper=(upper_B, upper_G, upper_R)
lower=(lower_B, lower_G, lower_R)


def cropcenter(img):
    '''
    :param img: The image to be cropped to train_height and train_width
    :return: return the cropped image
    '''
    img_height = img.shape[0]
    img_width = img.shape[1]
    train_height = int(config['CARDS']['image_height'])
    train_width = int(config['CARDS']['image_width'])

    if img_width > train_width:
        trim = int((img_width - train_width) / 2)
        xmin = trim
        xmax = img_width - trim
    else:
        xmin = 0
        xmax = img_width
    if img_height > train_height:
        trim = int((img_height - train_height) / 2)
        ymin = trim
        ymax = img_height - trim
    else:
        ymin = 0
        ymax = img_height
    centercrop = img[ymin:ymax, xmin:xmax].copy()

    return centercrop


def cropbycolor():

    # Find the ROI in target color - presumably this is the card table
    mask = cv2.inRange(img, lower, upper)

    # Crop out everything except that ROI
    ys, xs = np.nonzero(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    cropped = img[ymin:ymax, xmin:xmax].copy()

    if debug:
        cv2.imshow("cropped_roi", cropped)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return cropped

def rotateimage(img, degree):
    # re-orient pictures to be read left to right
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    # rotate dealer image by -90 degress
    M = cv2.getRotationMatrix2D(center, -90, 1.0)
    rotate = cv2.warpAffine(img, M, (w, h))

    return rotate


#  Seperate the Dealer's hand from the player's hand
def seperatedealer(cropped):
    '''
    :param cropped: raw image, cropped to the appropriate color ROI
    :return: dealerhand: image of just the dealer's hand
    :return: playerhands image of all the players hands
    '''
    # section image based on orientation
    img_height = cropped.shape[0]
    img_width = cropped.shape[1]
    y_midpoint = int(img_height/2)
    x_midpoint = int(img_width/2)
    if orientation is 1:  # Orientation = 1 = Dealer is Top Half of image
        dealerhand = cropped[0:y_midpoint, 0:img_width].copy()
        playerhands = cropped[y_midpoint:img_height, 0:img_width].copy()
        # No re-orientation required
    elif orientation is 2:  # Orientation = 2 = Dealer is Right Half of image
        dealerhand = cropped[0:img_height, x_midpoint:img_width].copy()
        dealerhand = rotateimage(dealerhand, -90)
        playerhands = cropped[0:img_height, 0:x_midpoint].copy()
        playerhands = rotateimage(playerhands, -90)
    elif orientation is 3:  # Orientation = 3 = Dealer is Bottom half of image
        dealerhand = cropped[y_midpoint:img_height, 0:img_width].copy()
        dealerhand = rotateimage(dealerhand, 180)
        playerhands = cropped[0:y_midpoint, 0:img_width].copy()
        playerhands = rotateimage(playerhands, 180)
    else:  # Orientation = 4 = dealer is Left half of image
        dealerhand = cropped[0:img_height, 0:x_midpoint].copy()
        dealerhand = rotateimage(dealerhand, 90)
        playerhands = cropped[0:img_height, x_midpoint:img_width].copy()
        playerhands = rotateimage(playerhands, 90)

    if debug:
        cv2.imshow("dealerhand", dealerhand)
        cv2.waitKey()
        cv2.imshow("playerhands", playerhands)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return dealerhand, playerhands


def readdealerhand(dealerhand):
    # Is image 720 x 720 or smaller?
    img_height = dealerhand.shape[0]
    img_width = dealerhand.shape[1]
    crop_from_edge = bool(config['CARDS']['dealer_crop_from_edge'])
    train_height = int(config['CARDS']['image_height'])
    train_width = int(config['CARDS']['image_width'])
    if img_width > train_width:
        trim = int((img_width - train_width) / 2)
        xmin = trim
        xmax = img_width - trim
    else:
        xmin = 0
        xmax = img_width

    if img_height > train_height:
        # do we want middle pixels? or start from edge of image
        if crop_from_edge:
            dealerhand = dealerhand[0:train_height, xmin:xmax].copy()
        else: # center the 720px and crop evenly from all sides
            ytrim = int((img_height - train_height) / 2)
            ymin = ytrim
            ymax = img_height - ytrim
            dealerhand = dealerhand[ymin:ymax, xmin:xmax].copy

    if debug:
        cv2.imshow("read dealer hand", dealerhand)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return dealerhand


def readplayerhands(img_playerhands):
    # How many players?
    playercount = int(config['TABLE']['num_of_players'])

    col_player_hands = []
    if playercount is 1:
        isoplayerhand = cropcenter(img_playerhands)
        col_player_hands.append(isoplayerhand)
        if debug:
            cv2.imshow("player 1 hand", isoplayerhand)
            cv2.waitKey()
            cv2.destroyAllWindows()
    elif playercount is 2:
        # divide image into 2 halves - center each half
        img_width = img_playerhands.shape[1]
        img_height = img_playerhands.shape[0]
        midpoint = int(img_width / 2)
        ymin = 0
        ymax = img_height
        xmin = 0
        xmax = midpoint
        player1_hand = img_playerhands[ymin:ymax, xmin:xmax].copy()
        xmin = midpoint
        xmax = img_width
        player2_hand = img_playerhands[ymin:ymax, xmin:xmax].copy()

        # Center Crop
        player1_hand = cropcenter(player1_hand)
        player2_hand = cropcenter(player2_hand)

        col_player_hands.append(player1_hand)
        col_player_hands.append(player2_hand)

        if debug:
            cv2.imshow("player 1 hand", player1_hand)
            cv2.waitKey()
            cv2.imshow("player 2 hand", player2_hand)
            cv2.waitKey()
            cv2.destroyAllWindows()
    else:
        # Divide img_playerhands image evenly by the number of players
        totalwidth = playerhands.shape[1]
        img_height = playerhands.shape[0]
        ea_handwidth = totalwidth / playercount

        # make sure handwidth is an integer (can't divide floats cleanly)
        int_ea_handwidth = int(ea_handwidth)

        playerindex = 0
        for player in range(playercount):
            playerindex += 1
            # from left to right crop out each hand
            ymin = 0
            ymax = img_height
            xmin = (int_ea_handwidth * playerindex) - int_ea_handwidth
            xmax = int_ea_handwidth * playerindex

            playerhand = img_playerhands[ymin:ymax, xmin:xmax]

            # center crop playerhand
            playerhand = cropcenter(playerhand)
            col_player_hands.append(playerhand)
            if debug:
                cv2.imshow("player_" + str(playerindex) + "_hand", playerhand)
                cv2.waitKey()
                cv2.destroyAllWindows()

    return col_player_hands




# read in the raw image
img = cv2.imread('cardshot.jpg')

# crop the image down as close to card table as possible
color_roi = cropbycolor()
if saveSteps:
    cv2.imwrite("crop_to_table.jpg", color_roi)

# Isolate region with dealer's hand
dealerhand, playerhands = seperatedealer(color_roi)
if saveSteps:
    cv2.imwrite("uncropped_dealerhand.jpg", dealerhand)
    cv2.imwrite("all_playerhands.jpg", playerhands)

# Read dealers hand
rdealerhand = readdealerhand(dealerhand)
if saveSteps:
    cv2.imwrite("cropped_dealerhand.jpg", rdealerhand)
# TODO execute Yolo reference here - this happens in the cloud.  Send this image via IoT Topic?

# Read players hand
playerhands = readplayerhands(playerhands)
if saveSteps:
    playerindex = 0
    for hand in playerhands:
        playerindex += 1
        cv2.imwrite("cropped_player" + str(playerindex) + "_hand.jpg", hand)
# TODO execute Yolo reference here - this happens in the cloud.  Send this image via IoT Topic?