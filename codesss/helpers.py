# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_biggest_face(array_of_faces):
    """
    :param array_of_faces: array of faces returned from detector.detectMultiScale()
    :return: a 1x4 array containing the face with the biggest area (width*height)
    """
    largest_area = 0
    largest_face = None
    for face in array_of_faces:
        area = face[2] * face[3]  # width * height
        if area > largest_area:
            largest_area = area
            largest_face = face
    return [largest_face]


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    """
    :param image:
    :param shape: A list or array of facial landmark points, shape 68 by 2
    :param colors:
    :param alpha:
    :return: hull_dict: a dictionary in which the key is the landmark's name and the value is the list of convex hull
    points for that landmark.
    """
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    hull_dict = {}

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220), (0, 0, 255)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        # print("pts")
        # print(pts)

        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            # hull = np.array(hull).reshape((-1, 1, 2)).astype(np.uint32)
            hull = [np.array(hull[:, 0, :]).astype(np.int32)]
            # print("hull")
            # print(hull)
            # this takes a python list of numpy signed integer (other formats may fail)
            # arrays, where each array is of shape (n, 2)
            cv2.drawContours(overlay, hull, -1, colors[i], -1)
            hull_dict[name] = hull

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output, hull_dict


def get_image_dict_from_hulls(frame, hull_dict):
    """
    Gets a dictionary of alpha-masked, cropped facial regions from a picture and a dictionary of convex hulls
    :param frame:
    :param hull_dict:
    :return: The dictionary of alpha-masked, cropped facial regions
    """
    face_region_names = hull_dict.keys()
    image_dict = {}
    height, width, channels = tuple(frame.shape)  # H x W x Channels
    scaling_factors = {'left_eye': 2.2, 'right_eye': 2.2, 'mouth': 1.8}
    for name in ['left_eye', 'right_eye', 'mouth']:
        hull = hull_dict[name].copy()[0]  # 68 x 2
        max_x = np.max(hull[:, 0])
        min_x = np.min(hull[:, 0])
        w = max_x - min_x
        min_x = max(0, int(min_x - w * 0.1))
        max_x = min(width, int(max_x + w * 0.1))
        max_y = np.max(hull[:, 1])
        min_y = np.min(hull[:, 1])
        h = max_y - min_y
        min_y = max(0, int(min_y - h * 0.1))
        max_y = min(height, int(max_y + h * 0.1))
        # print([min_x, max_x, min_y, max_y])
        cropped_img = frame[min_y: max_y, min_x: max_x, :]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)


        alpha_mask = np.zeros(cropped_img.shape[0:2]).astype(np.uint8)
        # hull = (hull - np.array([[min_x, min_y]]))*1.5  # adjust the convex hull points to the cropped image
        # hull = hull.astype(np.int32)
        hull = hull - np.array([[min_x, min_y]])
        cv2.drawContours(alpha_mask, [hull], -1, 255, -1)
        # circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # alpha_mask = cv2.dilate(alpha_mask, circular_kernel, iterations=1)
        blur_area = (5, 5)
        alpha_mask = cv2.GaussianBlur(alpha_mask, blur_area, 0)  # blur the edges of the alpha mask
        # make some facial features bigger
        if name in scaling_factors:
            scale = scaling_factors[name]
            cropped_img = cv2.resize(cropped_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            alpha_mask = cv2.resize(alpha_mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        alpha_mask[alpha_mask > 0] = 255

        cropped_img[..., 3][alpha_mask == 0] = 0  # set alpha to 0 for pixels not belonging to inside the facial region
        image_dict[name] = cropped_img
        # cv2.imshow('region', cropped_img)
        # cv2.imshow('alpha', alpha_mask)
    return image_dict


def paste_alpha_img(source_img, overlay, paste_center):
    overlay_height, overlay_width = tuple(overlay.shape[0:2])
    paste_top_left = [paste_center[0] - overlay_width // 2, paste_center[1] - overlay_height // 2]
    overlaid_img = overlay_transparent(source_img, overlay, paste_top_left[0], paste_top_left[1])
    # source_roi = source_img[paste_top_left[1]:paste_top_left[1] + overlay_height,
    #              paste_top_left[0]: paste_top_left[0] + overlay_width, :]
    # source_roi[overlay[..., 3] > 0] = overlay[..., :3]
    # source_roi =
    # cv2.imshow('filtered', overlaid_img)
    return overlaid_img
    # return source_img

def overlay_transparent(background, overlay, x, y):
    background = background[..., :3]
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def overlay_transparency(target_image, patch, startX, startY):
    """
    :param patch: Image to be overlayed in RGBA 4-channel representation
    """
    patch = np.copy(patch)
    max_patch_height = target_image.shape[0] - max(0, startY)
    max_patch_width = target_image.shape[1] - max(0, startX)
    # if patch is too tall (shouldn't happen in theory)
    target_width = target_image.shape[1] - startX
    if patch.shape[0] > max_patch_height:
        patch = patch[:max_patch_height, ...]
        print("g_transfer.py WARNING hand patch is too tall (should't happen)")

    # the fingertip is in the frame but left part of the hand is not
    if startX < 0:
        offset = - startX
        # max_patch_width -= startX + 1
        patch = patch[:, offset:]
    # the fingertip is in the frame but right part of the hand is not
    if startX > target_image.shape[1]:
        offset = startX - target_image.shape[1]
        patch = patch[:, :-offset]
        raise Exception("Error: Upper-left corner of hand image is to the right of fingertip location!")
    patch = patch[:, :max_patch_width]  # crop right part of the hand that's out of target image's right bound

    # startX = max(0, startX)
    # startY = max(0, startY)
    # target_area = target_image[startY: min(target_image.shape[0] - 1, startY + patch.shape[0]),
    #               startX: min(target_image.shape[1] - 1, startX + patch.shape[1]), ...]
    # print('target size' + str(target_area.shape))
    # # if patch is too large (pasting out of bounding), crop the patch image (hand)
    # if target_area.shape[0] != patch.shape[0] or target_area.shape[1] != patch.shape[1]:
    #     patch = patch[:target_area.shape[0], :target_area.shape[1], ...]
    #     print("hand overlay too large, cropping ...")
    #     # patch = patch[0:target_area.shape[0], 0:target_area.shape[1]]
    # print("Target upper-left point: " + str((startX, startY)))
    startX = max(startX, 0)  # since we've cropped the patches correctly, we can set negative indicies to 0
    startY = max(startY, 0)
    target_area = target_image[startY: startY + patch.shape[0],
                  startX: startX + patch.shape[1]]
    mask = patch[..., 3]
    # print("Target upper-left point after crop: " + str((startX, startY)))
    # print("patch size: " + str(patch.shape))
    # print("target area size: " + str(target_area.shape))
    target_area[..., [0, 1, 2]][mask == 255] = patch[..., [0, 1, 2]][mask == 255]
    target_image[startY:startY + patch.shape[0], startX: startX + patch.shape[1]] = target_area
    return target_image


def center_to_top_left(center_x, center_y, img):
    top = int(center_y - img.shape[0] / 2)
    left = int(center_x - img.shape[1] / 2)
    return top, left

def get_facial_landmark_dict():
    return FACIAL_LANDMARKS_IDXS


def visualize_eye_img(eye_img, use_activation=True):
    original_img = eye_img
    height, width = (eye_img.shape[0], eye_img.shape[1])
    center_x, center_y = (width // 2, height // 2)
    eye_img = eye_img.copy()
    eye_img[eye_img[..., 3] != 255] = 255  # mask non-eye pixels
    # eye_img[..., 2] = np.maximum(eye_img[..., 2] * 3, 255)  # make the red channel brighter
    eye_img = 255 - eye_img  # negate the image so the iris is bright rather than dark
    if use_activation:
        eye_img[eye_img <= np.mean(eye_img) * 1.3] = 0
    # eye_img = eye_img.astype(np.uint8)
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGRA2GRAY)
    cv2.imshow('negated and truncated eye image ', eye_img)
    eye_img = np.float32(eye_img)
    filter_height = int(min(width * 0.6, height * 0.6))
    filter_height = 8
    # the iris/pupil are brighter in the negated image of the eye, so we use a filter to determine its location
    conv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_height, filter_height)).astype(np.float32)
    conv_kernel -= 0.5
    # conv_kernel /= (filter_height * filter_height)  # normalize the sum of the kernel
    eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    if use_activation:
        eye_img[eye_img <= np.mean(eye_img) * 1.3] = 0
    cv2.imshow('1st conv + truncation ', eye_img)
    eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    cv2.imshow('2nd conv ', eye_img)
    # eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    # eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    # eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    top_20_max_pixels = eye_img.flatten().argsort()[-20:]
    # print(top_20_max_pixels)
    pupil_loc = np.array([0, 0])
    for p_idx in top_20_max_pixels:
        loc = np.unravel_index(p_idx, eye_img.shape)
        pupil_loc += loc
    pupil_loc //= 20
    # pupil_loc = np.unravel_index(np.argmax(eye_img), eye_img.shape)  # get the brightest point in filtered img
    pupil_loc = tuple(reversed(pupil_loc))  # (y, x) -> (x,y)
    eye_img = (eye_img - np.min(eye_img)) / (np.max(eye_img) - np.min(eye_img))
    # print(pupil_loc)
    original_img = cv2.circle(original_img, pupil_loc, 3, (255, 0, 0, 255), -1)
    # print(original_img.shape)
    cv2.imshow('Pupil location', original_img)

def eye_to_cartoon_eye(eye_img, eye_background_img, eyeball_img, left_eye=True, final_img=None, left_eye_loc=None, right_eye_loc=None):
    """
    :param eye_img: Image of an eye. A RGBA image in which non-eye regions have an alpha of 0, and pixels belong to the
    eye has an alpha of 255.
    :param left_eye: If true, the output will look like the left eye (on the right of face in a picture)
    :return: Cartoonized image of the eye
    """
    original_img = eye_img
    height, width = (eye_img.shape[0], eye_img.shape[1])
    center_x, center_y = (width // 2, height // 2)
    eye_img = eye_img.copy()
    eye_img[eye_img[..., 3] != 255] = 255  # mask non-eye pixels
    # eye_img[..., 2] = np.maximum(eye_img[..., 2] * 3, 255)  # make the red channel brighter
    eye_img = 255 - eye_img  # negate the image so the iris is bright rather than dark
    eye_img[eye_img <= np.mean(eye_img) * 1.3] = 0
    # eye_img = eye_img.astype(np.uint8)
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGRA2GRAY)
    cv2.imshow('eye_gray', eye_img)
    eye_img = np.float32(eye_img)
    filter_height = int(min(width * 0.6, height * 0.6))
    filter_height = 8
    # the iris/pupil are brighter in the negated image of the eye, so we use a filter to determine its location
    conv_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_height, filter_height)).astype(np.float32)
    conv_kernel -= 0.5
    # conv_kernel /= (filter_height * filter_height)  # normalize the sum of the kernel
    eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    eye_img[eye_img <= np.mean(eye_img) * 1.3] = 0
    eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    # eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    # eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    # eye_img = cv2.filter2D(eye_img, -1, conv_kernel)
    top_20_max_pixels = eye_img.flatten().argsort()[-20:]
    # print(top_20_max_pixels)
    pupil_loc = np.array([0,0])
    for p_idx in top_20_max_pixels:
        loc = np.unravel_index(p_idx, eye_img.shape)
        pupil_loc += loc
    pupil_loc //= 20
    # pupil_loc = np.unravel_index(np.argmax(eye_img), eye_img.shape)  # get the brightest point in filtered img
    pupil_loc = tuple(reversed(pupil_loc))  # (y, x) -> (x,y)
    eye_img = (eye_img - np.min(eye_img)) / (np.max(eye_img) - np.min(eye_img))
    # print(pupil_loc)
    original_img = cv2.circle(original_img, pupil_loc, 3, (255, 0, 0, 255), -1)
    # print(original_img.shape)
    cv2.imshow('eye_img', original_img)
    cv2.imshow('eye_conv', eye_img)

    dx_ratio = (pupil_loc[0] - center_x) / width
    dy_ratio = (pupil_loc[1] - center_y) / height
    print((dx_ratio, dy_ratio))
    # canvas_width, canvas_height = tuple(eye_background_img.shape[:2])
    # eyeball_center_x = int(canvas_width / 2 + canvas_width*dx_ratio)
    # eyeball_center_y = int(canvas_height / 2 + canvas_height*dy_ratio)
    # top, left = center_to_top_left(eyeball_center_x, eyeball_center_y, eyeball_img)
    # top, left = (eyeball_center_y, eyeball_center_x)
    # top += eye_loc[1]
    # left += eye_loc[0]
    # print((top, left))
    # eye = overlay_transparent(eye_background_img, eyeball_img, left, top)
    eye = overlay_transparent(final_img, eyeball_img, int(min(512, max(0, left_eye_loc[0] + dx_ratio * 80 - 48))),
                              int(min(512, max(0, left_eye_loc[1] + dy_ratio * 80 - 48))))
    eye = overlay_transparent(final_img, eyeball_img, int(min(512, max(0, right_eye_loc[0] + dx_ratio * 80 - 48))),
                              int(min(512, max(0, right_eye_loc[1] + dy_ratio * 80 - 48))))
    # eye = overlay_transparent(final_img, eye_background_img, int(min(512, max(0, eye_loc[0] - 48))),
    #                           int(min(512, max(0, eye_loc[1] - 48))))
    eye = cv2.circle(eye, left_eye_loc, 40, 32, 6)
    eye = cv2.circle(eye, right_eye_loc, 40, 32, 6)
    # cv2.imshow('cartoon_eye', eye)
    return eye

